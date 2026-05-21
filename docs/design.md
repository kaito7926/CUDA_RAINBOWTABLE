# desrt — design notes

This document explains the math behind the numbers reported by `desrt plan`,
the structure of the chain kernel, the choice of reduction function, and the
roadmap from textbook DES to a bitsliced implementation that would meet the
spec's 10 GH/s end-to-end target.

## 1. Keyspace and charset indexing

The attack covers `key_len = 8` characters drawn from the 36-character set
`"abcdefghijklmnopqrstuvwxyz0123456789"` (lowercase letters + digits). The
keyspace size is

```
N = 36^8 = 2,821,109,907,456  (≈ 2^41.4)
```

Each integer `idx ∈ [0, N)` maps to one ASCII string `s[0..7]` via standard
base-36 expansion. Digit 0..25 → 'a'..'z', digit 26..35 → '0'..'9'. We pack
the string into a 64-bit "DES key" with `s[0]` in the **most-significant
byte** of the uint64. The eight LSBs of each byte are the DES parity bits
and are discarded by PC-1, so the recovered key may not exactly match the
byte string used to produce it; it will however encrypt the plaintext to the
same ciphertext (DES-equivalence).

`include/desrt/common.h`:

```cpp
DESRT_HD inline uint64_t idx_to_key(uint64_t idx);
DESRT_HD inline void     key_to_string(uint64_t key, char out[8]);
DESRT_HD inline uint64_t string_to_idx(const char s[8]);   // UINT64_MAX on bad char
```

> Switching to a different charset is a single point of change in
> `common.h`: update `CHARSET_LEN`, `N_KEYSPACE`, and the two char↔digit
> branches in `idx_to_key` / `string_to_idx`. Nothing else (DES, reduction,
> chain kernel, shards, crack) depends on the charset directly.

## 2. Chain construction

```
start_idx = (LCG_A * chain_id + LCG_B) mod N
idx = start_idx
for r in [0, chain_len):
    key = idx_to_key(idx)
    ct  = DES_ECB_encrypt_block(key, plaintext)
    idx = reduce(ct, r, table_id, N)
endpoint = idx
Record { endpoint, start_idx }
```

`LCG_A = 6364136223846793005`, `LCG_B = 1442695040888963407` — Knuth's MMIX
constants. The LCG runs `mod 2^64` and we take `mod N` afterwards. With ~8 M
draws against `N ≈ 2.82 × 10^12`, accidental start-point collisions are
negligible.

The reduction is

```cpp
uint64_t reduce(uint64_t ct, uint32_t round, uint32_t table_id, uint64_t N) {
    uint64_t m = ct
              ^ splitmix64(round * 0x9E3779B97F4A7C15 ^ table_id * 0xC6BC279692B5C323);
    return splitmix64(m) % N;
}
```

Two reasons to mix `round` and `table_id` in:

* **Round-dependent reduction** keeps chains from merging. Without it, any
  collision between two chains at the same position will propagate forward,
  collapsing one chain into another.
* **Table-id mixing** lets multiple independent tables (different `table_id`)
  cover disjoint slices of the keyspace, so coverage adds rather than
  duplicates.

`splitmix64` was chosen because it has excellent avalanche on small inputs and
is one multiply-plus-shift sequence — cheap on the GPU.

## 3. Coverage model

For a single table:

```
p_one_table ≈ 1 - exp(- chain_len * chains_per_table / N)
```

With `chain_len = 2^20 = 1,048,576` and `chains = 8,100,000`:

```
chain_len * chains / N = 8.493e+12 / 2.821e+12 = 3.011
p_one_table         ≈ 1 - exp(-3.011) ≈ 0.9508
```

This is the upper bound that ignores chain merges. Real tables typically reach
`0.85 × p_one_table` to `0.95 × p_one_table` depending on chain-len. With
multiple tables of different `table_id`,

```
coverage ≈ 1 - (1 - p_one_table)^tables
```

`desrt plan` prints both `p_one_table` and the multi-table figure. The
warning about chain merges is also printed.

## 4. Raw size and shard layout

Each record is 16 bytes:

```cpp
#pragma pack(push, 1)
struct Record { uint64_t endpoint; uint64_t startpoint; };
#pragma pack(pop)
```

8.1 M × 16 B = **~130 MiB** per table. We partition by

```
shard_id = endpoint mod num_shards
```

with `num_shards ∈ {2048, 4096}`. 4096 shards gives ~32 KiB / shard which
fits in RAM trivially during the sort step. The on-disk layout:

```
<root>/raw/shard_NNNNNN.bin       (append-only during build)
<root>/sorted/shard_NNNNNN.bin    (sorted by endpoint after `desrt sort`)
```

`desrt sort` reads each raw shard, sorts in memory, writes the sorted shard,
and deletes the raw file. The sort runs `--jobs N` shards in parallel
(default = `hardware_concurrency()`).

## 5. GPU layout

`cmd_build.cu` runs **one CUDA thread per chain**. The kernel is launched in
batches of `--batch-size` chains, default 65536:

```cpp
__global__ void build_chains_kernel(
    Record* records, uint64_t chain_id_base, uint64_t chains_in_batch,
    uint32_t chain_len, uint32_t table_id, uint64_t plaintext, uint64_t N);
```

Each thread:

1. Computes `start_idx` from `chain_id`.
2. Walks `chain_len` chain steps, each step doing one full DES encryption.
3. Stores `{endpoint, start_idx}` to the batch output array.

Host side:

* One pinned host buffer (`cudaMallocHost`) of `batch_size * sizeof(Record)`.
* `cudaMemcpy` device → pinned host after each kernel.
* `ShardWriter::append_batch(...)` partitions Records into per-shard buffers
  and flushes when each buffer hits `--shard-buffer-kb`.

We use a single CUDA stream and synchronous copy because, with textbook DES,
the kernel dominates by orders of magnitude. There is no value in
double-buffering when the copy is < 0.01% of the loop.

Threads-per-block: 128 by default (`--block-size`). The whole keyspace is
covered by sweeping `chain_id` from 0 to `--chains`. To pick up where a
previous run left off, pass `--start-chain-id`.

## 6. Crack algorithm

For a target ciphertext `CT` we test every chain position `p ∈ [0, L)` as the
position where `ct_p == CT`. The "endpoint candidate" for that hypothesis is

```
y = reduce(CT, p, table_id, N)
for r in [p+1, L):
    y = chain_step(y, r, table_id, P, N)
endpoint_candidate = y
```

The GPU computes one candidate per thread, with `num_targets * chain_len`
threads total (host bound below 4 GiB by capping the candidate buffer to
8 B/thread). On the host:

1. Bucket candidates by `shard = endpoint % num_shards` and sort by
   `(shard, candidate)`.
2. Walk shards in order; load each sorted shard once; binary-search every
   candidate that falls in it.
3. For each match, replay forward from the recorded `startpoint` for `p`
   chain steps, compute `key = idx_to_key(idx_p)`, and check
   `DES(key, plaintext) == CT`. False positives (different chain converging
   to the same endpoint) fail this check.

Step 2 is the I/O-heavy phase. With 4096 shards and a uniform candidate
distribution, every shard is loaded once → ~10 GiB of disk reads for the full
table. Subsequent target groups within a single `desrt crack` invocation
share the load by being merged into the same probe set.

## 7. Performance — where the time goes

The textbook DES kernel is bound by bit-by-bit permutation loops (the `IP`,
`FP`, `E`, `P`, and `PC-2` permutations are each 32–64 iterations of one
shift + one OR). On an L4 (Ada Lovelace, SM 8.9, 7.4 TFLOPS FP32) we expect
something like:

* 50 – 300 MH/s **per GPU**
* 100 – 600 MH/s across both L4s
* full 8.1 M-chain build (×1,048,576 = ~8.5e12 DES ops): tens of minutes to
  a few hours depending on which point in that throughput range you land.
  A useful sanity check is `desrt bench` followed by
  `desrt build --chains <small> --chain-len <small>` to extrapolate.

For the original 62-char keyspace the build numbers were two orders of
magnitude larger (~10 TiB-equivalent of work). The 10 GH/s target from the
spec was sized for that — for the 36-char keyspace it is comfortably met by
the textbook kernel. A bitsliced rewrite is still the natural next step if
you want to push to multi-table coverage or much larger keyspaces:

* 64-way data-parallel S-boxes laid out as gate networks (Matthew Kwan or
  Roman Rusakov style). Each thread processes 64 keys simultaneously by
  treating one register bit per key.
* No bit permutations at runtime — they are absorbed into the wiring of the
  next round at compile time.
* Realistic on L4: 3 – 8 GH/s per GPU.

This implementation is structured so a bitsliced kernel can be dropped in by
replacing `chain_step` (and slicing the chain so 64 keys share one thread's
register file).

## 8. Multi-GPU and resume

The chain index is the only state that matters for resumability. Running

```
desrt build --start-chain-id 0          --chains 311900000 --gpu 0 --out ./tab &
desrt build --start-chain-id 311900000  --chains 311900000 --gpu 1 --out ./tab &
```

splits the work across two GPUs and writes into the same `./tab/raw`
directory. Both processes open shards in append mode; the writes are not
interleaved at byte level because `ShardWriter` does `fwrite` of whole buffer
chunks, and 16-byte records are small enough that the kernel's per-`write`
atomicity holds. After both finish, run `desrt sort --in ./tab` once.

Running both GPUs in one process is not implemented in this revision — it
would need either thread-safe `ShardWriter` (per-shard mutex) or per-GPU
shard subdirs that are merged at sort time. Either is a small change; the
current "two processes" recipe is simpler and exercises the resume path.

## 9. File summary

| File | Role |
|---|---|
| `include/desrt/common.h` | Config constants, `Record`, charset/key helpers, LCG, `splitmix64`, `reduce_idx`. |
| `include/desrt/des.h`    | FIPS-46 DES tables + `encrypt_block` + `chain_step`. |
| `include/desrt/shard.h`  | `ShardWriter`, `sort_shard`, `read_shard`, `find_endpoint`. |
| `include/desrt/cli.h`    | Flag parser. |
| `src/cmd_build.cu`       | Chain kernel and host-side streaming. |
| `src/cmd_crack.cu`       | Candidate kernel and shard-walk lookup. |
| `src/cmd_bench.cu`       | DES KATs and throughput micro-benchmark. |
| `src/cmd_plan.cpp`       | Coverage and time estimates. |
| `src/cmd_sort.cpp`       | Per-shard sort. |
| `src/cmd_make_target.cpp`| Random target generator. |
| `src/cmd_stats.cpp`      | Shard summary statistics. |
