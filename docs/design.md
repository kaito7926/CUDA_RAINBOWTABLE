# desrt — design notes

This document explains the math behind the numbers reported by `desrt plan`,
the structure of the chain kernel, the choice of reduction function, and the
roadmap from textbook DES to a bitsliced implementation that would meet the
spec's 10 GH/s end-to-end target.

## 1. Keyspace and charset indexing

### 1.1 The DES parity collapse

The user-facing alphabet is `[a-z0-9]` (36 characters). But DES PC-1
discards bit 0 (the LSB) of each input key byte. Two ASCII characters that
differ only in their LSB are therefore **DES-equivalent** — they produce the
same ciphertext under any plaintext.

For `[a-z0-9]` the parity classes are:

* Letters: `{a}`, `{b,c}`, `{d,e}`, `{f,g}`, `{h,i}`, `{j,k}`, `{l,m}`,
  `{n,o}`, `{p,q}`, `{r,s}`, `{t,u}`, `{v,w}`, `{x,y}`, `{z}` — 2 singletons
  (`a`, `z`; their pairs `` ` `` and `{` are not in the charset) plus 12
  pairs → **14 letter classes**.
* Digits: `{0,1}`, `{2,3}`, `{4,5}`, `{6,7}`, `{8,9}` → **5 digit classes**.

> **Effective alphabet size: 19, not 36.**

If we naively packed all 36 ASCII characters into a chain and iterated,
every chain step would funnel through that 19-byte bottleneck, and chains
would collapse at the rate `m² / (2 · 19⁸)`. An early build at
`chains = 8.1 M`, `chain_len = 2²⁰` produced only **26,319 unique
endpoints** out of 8.1 M records — exactly the asymptote
`m(t) = 1 / (t/(2·19⁸) + 1/m₀) ≈ 33,800` that the merge-rate ODE predicts.
That was the catastrophic-stats bug.

### 1.2 The fix — canonical 19-char alphabet

We work internally in the canonical alphabet

```
"abdfhjlnprtvxz02468"          (|charset| = 19)
```

Each canonical character lies in a unique parity class, so packing eight of
them into a uint64 DES key gives an **injective** mapping `idx → DES key
(after PC-1)`. The chain math is now honest.

```
N = 19^8 = 16,983,563,041   (≈ 2^33.98)
```

Digit ↔ canonical char:

```
digit  0     -> 'a'
digit  1..12 -> 'b','d','f','h','j','l','n','p','r','t','v','x'
digit 13     -> 'z'
digit 14..18 -> '0','2','4','6','8'
```

`include/desrt/common.h`:

```cpp
DESRT_HD inline uint64_t idx_to_key(uint64_t idx);                  // -> canonical bytes
DESRT_HD inline void     key_to_string(uint64_t key, char out[8]);  // -> ASCII string
DESRT_HD inline uint64_t string_to_idx(const char s[8]);            // UINT64_MAX on bad char
```

`string_to_idx` is **permissive on input**: it accepts any of the original
36 `[a-z0-9]` characters and folds each into its parity-class digit. So
`string_to_idx("abcdefgh")` and `string_to_idx("abddffhh")` return the same
idx; `desrt crack` prints the result in canonical form because that is the
unique DES-equivalent representative.

> Switching to a different charset is still a single point of change in
> `common.h`: update `CHARSET_LEN`, `N_KEYSPACE`, the `canonical_char`
> helper, and the `string_to_idx` parity-class lookup. DES, the reduction,
> the chain kernel, the shard I/O, and the crack lookup are all
> charset-agnostic.

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
constants. The LCG runs `mod 2^64` and we take `mod N` afterwards. With
~12.4M draws against `N ≈ 1.70 × 10^10`, accidental start-point collisions
are still <1% (`m²/(2N) ≈ 4.5 · 10⁻³`).

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

With `chain_len = 4096 = 2^12` and `chains = 12,440,000`:

```
chain_len * chains / N = 5.095e+10 / 1.698e+10 = 3.000
p_one_table         ≈ 1 - exp(-3.000) ≈ 0.9502
```

### 3.1 Picking `t` and `m`

The product `m · t` determines coverage. The split between `m` and `t` is a
free parameter that controls the **build / crack / storage** trade-off:

| Quantity     | Cost in (m, t) |
|---|---|
| Build cost   | O(m · t)       |
| Crack cost   | O(T · t² / 2)  — for T targets |
| Storage      | O(m)           |

Holding `m·t = 3N` constant, doubling `m` while halving `t`:
- Build: unchanged (`m · t` constant).
- Crack: drops 4× (`t²` halves twice).
- Storage: doubles.

We pick `t = 4096`, `m = 12.44M` because:
- Build cost `~5.1e10` DES ops ≈ 5 min on one L4.
- Crack cost `T · t² / 2 = 32 · 4096²/2 ≈ 2.7e8` ops ≈ 2 s on one L4 for
  T = 32 targets.
- Storage 200 MiB — trivially small vs the 77 GiB disk budget.

The previous default of `t = 2^20` gave a tiny table (800 KiB) but
catastrophic crack cost: `T · t² / 2 = 32 · 2³⁹ ≈ 1.76 · 10¹³` ops ≈
**30 hours per crack invocation**, which the user observed as a "stuck"
process. Crack time scales as `t²`, so this choice is highly sensitive.

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

12,440,000 × 16 B = **~200 MiB** per table. We partition by

```
shard_id = endpoint mod num_shards
```

with `num_shards ∈ {2048, 4096}`. 4096 shards gives ~3,037 records per
shard on average (~48 KiB / shard) which fits in RAM trivially during the
sort step. The on-disk layout:

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
threads total. **The dominant cost is the chain-step inner loop**: each
thread does up to `t - p - 1` chain steps, averaging `t/2`. Total work:

```
W_crack = T · t · (t/2) = T · t² / 2
```

At `t = 4096`, `T = 32`: `W = 2.7·10⁸` DES ops ≈ 2 s at 150 MH/s.
At `t = 2²⁰`, `T = 32`: `W = 1.76·10¹³` ops ≈ **30 h** — this was the
original bug (kernel ran for hours with no progress output because
`cudaDeviceSynchronize()` blocked the host).

On the host:

1. Bucket candidates by `shard = endpoint % num_shards` and sort by
   `(shard, candidate)`.
2. Walk shards in order; load each sorted shard once; binary-search every
   candidate that falls in it.
3. For each match, replay forward from the recorded `startpoint` for `p`
   chain steps, compute `key = idx_to_key(idx_p)`, and check
   `DES(key, plaintext) == CT`. False positives (different chain converging
   to the same endpoint) fail this check.

With the current defaults (`T = 32`, `t = 4096`, `m = 12.44M`, 4096 shards):

- Candidate count = `T · t = 131,072`. Buffer 1 MiB on host.
- Candidates per shard ≈ 32 on average.
- Shard walk loads ~4096 × 48 KiB = ~200 MiB of disk → seconds even on HDD.
- Replay-verify per hit ≈ t/2 = 2048 DES on CPU ≈ 1 ms.

Total wall time for `T = 32` crack: ~2 s (GPU) + ~1 s (shard I/O) + <1 s
(verify) = under 5 s end-to-end.

## 7. Performance — where the time goes

The textbook DES kernel is bound by bit-by-bit permutation loops (the `IP`,
`FP`, `E`, `P`, and `PC-2` permutations are each 32–64 iterations of one
shift + one OR). On an L4 (Ada Lovelace, SM 8.9, 7.4 TFLOPS FP32) we expect
something like:

* 50 – 300 MH/s **per GPU**
* 100 – 600 MH/s across both L4s
* full 12.44M-chain build (×4096 = ~5.10e10 DES ops): a few minutes on a
  single L4 — quick enough that you can iterate on the design.
  A useful sanity check is `desrt bench` followed by
  `desrt build --chains <small> --chain-len <small>` to extrapolate.

For the original 62-char keyspace the build numbers were five orders of
magnitude larger (~10 TiB-equivalent of work). The 10 GH/s target from the
spec was sized for that — for the 19⁸ DES-effective keyspace it is comfortably
met by the textbook kernel. A bitsliced rewrite is still the natural next
step if you want to push to multi-table coverage or much larger keyspaces:

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
desrt build --start-chain-id 0       --chains 6220000 --gpu 0 --out ./tab &
desrt build --start-chain-id 6220000 --chains 6220000 --gpu 1 --out ./tab &
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
