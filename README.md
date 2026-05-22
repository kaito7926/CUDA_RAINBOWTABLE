# desrt — DES chosen-plaintext rainbow table (CUDA 12, C++17)

A learning-project rainbow-table builder and cracker for the DES cipher under
a **single fixed plaintext** `0x1122334455667788` and **8-character lowercase
alnum keys** (`[a-z0-9]^8`).

> **DES parity collapse.** DES PC-1 drops the LSB of each input key byte, so
> ASCII characters that differ only in their LSB (e.g. 'b'/'c', '0'/'1') are
> **DES-equivalent**. Within `[a-z0-9]^8` this collapses the effective key
> space from 36⁸ to **19⁸** equivalence classes — only 19 distinct DES-key
> bytes per position. The 19-char canonical alphabet used internally is
> `abdfhjlnprtvxz02468`. `string_to_idx` accepts any of the original 36
> characters and canonicalises them, so you can still type "abcdefgh" — the
> recovered key will simply print in canonical form (here "abddffhh"). See
> `docs/design.md` §1 for the full derivation.

`N = 19⁸ = 16,983,563,041 (≈ 1.70 × 10¹⁰)`. With `chain_len = 2^20` a single
table reaches ~95% model coverage at about **50,000 chains** (raw size
≈ 800 KiB).

## Layout

```
include/desrt/
    common.h          # config, charset/key helpers, splitmix64, reduction, LCG, Record
    des.h             # textbook FIPS-46 DES (host + device)
    shard.h           # ShardWriter, sort_shard, read_shard, find_endpoint
    cli.h             # tiny flag parser

src/
    main.cpp          # CLI dispatch
    shard.cpp
    cmd_plan.cpp
    cmd_sort.cpp
    cmd_make_target.cpp
    cmd_stats.cpp
    cmd_bench.cu      # KAT + host/GPU throughput
    cmd_build.cu      # chain kernel + streaming to shards
    cmd_crack.cu      # candidate kernel + shard lookup + replay-verify

docs/
    design.md         # chain math, coverage model, GPU layout, roadmap
    commands.md       # per-command flag reference

scripts/
    build.sh          # cmake + ninja wrapper for the server

CMakeLists.txt
```

## Build (on the Linux server with CUDA 12.7 and the 2× L4 GPUs)

```bash
# 1. Copy this tree to the server, then:
cd cuda_rainbowtable
./scripts/build.sh        # or run the cmake commands below directly

# Equivalent manual commands:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j

# Resulting binary:
./build/desrt
```

## Quick start (small test run)

The full build takes a long time even on 2× L4 (see `docs/design.md` —
textbook DES is correctness-first, not speed-first). Start with a small run to
verify correctness end-to-end:

```bash
# 1. Sanity check: known-answer tests + microbench.
./build/desrt bench --host-iters 100000 --gpu-threads 16384 --gpu-iters 256

# 2. Plan numbers (defaults target ~95% coverage of N = 19^8).
./build/desrt plan --chain-len 1048576 --chains 50000 --shards 4096

# 3. Smoke build: 2k chains of length 1024.
./build/desrt build --out ./tab --chains 2000 --chain-len 1024 \
                    --shards 64 --batch-size 1024

# 4. Sort the raw shards.
./build/desrt sort --in ./tab --shards 64

# 5. Pick one of the keys we just covered as a target and crack it.
#    To guarantee a hit, take a startpoint and walk a single step out:
./build/desrt make-target --out targets.txt --count 4
./build/desrt crack --table ./tab --target targets.txt \
                    --chain-len 1024 --shards 64

# 6. Stats.
./build/desrt stats --table ./tab --shards 64
```

For the full-size run, target ~95% coverage of the 19⁸ DES-effective keyspace:

```bash
./build/desrt build  --out /data/desrt-table --table-id 0 \
                     --chains 50000 --chain-len 1048576 --shards 4096 \
                     --batch-size 65536 --shard-buffer-kb 64
./build/desrt sort   --in    /data/desrt-table --shards 4096 --jobs 8
./build/desrt stats  --table /data/desrt-table --shards 4096
```

Expected `desrt stats` output for a healthy build at `mt/N ≈ 3`:

```
total records      : 50000
total unique eps   : ~19,500–19,800   (≈ 0.39 of records)
records/shard      : min 0  max ~50   mean 12.2   empty ≈ 30
```

> **`unique_eps/records` is NOT coverage.** It measures effective chains
> surviving merges. For a rainbow table tuned to 95% coverage (`mt/N ≈ 3`),
> the merge ODE gives `m_t = 1/(t/(2N) + 1/m₀) ≈ 0.4 · m₀`. That low ratio
> is the *cost* of oversampling to get high coverage — not a build bug.
> The real coverage is measured by `desrt crack` on random targets;
> expect 85–95% hit rate.
>
> If you instead see `total unique eps` at a few thousand or `0.003` of
> records, you are running an oversaturated build (e.g. 8.1M chains over
> the 19⁸ space). Reduce `--chains` to ~50,000.

## Validating DES

`desrt bench` runs three known-answer tests, the first of which is the spec's
test vector:

```
key       = 0x133457799BBCDFF1
plaintext = 0x0123456789ABCDEF
ciphertext should be 0x85E813540F0AB405
```

If any KAT fails, the bench command refuses to continue and exits 3.

## Caveats and known limitations

- **Speed.** This is the textbook (bit-by-bit permutation) DES, picked for
  correctness over throughput. On L4 you should expect roughly tens to a few
  hundred MH/s per GPU, so the full 18 h target from the spec is *not*
  achievable with this implementation. See `docs/design.md` for the bitsliced
  roadmap that would close the gap.
- **Single-GPU per process.** `--gpu DEVICE` selects one GPU. To use both L4
  cards, run two `desrt build` processes with disjoint `--start-chain-id`
  ranges (chains are independent given the chain_id, so output shards from
  both processes can be sorted together in one `desrt sort` pass — the raw
  shard files are open for append).
- **DES parity bits.** PC-1 drops the eight LSBs of the input key. Two ASCII
  alnum keys that differ only in those LSBs are DES-equivalent, so a recovered
  key may not match the *exact* byte string from `make-target`; the
  ciphertexts will match.

## Reading order

1. `include/desrt/common.h` — the small primitives.
2. `include/desrt/des.h` — straight-line FIPS-46 DES.
3. `src/cmd_build.cu` — the chain kernel.
4. `src/cmd_crack.cu` — rainbow-table lookup.
5. `docs/design.md` — the math behind the numbers in `desrt plan`.
