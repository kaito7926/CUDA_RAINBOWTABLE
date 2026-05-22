# desrt — per-command reference

All commands accept `--help` (alias `-h`). All values that look like integers
are parsed by `strtoull(..., 0)`, so `0x` and `0` prefixes work.

## `desrt plan`

Reports the size, coverage, and build-time estimate for a configuration. No
side effects.

```
desrt plan
    [--chain-len N]   default 1048576
    [--chains N]      default 50000     (~95% coverage of N = 19^8)
    [--tables N]      default 1
    [--shards N]      default 4096
    [--rate GH/s]     default 10.0      (used for the time estimate only)
```

> The internal keyspace is 19⁸, not 36⁸ — see `docs/design.md` §1 for the
> DES parity-collapse derivation. Building with the older `--chains 8100000`
> sizes will *oversaturate* and `desrt stats` will report ~26K unique
> endpoints out of 8.1M records (0.3% of records), because every chain step
> collapses through the 19⁸ effective bottleneck.

The "model coverage" line is the upper-bound `1 - exp(-LC/N)`; chain merges
reduce real coverage. See `docs/design.md` §3.

## `desrt bench`

Runs three FIPS-46 known-answer tests, then measures host-side DES throughput
(one CPU thread) and GPU throughput (one DES per thread, no chain reductions).

```
desrt bench
    [--host-iters N]    default 200000
    [--gpu-threads N]   default 65536
    [--gpu-iters N]     default 4096   (per-thread inner-loop count)
    [--gpu DEVICE]      default 0
    [--skip-host] [--skip-gpu]
```

Exits 3 if any KAT fails.

## `desrt build`

Generates the rainbow-table chains for one `table_id` and streams Records to
per-shard raw files at `<out>/raw/shard_NNNNNN.bin`.

```
desrt build
    --out DIR             required
    [--chains N]            default 50000    (~95% coverage of N = 19^8)
    [--start-chain-id N]    default 0        (for resume/multi-GPU)
    [--chain-len N]         default 1048576
    [--table-id N]          default 0
    [--shards N]            default 4096
    [--batch-size N]        default 65536    (chains per kernel launch)
    [--shard-buffer-kb N]   default 64       (per-shard host buffer)
    [--block-size N]        default 128      (CUDA threads per block)
    [--gpu DEVICE]          default 0
    [--plaintext 0xHEX]     default 0x1122334455667788
```

The "batch" is `--batch-size` chains; each batch is one kernel launch and one
device→host copy of `batch * 16` bytes. Bigger batches reduce overhead but
make individual kernels longer and progress logging coarser.

`--start-chain-id` lets you resume or split work across GPUs. Each chain is
fully determined by `chain_id` plus the global constants, so two runs with
disjoint `--start-chain-id` / `--chains` ranges over the same `--out`
directory produce a single coherent table after `desrt sort`.

## `desrt sort`

Reads each `<in>/raw/shard_NNNNNN.bin`, sorts by `endpoint`, writes
`<in>/sorted/shard_NNNNNN.bin`, and removes the raw file (unless `--keep-raw`).

```
desrt sort
    --in DIR        required
    [--shards N]    default 4096
    [--jobs N]      default = hardware_concurrency()
    [--keep-raw]    keep the raw shards instead of deleting them
```

Missing raw shards are silently skipped (treated as empty).

## `desrt make-target`

Picks random alnum keys, encrypts the fixed plaintext under each, and writes
a text file of `<idx>\t<key>\t<ciphertext_hex>` lines.

```
desrt make-target
    [--out FILE]         default targets.txt
    [--count N]          default 8
    [--seed N]           default = wall clock
    [--plaintext 0xHEX]  default 0x1122334455667788
    [--key STRING]       8 chars from [a-z0-9]; auto-canonicalised internally
    [--key-index N]      0 <= N < 19^8;     only this one target is emitted
```

The emitted `<key>` column is the canonical 19-char form (`abdfhjlnprtvxz02468`)
even if you typed something with non-canonical parity (e.g. you passed
`--key abcdefgh`, you'll see `abddffhh` in the output). The ciphertext is
`DES(canonical_key, plaintext)`, which is exactly what `desrt crack` recovers.

The `--key`/`--key-index` form is useful for hand-constructed crack tests:
pick a chain start, walk a few steps on paper, hand the ciphertext to
`desrt crack`, and you can predict exactly which `p` the hit will land at.

## `desrt crack`

Recovers keys for one or more ciphertexts.

```
desrt crack
    --table DIR           required, points at a sorted-shards directory
    --target FILE         or use --ct
    --ct HEX              single-target shortcut
    [--chain-len N]       default 1048576    (must match the build)
    [--table-id N]        default 0          (must match the build)
    [--shards N]          default 4096       (must match the build)
    [--gpu DEVICE]        default 0
    [--plaintext 0xHEX]   default 0x1122334455667788
    [--block-size N]      default 128
```

Output:

```
Results (3/8 solved):
  [HIT ] aB3xY9Qr  ct=85E813540F0AB405  key=aB3xY9Qr  idx=12345678
  [MISS] T#1       ct=DEADBEEFCAFEBABE
  ...
shards loaded=4096  false positives=37  lookup=2.34s  total=12.05s
```

Exit code: 0 if every target is solved, 4 otherwise.

## `desrt stats`

Summary of a built table.

```
desrt stats
    --table DIR     required
    [--shards N]    default 4096
    [--jobs N]      default = hardware_concurrency()
    [--per-shard]   print one line per shard
```

Reports total records, unique endpoints, raw byte size, and per-shard
record-count distribution.
