// `desrt plan` — back-of-the-envelope size, coverage, and build-time numbers.
//
// Inputs (with sensible defaults for the 19^8 DES-effective keyspace):
//   --chain-len    chain length (default 4096)
//   --chains       chain count per table (default 12,440,000)
//   --tables       number of tables (default 1)
//   --shards       number of shards on disk (default 4096)
//   --rate         assumed end-to-end DES rate in GH/s (default 10.0)
//
// We picked (t = 4096, m ≈ 12.44M) instead of (t = 2^20, m = 50K) because
// crack cost scales as T·t²/2 while build cost scales as m·t. With
// m·t/N ≈ 3 fixed (95% coverage), shrinking t by 256× and growing m by
// 256× preserves build cost (5.1e10 ops) but reduces crack work by 65,536×.
// At t = 4096, a 32-target crack runs in ~2 s on a single L4 instead of
// ~30 hours. Table size grows from 0.8 MiB to ~200 MiB — well within disk.
//
// Coverage uses the standard rainbow-table model: probability that a uniformly
// random key index lands somewhere in the table is approximately
//    1 - exp(- chain_len * chains_per_table / N)    for one table.
// With `tables` independent tables (different table_ids), per-key coverage is
//    1 - (1 - p_one_table)^tables
// This ignores chain-merges, which always lowers actual coverage below this
// upper bound. Real coverage is typically 0.85 - 0.95 of the model figure.

#include <cmath>
#include <cstdint>
#include <cstdio>

#include "desrt/cli.h"
#include "desrt/common.h"

static double estimate_coverage_one_table(double chain_len, double chains, double N) {
    if (N <= 0.0) return 0.0;
    return 1.0 - std::exp(-(chain_len * chains) / N);
}

int cmd_plan(int argc, char** argv) {
    auto a = desrt::cli::wrap(argc, argv);
    if (a.has("--help", "-h")) {
        std::printf(
            "desrt plan [--chain-len N] [--chains N] [--tables N] [--shards N] [--rate GH/s]\n");
        return 0;
    }

    // Defaults tuned for the 19-char canonical (DES-injective) keyspace:
    //   N = 19^8 = 16,983,563,041 ≈ 1.70e10.
    // For ~95% model coverage we need chain_len * chains ≈ 3·N ≈ 5.10e10.
    // We pick chain_len = 4096 = 2^12 (small for fast crack: T·t²/2 ≈ 2.7e8
    // ops for T=32 targets ≈ 2 s on one L4) and chains ≈ 12.44M to keep
    // m·t/N ≈ 3.
    const uint32_t chain_len = a.opt_u32("--chain-len", 4096);
    const uint64_t chains    = a.opt_u64("--chains",    12440000ULL);
    const uint32_t tables    = a.opt_u32("--tables",    1);
    const uint32_t shards    = a.opt_u32("--shards",    4096);
    const double   rate_ghps = a.opt_double("--rate",   10.0);

    const uint64_t N = desrt::N_KEYSPACE;

    const uint64_t records_per_table   = chains;
    const uint64_t raw_bytes_per_table = records_per_table * sizeof(desrt::Record);
    const uint64_t raw_bytes_total     = raw_bytes_per_table * tables;

    const double cov_one = estimate_coverage_one_table(
        static_cast<double>(chain_len),
        static_cast<double>(chains),
        static_cast<double>(N));
    const double cov_all = 1.0 - std::pow(1.0 - cov_one, tables);

    const double work_ops_per_table = static_cast<double>(chain_len) * static_cast<double>(chains);
    const double work_ops_total     = work_ops_per_table * tables;
    const double seconds_total      = work_ops_total / (rate_ghps * 1.0e9);
    const double hours_total        = seconds_total / 3600.0;

    const double avg_bytes_per_shard = static_cast<double>(raw_bytes_per_table) / shards;

    std::printf("desrt plan\n");
    std::printf("  charset (canonical): abdfhjlnprtvxz02468 (|charset|=%d)\n",
                desrt::CHARSET_LEN);
    std::printf("  accepts at input   : [a-z0-9] (auto-canonicalised in string_to_idx)\n");
    std::printf("  key length         : %d\n", desrt::KEY_LEN);
    std::printf("  N (keyspace)       : %llu (19^8, DES-injective)\n",
                static_cast<unsigned long long>(N));
    std::printf("  plaintext          : 0x%016llX\n",
                static_cast<unsigned long long>(desrt::DEFAULT_PLAINTEXT));
    std::printf("\n");
    std::printf("  chain length       : %u\n", chain_len);
    std::printf("  chains per table   : %llu\n",
                static_cast<unsigned long long>(chains));
    std::printf("  tables             : %u\n", tables);
    std::printf("  shards (per table) : %u  (~%.1f MiB avg per shard)\n",
                shards, avg_bytes_per_shard / (1024.0 * 1024.0));
    std::printf("\n");
    std::printf("  raw size per table : %.2f GiB  (%llu records, 16 B each)\n",
                static_cast<double>(raw_bytes_per_table) / (1024.0 * 1024.0 * 1024.0),
                static_cast<unsigned long long>(records_per_table));
    std::printf("  raw size total     : %.2f GiB\n",
                static_cast<double>(raw_bytes_total) / (1024.0 * 1024.0 * 1024.0));
    std::printf("\n");
    std::printf("  model coverage     : %.4f per table, %.4f across %u tables\n",
                cov_one, cov_all, tables);
    std::printf("  (model ignores chain merges; expect ~0.85 * model in practice)\n");
    std::printf("\n");
    std::printf("  DES ops per table  : %.3e\n", work_ops_per_table);
    std::printf("  DES ops total      : %.3e\n", work_ops_total);
    std::printf("  assumed rate       : %.2f GH/s (end-to-end)\n", rate_ghps);
    std::printf("  est. build time    : %.1f s  ==  %.2f h\n",
                seconds_total, hours_total);

    return 0;
}
