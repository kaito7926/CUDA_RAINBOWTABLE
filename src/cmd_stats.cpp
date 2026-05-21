// `desrt stats` — summarise a built rainbow table.
//
// Reports per-shard record counts, unique-endpoint counts (cheap because the
// sorted shards are already sorted by endpoint), total records, and a coarse
// shard-balance metric (min, max, mean records per shard).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "desrt/cli.h"
#include "desrt/shard.h"

namespace fs = std::filesystem;

namespace {

struct ShardStat {
    uint32_t shard;
    size_t   records;
    size_t   uniques;
};

void print_help() {
    std::printf(
        "desrt stats --table DIR [--shards N] [--jobs N] [--per-shard]\n"
        "\n"
        "Walks <DIR>/sorted/shard_*.bin and reports counts. --per-shard prints\n"
        "one line per shard (verbose).\n");
}

} // namespace

int cmd_stats(int argc, char** argv) {
    auto a = desrt::cli::wrap(argc, argv);
    if (a.has("--help", "-h")) { print_help(); return 0; }

    const std::string root  = a.opt_str("--table", "");
    if (root.empty()) {
        std::fprintf(stderr, "stats: --table DIR is required\n");
        print_help();
        return 1;
    }
    const uint32_t num_shards = a.opt_u32("--shards", 4096U);
    const bool     per_shard  = a.has("--per-shard");
    const int      jobs_in    = a.opt_int("--jobs", 0);
    unsigned jobs = (jobs_in > 0)
                    ? static_cast<unsigned>(jobs_in)
                    : std::max(1u, std::thread::hardware_concurrency());

    std::vector<ShardStat> stats(num_shards);
    std::atomic<uint32_t> next_shard{0};
    std::atomic<size_t>   total_records{0};
    std::atomic<size_t>   total_uniques{0};
    std::mutex log_mu;

    auto worker = [&]() {
        for (;;) {
            uint32_t s = next_shard.fetch_add(1);
            if (s >= num_shards) return;
            const std::string path = desrt::sorted_shard_path(root, s);
            ShardStat st{s, 0, 0};
            if (fs::exists(path)) {
                auto recs = desrt::read_shard(path);
                st.records = recs.size();
                if (!recs.empty()) {
                    size_t u = 1;
                    for (size_t i = 1; i < recs.size(); i++) {
                        if (recs[i].endpoint != recs[i - 1].endpoint) u++;
                    }
                    st.uniques = u;
                }
            }
            stats[s] = st;
            total_records.fetch_add(st.records);
            total_uniques.fetch_add(st.uniques);
        }
    };

    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> ts;
    for (unsigned i = 0; i < jobs; i++) ts.emplace_back(worker);
    for (auto& t : ts) t.join();
    auto t1 = std::chrono::steady_clock::now();

    size_t min_rec = SIZE_MAX, max_rec = 0;
    uint32_t empty_shards = 0;
    for (const auto& st : stats) {
        if (st.records < min_rec) min_rec = st.records;
        if (st.records > max_rec) max_rec = st.records;
        if (st.records == 0) empty_shards++;
    }
    if (min_rec == SIZE_MAX) min_rec = 0;

    if (per_shard) {
        for (const auto& st : stats) {
            std::printf("  shard %06u  records=%zu  uniques=%zu\n",
                        st.shard, st.records, st.uniques);
        }
    }

    double secs = std::chrono::duration<double>(t1 - t0).count();
    size_t total = total_records.load();
    size_t uniq  = total_uniques.load();
    double mean  = num_shards > 0 ? double(total) / num_shards : 0.0;
    double uniq_frac = total > 0 ? double(uniq) / double(total) : 0.0;

    std::printf("\ndesrt stats: table=%s shards=%u jobs=%u (read in %.2fs)\n",
                root.c_str(), num_shards, jobs, secs);
    std::printf("  total records      : %zu\n", total);
    std::printf("  total unique eps   : %zu  (%.4f of records)\n",
                uniq, uniq_frac);
    std::printf("  raw bytes (16B/rec): %.2f GiB\n",
                double(total) * 16.0 / (1024.0 * 1024.0 * 1024.0));
    std::printf("  records/shard      : min=%zu  max=%zu  mean=%.1f  empty=%u\n",
                min_rec, max_rec, mean, empty_shards);
    return 0;
}
