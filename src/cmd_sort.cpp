// `desrt sort` — for each raw shard, read it into memory, sort by endpoint,
// write the sorted shard, and (optionally) delete the raw shard.
//
// With 4096 shards over ~10 GiB, each shard is ~2.5 MiB which fits trivially
// in RAM. Sorting is single-threaded per shard by default; --jobs N runs
// multiple shard sorts concurrently.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "desrt/cli.h"
#include "desrt/shard.h"

namespace {

void print_help() {
    std::printf(
        "desrt sort --in DIR [--shards N] [--jobs N] [--keep-raw]\n"
        "\n"
        "Reads <DIR>/raw/shard_*.bin and writes <DIR>/sorted/shard_*.bin sorted by\n"
        "endpoint. Deletes the raw file after the sorted one is on disk unless\n"
        "--keep-raw is given. Default --jobs is std::thread::hardware_concurrency().\n");
}

} // namespace

int cmd_sort(int argc, char** argv) {
    auto a = desrt::cli::wrap(argc, argv);
    if (a.has("--help", "-h")) { print_help(); return 0; }

    const std::string in_root = a.opt_str("--in", "");
    if (in_root.empty()) {
        std::fprintf(stderr, "desrt sort: --in DIR is required\n");
        print_help();
        return 1;
    }
    const uint32_t num_shards = a.opt_u32("--shards", 4096U);
    const bool     keep_raw   = a.has("--keep-raw");
    const int      jobs_in    = a.opt_int("--jobs", 0);
    unsigned jobs = (jobs_in > 0)
                    ? static_cast<unsigned>(jobs_in)
                    : std::max(1u, std::thread::hardware_concurrency());

    std::printf("desrt sort: in=%s shards=%u jobs=%u %s\n",
                in_root.c_str(), num_shards, jobs,
                keep_raw ? "(keep-raw)" : "(deleting raw after sort)");

    desrt::ensure_shard_dirs(in_root);

    std::atomic<uint32_t> next_shard{0};
    std::atomic<size_t>   total_records{0};
    std::atomic<uint32_t> done_shards{0};
    std::atomic<uint32_t> failed_shards{0};

    auto t_start = std::chrono::steady_clock::now();
    std::mutex log_mu;

    auto worker = [&]() {
        for (;;) {
            uint32_t s = next_shard.fetch_add(1);
            if (s >= num_shards) return;
            try {
                size_t n = desrt::sort_shard(in_root, s, !keep_raw);
                total_records.fetch_add(n);
            } catch (const std::exception& ex) {
                failed_shards.fetch_add(1);
                std::lock_guard<std::mutex> lk(log_mu);
                std::fprintf(stderr, "  shard %u FAILED: %s\n", s, ex.what());
            }
            uint32_t done = done_shards.fetch_add(1) + 1;
            if (done % 64 == 0 || done == num_shards) {
                std::lock_guard<std::mutex> lk(log_mu);
                double secs = std::chrono::duration<double>(
                                  std::chrono::steady_clock::now() - t_start).count();
                std::printf("  sorted %u / %u shards  records=%zu  elapsed=%.1fs\n",
                            done, num_shards, total_records.load(), secs);
                std::fflush(stdout);
            }
        }
    };

    std::vector<std::thread> ts;
    for (unsigned i = 0; i < jobs; i++) ts.emplace_back(worker);
    for (auto& t : ts) t.join();

    double secs = std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - t_start).count();
    std::printf("desrt sort: %u shards, %zu records, %.2f s, %u failed\n",
                num_shards, total_records.load(), secs, failed_shards.load());
    return failed_shards.load() ? 1 : 0;
}
