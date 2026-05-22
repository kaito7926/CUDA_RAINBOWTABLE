// `desrt crack` — recover keys for one or more ciphertexts using one or more
// rainbow tables.
//
// Multi-table support: `--table DIR1,DIR2,...` accepts a comma-separated list
// of table directories; `--table-id N0,N1,...` lists their build-time
// table_ids (defaults to 0,1,2,... matching position). The crack runs each
// table in sequence and prunes already-solved targets between passes — so a
// 3-table run does the full work for ~T targets on table 0, then ~T·(1-p)
// targets on table 1, then ~T·(1-p)² on table 2. Multiple independent tables
// boost coverage as 1 - (1 - p)^k, the standard Hellman/Oechslin idea.
//
// Algorithm (single table):
//   For each target ciphertext CT and each position p in [0, chain_len):
//     y_{p+1} = reduce(CT, p, table_id, N)              (hypothesis: ct_p == CT)
//     y_L    = chain_step^{L-p-1}(y_{p+1}, p+1.., table_id, P, N)
//     endpoint_candidate = y_L
//   Look up endpoint_candidate in sorted shards. On a hit, replay from the
//   recorded startpoint for p chain steps; compute key_p, verify by encrypting
//   plaintext with key_p and comparing to CT.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>

#include "desrt/cli.h"
#include "desrt/common.h"
#include "desrt/des.h"
#include "desrt/shard.h"

namespace {

#define CUDA_OK(expr) do {                                                      \
    cudaError_t _e = (expr);                                                    \
    if (_e != cudaSuccess) {                                                    \
        std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__,   \
                     __LINE__, cudaGetErrorString(_e));                         \
        std::exit(2);                                                           \
    }                                                                           \
} while (0)

struct Target {
    uint64_t ciphertext;
    std::string label; // e.g. original key or "T#3"
};

// Parse the make-target output format: "<index>\t<key>\t<ciphertext_hex>"
// Also accept a bare hex ciphertext on its own line.
std::vector<Target> load_targets(const std::string& path) {
    std::vector<Target> out;
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "crack: cannot open target file %s\n", path.c_str());
        std::exit(1);
    }
    std::string line;
    int n = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> toks;
        std::stringstream ss(line);
        std::string t;
        while (ss >> t) toks.push_back(t);
        if (toks.empty()) continue;
        const std::string& cthex = toks.back();
        uint64_t ct = std::strtoull(cthex.c_str(), nullptr, 16);
        Target tgt;
        tgt.ciphertext = ct;
        if (toks.size() >= 2) tgt.label = toks[toks.size() - 2];
        else                  tgt.label = "T#" + std::to_string(n);
        out.push_back(tgt);
        n++;
    }
    return out;
}

// Split a comma-separated string. Empty input → empty vector.
std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    if (s.empty()) return out;
    size_t start = 0;
    while (start <= s.size()) {
        size_t comma = s.find(',', start);
        if (comma == std::string::npos) comma = s.size();
        std::string tok = s.substr(start, comma - start);
        // Trim whitespace.
        size_t b = tok.find_first_not_of(" \t");
        size_t e = tok.find_last_not_of(" \t");
        if (b != std::string::npos) out.push_back(tok.substr(b, e - b + 1));
        if (comma == s.size()) break;
        start = comma + 1;
    }
    return out;
}

__global__ void crack_compute_kernel(
    uint64_t* d_candidates,        // length = num_targets * chain_len
    const uint64_t* d_targets,     // length = num_targets
    uint32_t num_targets,
    uint32_t chain_len,
    uint32_t table_id,
    uint64_t plaintext,
    uint64_t N)
{
    uint64_t gtid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)num_targets * chain_len;
    if (gtid >= total) return;
    uint32_t p = static_cast<uint32_t>(gtid % chain_len);
    uint32_t t = static_cast<uint32_t>(gtid / chain_len);

    uint64_t ct = d_targets[t];
    uint64_t y  = desrt::reduce_idx(ct, p, table_id, N);

    for (uint32_t r = p + 1; r < chain_len; r++) {
        y = desrt::des::chain_step(y, r, table_id, plaintext, N);
    }
    d_candidates[gtid] = y;
}

void print_help() {
    std::printf(
        "desrt crack --table DIR[,DIR2,...] --target FILE [--chain-len N]\n"
        "            [--table-id N[,N2,...]] [--shards N] [--gpu DEVICE]\n"
        "            [--plaintext 0xHEX] [--block-size N] [--ct HEX] [--jobs N]\n"
        "\n"
        "--table accepts a comma-separated list of table directories. Each table\n"
        "is queried in sequence; targets solved by an earlier table are skipped\n"
        "in later passes. Multiple independent tables (different --table-id at\n"
        "build) boost coverage as 1 - (1 - p)^k, so 3 tables with p ≈ 0.6 reach\n"
        "~94%% hit rate where one table reached ~60%%.\n"
        "\n"
        "--table-id is a parallel list; defaults to 0,1,2,... matching the order\n"
        "of --table paths. Each table_id must match what `desrt build --table-id`\n"
        "used.\n"
        "\n"
        "--jobs N parallelises the host-side shard walk + replay-verify across\n"
        "N CPU threads. Defaults to std::thread::hardware_concurrency().\n");
}

struct Probe {
    uint32_t shard;
    uint32_t target;    // index into the local (unsolved) target list for this pass
    uint32_t p;
    uint64_t cand;
};

uint64_t replay_forward(uint64_t idx, uint32_t start_round, uint32_t steps,
                        uint32_t table_id, uint64_t plaintext, uint64_t N)
{
    for (uint32_t i = 0; i < steps; i++) {
        idx = desrt::des::chain_step(idx, start_round + i, table_id, plaintext, N);
    }
    return idx;
}

struct PassStats {
    uint32_t targets_input;     // how many unsolved targets fed into this pass
    uint32_t newly_solved;
    double   gpu_secs;
    double   lookup_secs;
    uint32_t shards_loaded;
    uint64_t false_positives;
};

// Run one rainbow-table crack pass against `unsolved_idx` (indices into the
// global `targets` / `solved` arrays). Updates the global solved/recovered
// arrays for any hits. Designed to be called once per (--table, --table-id)
// pair when multi-table cracking.
PassStats crack_one_pass(
    const std::string& table_root,
    uint32_t table_id,
    const std::vector<Target>& targets,
    const std::vector<uint32_t>& unsolved_idx,
    std::vector<uint8_t>& solved,
    std::vector<uint64_t>& recovered_idx,
    std::vector<std::string>& recovered_key,
    uint32_t chain_len,
    uint32_t num_shards,
    int device,
    uint64_t plaintext,
    int block_size,
    unsigned jobs,
    uint64_t N)
{
    PassStats ps{};
    ps.targets_input = static_cast<uint32_t>(unsolved_idx.size());

    if (unsolved_idx.empty()) return ps;
    const uint32_t T = static_cast<uint32_t>(unsolved_idx.size());

    // Local target list (this pass's view).
    std::vector<uint64_t> h_targets(T);
    for (uint32_t i = 0; i < T; i++) {
        h_targets[i] = targets[unsolved_idx[i]].ciphertext;
    }

    // ---- 1. GPU computes candidates ----
    CUDA_OK(cudaSetDevice(device));

    uint64_t* d_targets    = nullptr;
    uint64_t* d_candidates = nullptr;
    const size_t cand_bytes = (size_t)T * chain_len * sizeof(uint64_t);
    CUDA_OK(cudaMalloc(&d_targets, T * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc(&d_candidates, cand_bytes));
    CUDA_OK(cudaMemcpy(d_targets, h_targets.data(), T * sizeof(uint64_t),
                       cudaMemcpyHostToDevice));

    const uint64_t total_threads = (uint64_t)T * chain_len;
    const uint64_t grid = (total_threads + block_size - 1) / block_size;

    std::printf("  GPU kernel: grid=%llu block=%d threads=%llu\n",
                (unsigned long long)grid, block_size,
                (unsigned long long)total_threads);
    std::fflush(stdout);

    auto t_gpu0 = std::chrono::steady_clock::now();
    crack_compute_kernel<<<grid, block_size>>>(
        d_candidates, d_targets, T, chain_len, table_id, plaintext, N);
    CUDA_OK(cudaDeviceSynchronize());
    auto t_gpu1 = std::chrono::steady_clock::now();
    ps.gpu_secs = std::chrono::duration<double>(t_gpu1 - t_gpu0).count();
    std::printf("  GPU candidates: %.2f s\n", ps.gpu_secs);
    std::fflush(stdout);

    std::vector<uint64_t> h_candidates((size_t)T * chain_len);
    CUDA_OK(cudaMemcpy(h_candidates.data(), d_candidates, cand_bytes,
                       cudaMemcpyDeviceToHost));
    CUDA_OK(cudaFree(d_candidates));
    CUDA_OK(cudaFree(d_targets));

    // ---- 2. Bucket candidates by shard ----
    std::vector<Probe> probes;
    probes.reserve((size_t)T * chain_len);
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t p = 0; p < chain_len; p++) {
            uint64_t cand = h_candidates[(size_t)t * chain_len + p];
            probes.push_back(Probe{
                desrt::shard_for_endpoint(cand, num_shards), t, p, cand
            });
        }
    }
    std::sort(probes.begin(), probes.end(),
              [](const Probe& a, const Probe& b) {
                  if (a.shard != b.shard) return a.shard < b.shard;
                  return a.cand < b.cand;
              });

    // ---- 3. Walk shards, look up, replay-verify (multi-threaded) ----
    // Per-pass local hit state — we publish to the global solved[] only on
    // confirmed verify. solved_local stays the size of the GLOBAL target
    // list so we can short-circuit using global solved state.
    std::mutex result_mu;

    struct ShardRange { size_t lo; size_t hi; uint32_t shard; };
    std::vector<ShardRange> ranges;
    ranges.reserve(num_shards);
    {
        size_t i = 0;
        while (i < probes.size()) {
            uint32_t s = probes[i].shard;
            size_t j = i;
            while (j < probes.size() && probes[j].shard == s) j++;
            ranges.push_back({i, j, s});
            i = j;
        }
    }

    std::printf("  lookup phase: %zu shard-ranges, %u host threads\n",
                ranges.size(), jobs);
    std::fflush(stdout);

    std::atomic<size_t>   next_range{0};
    std::atomic<uint32_t> shards_loaded{0};
    std::atomic<uint64_t> false_positives{0};
    std::atomic<uint32_t> newly_solved{0};
    // Local "solved" tracker keyed by local target index, to skip already-
    // solved targets within this pass.
    std::vector<uint8_t> pass_solved(T, 0);

    auto worker = [&]() {
        for (;;) {
            size_t r = next_range.fetch_add(1, std::memory_order_relaxed);
            if (r >= ranges.size()) return;
            const ShardRange& rng = ranges[r];

            auto shard = desrt::read_shard(
                desrt::sorted_shard_path(table_root, rng.shard));
            shards_loaded.fetch_add(1, std::memory_order_relaxed);
            if (shard.empty()) continue;

            uint64_t local_fp = 0;
            for (size_t k = rng.lo; k < rng.hi; k++) {
                uint32_t local_t = probes[k].target;
                if (pass_solved[local_t]) continue;

                auto lo_it = std::lower_bound(
                    shard.begin(), shard.end(), probes[k].cand,
                    [](const desrt::Record& r, uint64_t v) {
                        return r.endpoint < v;
                    });
                for (auto it = lo_it;
                     it != shard.end() && it->endpoint == probes[k].cand;
                     ++it)
                {
                    uint64_t idx_p = replay_forward(
                        it->startpoint, /*start_round=*/0,
                        /*steps=*/probes[k].p,
                        table_id, plaintext, N);
                    uint64_t key = desrt::idx_to_key(idx_p);
                    uint64_t ct  = desrt::des::encrypt_block(key, plaintext);
                    if (ct == h_targets[local_t]) {
                        std::lock_guard<std::mutex> lk(result_mu);
                        uint32_t global_t = unsolved_idx[local_t];
                        if (!solved[global_t]) {
                            solved[global_t]        = 1;
                            recovered_idx[global_t] = idx_p;
                            char keybuf[9];
                            desrt::key_to_string(key, keybuf);
                            keybuf[8] = '\0';
                            recovered_key[global_t] = keybuf;
                            newly_solved.fetch_add(1, std::memory_order_relaxed);
                        }
                        pass_solved[local_t] = 1;
                        break;
                    } else {
                        local_fp++;
                    }
                }
            }
            false_positives.fetch_add(local_fp, std::memory_order_relaxed);
        }
    };

    auto t_lookup0 = std::chrono::steady_clock::now();
    std::vector<std::thread> threads;
    threads.reserve(jobs);
    for (unsigned i = 0; i < jobs; i++) threads.emplace_back(worker);
    for (auto& th : threads) th.join();
    auto t_lookup1 = std::chrono::steady_clock::now();
    ps.lookup_secs = std::chrono::duration<double>(t_lookup1 - t_lookup0).count();
    ps.shards_loaded   = shards_loaded.load();
    ps.false_positives = false_positives.load();
    ps.newly_solved    = newly_solved.load();
    return ps;
}

} // namespace

int cmd_crack(int argc, char** argv) {
    auto a = desrt::cli::wrap(argc, argv);
    if (a.has("--help", "-h")) { print_help(); return 0; }

    // ---- Parse table list and table_id list ----
    const std::string tables_str = a.opt_str("--table", "");
    std::vector<std::string> table_dirs = split_csv(tables_str);
    if (table_dirs.empty()) {
        std::fprintf(stderr, "crack: --table DIR[,DIR2,...] is required\n");
        print_help();
        return 1;
    }
    const std::string ids_str = a.opt_str("--table-id", "");
    std::vector<uint32_t> table_ids;
    if (ids_str.empty()) {
        // Default: 0, 1, 2, ... matching position.
        for (size_t i = 0; i < table_dirs.size(); i++) {
            table_ids.push_back(static_cast<uint32_t>(i));
        }
    } else {
        for (const auto& tok : split_csv(ids_str)) {
            table_ids.push_back(static_cast<uint32_t>(std::strtoul(tok.c_str(), nullptr, 0)));
        }
        if (table_ids.size() != table_dirs.size()) {
            std::fprintf(stderr,
                "crack: --table-id list (%zu) must match --table list (%zu)\n",
                table_ids.size(), table_dirs.size());
            return 1;
        }
    }

    const std::string target_path = a.opt_str("--target", "");
    const std::string single_ct   = a.opt_str("--ct", "");
    const uint32_t chain_len  = a.opt_u32("--chain-len", 4096U);
    const uint32_t num_shards = a.opt_u32("--shards", 4096U);
    const int      device     = a.opt_int("--gpu", 0);
    const uint64_t plaintext  = a.opt_u64("--plaintext", desrt::DEFAULT_PLAINTEXT);
    const int      block_size = a.opt_int("--block-size", 128);
    const uint64_t N = desrt::N_KEYSPACE;

    const int jobs_in = a.opt_int("--jobs", 0);
    unsigned jobs = (jobs_in > 0)
                    ? static_cast<unsigned>(jobs_in)
                    : std::max(1u, std::thread::hardware_concurrency());

    std::vector<Target> targets;
    if (!single_ct.empty()) {
        Target t;
        t.ciphertext = std::strtoull(single_ct.c_str(), nullptr, 16);
        t.label = "T#0";
        targets.push_back(t);
    }
    if (!target_path.empty()) {
        auto more = load_targets(target_path);
        targets.insert(targets.end(), more.begin(), more.end());
    }
    if (targets.empty()) {
        std::fprintf(stderr, "crack: provide --target FILE or --ct HEX\n");
        print_help();
        return 1;
    }
    const uint32_t T = static_cast<uint32_t>(targets.size());

    std::printf("desrt crack: targets=%u tables=%zu chain-len=%u shards=%u jobs=%u\n",
                T, table_dirs.size(), chain_len, num_shards, jobs);
    std::printf("  plaintext=0x%016llX  gpu=%d\n",
                (unsigned long long)plaintext, device);
    const double est_work_des = static_cast<double>(T) *
                                static_cast<double>(chain_len) *
                                static_cast<double>(chain_len) * 0.5;
    std::printf("  per-table GPU work ~ %.3e DES ops (T·t²/2)\n", est_work_des);
    if (chain_len >= (1u << 18)) {
        std::printf("  WARNING: chain-len=%u is large. Crack cost scales as t² —\n"
                    "           expect minutes per table. Consider rebuilding with\n"
                    "           --chain-len 4096.\n", chain_len);
    }
    std::fflush(stdout);

    // Global state across all tables.
    std::vector<uint8_t>     solved(T, 0);
    std::vector<uint64_t>    recovered_idx(T, 0);
    std::vector<std::string> recovered_key(T);

    // Combined stats for the final report.
    double   total_gpu_secs        = 0.0;
    double   total_lookup_secs     = 0.0;
    uint32_t total_shards_loaded   = 0;
    uint64_t total_false_positives = 0;

    auto t_all0 = std::chrono::steady_clock::now();

    for (size_t ti = 0; ti < table_dirs.size(); ti++) {
        // Build unsolved-target list.
        std::vector<uint32_t> unsolved_idx;
        unsolved_idx.reserve(T);
        for (uint32_t i = 0; i < T; i++) {
            if (!solved[i]) unsolved_idx.push_back(i);
        }
        std::printf("\n[table %zu/%zu] dir=%s table_id=%u  unsolved=%zu/%u\n",
                    ti + 1, table_dirs.size(),
                    table_dirs[ti].c_str(), table_ids[ti],
                    unsolved_idx.size(), T);
        std::fflush(stdout);

        if (unsolved_idx.empty()) {
            std::printf("  all targets already solved — skipping remaining tables\n");
            break;
        }

        PassStats ps = crack_one_pass(
            table_dirs[ti], table_ids[ti],
            targets, unsolved_idx,
            solved, recovered_idx, recovered_key,
            chain_len, num_shards, device, plaintext,
            block_size, jobs, N);

        std::printf("  pass result: %u newly solved  |  %llu false positives  |  "
                    "gpu=%.2fs  lookup=%.2fs\n",
                    ps.newly_solved,
                    (unsigned long long)ps.false_positives,
                    ps.gpu_secs, ps.lookup_secs);

        total_gpu_secs        += ps.gpu_secs;
        total_lookup_secs     += ps.lookup_secs;
        total_shards_loaded   += ps.shards_loaded;
        total_false_positives += ps.false_positives;
    }

    auto t_all1 = std::chrono::steady_clock::now();
    double wall_secs = std::chrono::duration<double>(t_all1 - t_all0).count();

    // ---- Report ----
    int solved_count = 0;
    for (uint8_t b : solved) if (b) solved_count++;
    std::printf("\nResults (%d/%u solved):\n", solved_count, T);
    for (uint32_t t = 0; t < T; t++) {
        std::printf("  [%s] %s  ct=%016llX",
                    solved[t] ? "HIT " : "MISS",
                    targets[t].label.c_str(),
                    (unsigned long long)targets[t].ciphertext);
        if (solved[t]) {
            std::printf("  key=%s  idx=%llu",
                        recovered_key[t].c_str(),
                        (unsigned long long)recovered_idx[t]);
        }
        std::printf("\n");
    }
    std::printf("\ntables=%zu shards loaded=%u (cumulative)  false positives=%llu  "
                "gpu=%.2fs  lookup=%.2fs  wall=%.2fs\n",
                table_dirs.size(),
                total_shards_loaded,
                (unsigned long long)total_false_positives,
                total_gpu_secs, total_lookup_secs, wall_secs);

    return (solved_count == static_cast<int>(T)) ? 0 : 4;
}
