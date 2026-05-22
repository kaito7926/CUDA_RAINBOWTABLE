// `desrt crack` — recover keys for one or more ciphertexts using a built
// rainbow table.
//
// Algorithm (standard rainbow-table lookup):
//   For each target ciphertext CT and each position p in [0, chain_len):
//     y_{p+1} = reduce(CT, p, table_id, N)              (hypothesis: ct_p == CT)
//     y_L    = chain_step^{L-p-1}(y_{p+1}, p+1.., table_id, P, N)
//     endpoint_candidate = y_L
//   Look up endpoint_candidate in sorted shards. On a hit, replay from the
//   recorded startpoint for p chain steps; compute key_p, verify by encrypting
//   plaintext with key_p and comparing to CT. False positives (different chain
//   that converged to the same endpoint) fail this last check.
//
// Implementation:
//   1. GPU kernel writes a (num_targets * chain_len) array of endpoint
//      candidates.
//   2. Host buckets candidates by shard_id and walks shards one at a time:
//      load the sorted shard, binary-search the candidates that fall in it.
//   3. On a hit, replay on CPU (DES on host is plenty fast for L=1e6 steps
//      per replay, ~0.1 s).

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
        // Find last whitespace-delimited token; treat it as ciphertext.
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

    // Continue the chain from position p+1 through chain_len-1.
    for (uint32_t r = p + 1; r < chain_len; r++) {
        y = desrt::des::chain_step(y, r, table_id, plaintext, N);
    }
    d_candidates[gtid] = y;
}

void print_help() {
    std::printf(
        "desrt crack --table DIR --target FILE [--chain-len N] [--table-id N]\n"
        "            [--shards N] [--gpu DEVICE] [--plaintext 0xHEX] [--block-size N]\n"
        "            [--ct HEX] [--jobs N]\n"
        "\n"
        "Either --target FILE (lines: <idx> <key> <ct_hex> or just <ct_hex>) or a\n"
        "single --ct HEX may be given. --jobs N parallelises the host-side shard\n"
        "walk and replay-verify across N CPU threads (defaults to all hardware\n"
        "threads). At m·t/N≈3 the table merges heavily, so the replay-verify\n"
        "loop dominates lookup time — multi-threading is essential.\n");
}

struct Probe {
    uint32_t shard;
    uint32_t target;
    uint32_t p;
    uint64_t cand;
};

// CPU replay of `steps` chain iterations starting at idx.
uint64_t replay_forward(uint64_t idx, uint32_t start_round, uint32_t steps,
                        uint32_t table_id, uint64_t plaintext, uint64_t N)
{
    for (uint32_t i = 0; i < steps; i++) {
        idx = desrt::des::chain_step(idx, start_round + i, table_id, plaintext, N);
    }
    return idx;
}

} // namespace

int cmd_crack(int argc, char** argv) {
    auto a = desrt::cli::wrap(argc, argv);
    if (a.has("--help", "-h")) { print_help(); return 0; }

    const std::string table_root = a.opt_str("--table", "");
    if (table_root.empty()) {
        std::fprintf(stderr, "crack: --table DIR is required\n");
        print_help();
        return 1;
    }
    const std::string target_path = a.opt_str("--target", "");
    const std::string single_ct   = a.opt_str("--ct", "");
    const uint32_t chain_len  = a.opt_u32("--chain-len", 4096U);
    const uint32_t table_id   = a.opt_u32("--table-id", 0U);
    const uint32_t num_shards = a.opt_u32("--shards", 4096U);
    const int      device     = a.opt_int("--gpu", 0);
    const uint64_t plaintext  = a.opt_u64("--plaintext", desrt::DEFAULT_PLAINTEXT);
    const int      block_size = a.opt_int("--block-size", 128);
    const uint64_t N = desrt::N_KEYSPACE;

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

    std::printf("desrt crack: targets=%u chain-len=%u table-id=%u shards=%u\n",
                T, chain_len, table_id, num_shards);
    std::printf("  plaintext=0x%016llX  gpu=%d\n",
                (unsigned long long)plaintext, device);
    // Heads-up so the user can see whether we are stuck.
    const double est_work_des = static_cast<double>(T) *
                                static_cast<double>(chain_len) *
                                static_cast<double>(chain_len) * 0.5;
    std::printf("  est. GPU work ~ %.3e DES ops (T·t²/2). At 150 MH/s ≈ %.1f s.\n",
                est_work_des, est_work_des / 1.5e8);
    if (chain_len >= (1u << 18)) {
        std::printf("  WARNING: chain-len=%u is large. Crack cost scales as t² —\n"
                    "           expect minutes to hours. Consider rebuilding with\n"
                    "           --chain-len 4096 (~2 s for T=32 on one L4).\n",
                    chain_len);
    }
    std::fflush(stdout);

    // ---- 1. GPU computes candidates ----
    CUDA_OK(cudaSetDevice(device));

    std::vector<uint64_t> h_targets(T);
    for (uint32_t i = 0; i < T; i++) h_targets[i] = targets[i].ciphertext;

    uint64_t* d_targets = nullptr;
    uint64_t* d_candidates = nullptr;
    const size_t cand_bytes = (size_t)T * chain_len * sizeof(uint64_t);
    CUDA_OK(cudaMalloc(&d_targets, T * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc(&d_candidates, cand_bytes));
    CUDA_OK(cudaMemcpy(d_targets, h_targets.data(), T * sizeof(uint64_t),
                       cudaMemcpyHostToDevice));

    const uint64_t total_threads = (uint64_t)T * chain_len;
    const uint64_t grid = (total_threads + block_size - 1) / block_size;

    std::printf("  launching GPU candidate kernel: grid=%llu block=%d "
                "threads=%llu\n",
                (unsigned long long)grid, block_size,
                (unsigned long long)total_threads);
    std::fflush(stdout);

    auto t_gpu0 = std::chrono::steady_clock::now();
    crack_compute_kernel<<<grid, block_size>>>(
        d_candidates, d_targets, T, chain_len, table_id, plaintext, N);
    CUDA_OK(cudaDeviceSynchronize());
    auto t_gpu1 = std::chrono::steady_clock::now();
    double gpu_secs = std::chrono::duration<double>(t_gpu1 - t_gpu0).count();
    std::printf("  GPU candidate phase: %.2f s  (%.3e endpoint computations)\n",
                gpu_secs, static_cast<double>(total_threads));
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
    // Each shard's probes form a contiguous range in `probes` (since we sorted
    // by shard). Workers atomically claim one shard at a time, read it, and
    // do all the replay-verify for its probes locally; results are merged at
    // the end. With T·t² / 2 replay work concentrated in this phase and
    // ~117K false positives at m·t/N≈3, single-thread crack took ~10 min on
    // the user's 24-core server (588 s). Parallelising drops it to seconds.
    std::vector<uint8_t> solved(T, 0);          // uint8_t: indexed atomically
    std::vector<uint64_t> recovered_idx(T, 0);
    std::vector<std::string> recovered_key(T);
    std::mutex result_mu;                        // protects vectors above

    const int jobs_in = a.opt_int("--jobs", 0);
    unsigned jobs = (jobs_in > 0)
                    ? static_cast<unsigned>(jobs_in)
                    : std::max(1u, std::thread::hardware_concurrency());

    // Index the boundaries: list of (probe_start, probe_end, shard_id) ranges
    // — one per shard that actually has probes. Workers consume these by
    // atomic counter.
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

    std::printf("  starting lookup phase: %zu shard-ranges, %u host threads\n",
                ranges.size(), jobs);
    std::fflush(stdout);

    std::atomic<size_t> next_range{0};
    std::atomic<uint32_t> shards_loaded{0};
    std::atomic<uint64_t> false_positives{0};

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
                uint32_t tgt = probes[k].target;
                // Cheap unsynchronised early-out: solved[tgt] is byte-sized
                // and only ever goes 0→1. A torn read at worst causes a
                // redundant replay, which is harmless.
                if (solved[tgt]) continue;

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
                    if (ct == h_targets[tgt]) {
                        // Lock to publish the hit. Contention is tiny
                        // (≤T hits total).
                        std::lock_guard<std::mutex> lk(result_mu);
                        if (!solved[tgt]) {
                            solved[tgt]        = 1;
                            recovered_idx[tgt] = idx_p;
                            char keybuf[9];
                            desrt::key_to_string(key, keybuf);
                            keybuf[8] = '\0';
                            recovered_key[tgt] = keybuf;
                        }
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
    double lookup_secs = std::chrono::duration<double>(t_lookup1 - t_lookup0).count();

    // ---- 4. Report ----
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
    std::printf("\nshards loaded=%u  false positives=%llu  lookup=%.2fs  total=%.2fs\n",
                shards_loaded.load(),
                (unsigned long long)false_positives.load(),
                lookup_secs, gpu_secs + lookup_secs);

    return (solved_count == static_cast<int>(T)) ? 0 : 4;
}
