// `desrt build` — generate rainbow table chains on the GPU and stream
// Records to per-shard raw files on disk.
//
// One CUDA thread = one chain. After each batch we cudaMemcpy the batch into
// a pinned host buffer, then the ShardWriter dispatches each Record into a
// per-shard buffer keyed on (endpoint % num_shards).
//
// Resumability: --start-chain-id lets a subsequent run continue from where a
// previous run left off (chains are independent given chain_id, so we can
// pause / resume / split across machines without coordination).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
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

__global__ void build_chains_kernel(
    desrt::Record* records,
    uint64_t chain_id_base,
    uint64_t chains_in_batch,
    uint32_t chain_len,
    uint32_t table_id,
    uint64_t plaintext,
    uint64_t N)
{
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= chains_in_batch) return;
    uint64_t chain_id = chain_id_base + tid;
    uint64_t start_idx = desrt::startpoint_for(chain_id, N);
    uint64_t idx = start_idx;
    for (uint32_t r = 0; r < chain_len; r++) {
        idx = desrt::des::chain_step(idx, r, table_id, plaintext, N);
    }
    desrt::Record rec;
    rec.endpoint   = idx;
    rec.startpoint = start_idx;
    records[tid] = rec;
}

void print_help() {
    std::printf(
        "desrt build --out DIR [--chains N] [--start-chain-id N] [--chain-len N]\n"
        "            [--table-id N] [--shards N] [--batch-size N]\n"
        "            [--shard-buffer-kb N] [--gpu DEVICE] [--plaintext 0xHEX]\n"
        "            [--block-size N]\n"
        "\n"
        "Defaults: chain-len=4096, chains=12440000 (~95%% of N=19^8),\n"
        "          shards=4096, batch-size=65536, shard-buffer-kb=64.\n"
        "\n"
        "Crack cost scales as T·t²/2; we trade large m for small t so the\n"
        "online lookup runs in seconds. See docs/design.md §6 for the math.\n");
}

} // namespace

int cmd_build(int argc, char** argv) {
    auto a = desrt::cli::wrap(argc, argv);
    if (a.has("--help", "-h")) { print_help(); return 0; }

    const std::string out_root = a.opt_str("--out", "");
    if (out_root.empty()) {
        std::fprintf(stderr, "desrt build: --out DIR is required\n");
        print_help();
        return 1;
    }

    const uint64_t chains       = a.opt_u64("--chains", 12440000ULL);
    const uint64_t start_chain  = a.opt_u64("--start-chain-id", 0ULL);
    const uint32_t chain_len    = a.opt_u32("--chain-len", 4096U);
    const uint32_t table_id     = a.opt_u32("--table-id", 0U);
    const uint32_t num_shards   = a.opt_u32("--shards", 4096U);
    const uint64_t batch_size   = a.opt_u64("--batch-size", 65536ULL);
    const uint32_t shard_buf_kb = a.opt_u32("--shard-buffer-kb", 64U);
    const int      device       = a.opt_int("--gpu", 0);
    const uint64_t plaintext    = a.opt_u64("--plaintext", desrt::DEFAULT_PLAINTEXT);
    const int      block_size   = a.opt_int("--block-size", 128);

    const uint64_t N = desrt::N_KEYSPACE;

    std::printf("desrt build\n");
    std::printf("  out=%s table-id=%u shards=%u\n",
                out_root.c_str(), table_id, num_shards);
    std::printf("  chain-len=%u chains=%llu start-chain-id=%llu\n",
                chain_len,
                (unsigned long long)chains,
                (unsigned long long)start_chain);
    std::printf("  batch-size=%llu block-size=%d shard-buffer-kb=%u gpu=%d\n",
                (unsigned long long)batch_size, block_size,
                shard_buf_kb, device);
    std::printf("  plaintext=0x%016llX\n", (unsigned long long)plaintext);

    CUDA_OK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_OK(cudaGetDeviceProperties(&prop, device));
    std::printf("  device: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);

    desrt::Record* d_records = nullptr;
    desrt::Record* h_pinned  = nullptr;
    CUDA_OK(cudaMalloc(&d_records, batch_size * sizeof(desrt::Record)));
    CUDA_OK(cudaMallocHost(&h_pinned, batch_size * sizeof(desrt::Record)));

    desrt::ShardWriter writer(out_root, num_shards, shard_buf_kb * 1024ULL);

    const uint64_t end_chain = start_chain + chains;
    uint64_t chain_id = start_chain;
    uint64_t batches  = 0;
    auto t_start = std::chrono::steady_clock::now();
    double last_log = 0.0;

    while (chain_id < end_chain) {
        uint64_t this_batch = std::min<uint64_t>(batch_size, end_chain - chain_id);
        uint64_t grid = (this_batch + block_size - 1) / block_size;

        build_chains_kernel<<<grid, block_size>>>(
            d_records, chain_id, this_batch, chain_len,
            table_id, plaintext, N);
        // Synchronous copy so the kernel result is available before we touch h_pinned.
        // For textbook DES, kernel time dominates copy/IO so this is fine.
        CUDA_OK(cudaMemcpy(h_pinned, d_records,
                           this_batch * sizeof(desrt::Record),
                           cudaMemcpyDeviceToHost));

        writer.append_batch(h_pinned, this_batch);

        chain_id += this_batch;
        batches  += 1;

        double secs = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t_start).count();
        if (secs - last_log >= 5.0 || chain_id == end_chain) {
            uint64_t done = chain_id - start_chain;
            double done_frac = static_cast<double>(done) / static_cast<double>(chains);
            double ops = static_cast<double>(done) * static_cast<double>(chain_len);
            double rate = ops / secs;
            double eta = secs * (1.0 - done_frac) / std::max(done_frac, 1e-9);
            std::printf("  [%6.2f%%] chains=%llu batches=%llu  "
                        "elapsed=%.1fs  rate=%.2f MH/s  eta=%.1fs\n",
                        100.0 * done_frac,
                        (unsigned long long)done,
                        (unsigned long long)batches,
                        secs, rate / 1.0e6, eta);
            std::fflush(stdout);
            last_log = secs;
        }
    }

    std::printf("  flushing shard writers...\n");
    writer.flush_all();

    CUDA_OK(cudaFree(d_records));
    CUDA_OK(cudaFreeHost(h_pinned));

    auto t_end = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t_end - t_start).count();
    double ops = static_cast<double>(chains) * static_cast<double>(chain_len);
    std::printf("desrt build: %llu chains in %.2f s  (%.3e DES ops, %.2f MH/s)\n",
                (unsigned long long)chains, secs, ops, (ops / secs) / 1.0e6);
    std::printf("desrt build: raw shards in %s/raw/. Next step: desrt sort --in %s --shards %u\n",
                out_root.c_str(), out_root.c_str(), num_shards);
    return 0;
}
