// `desrt bench` — first run known-answer tests against the FIPS-46 DES test
// vector, then measure DES throughput on the host (one thread) and on the GPU
// (one thread per chain "step").
//
// The host KAT also stresses the bit conventions used by everything else in
// the project: if these numbers come out wrong, do NOT trust any later step.

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

namespace {

#define CUDA_OK(expr) do {                                                      \
    cudaError_t _e = (expr);                                                    \
    if (_e != cudaSuccess) {                                                    \
        std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__,   \
                     __LINE__, cudaGetErrorString(_e));                         \
        std::exit(2);                                                           \
    }                                                                           \
} while (0)

struct Kat {
    const char* name;
    uint64_t key;
    uint64_t plaintext;
    uint64_t expected;
};

// Standard DES known-answer tests. The first one is the FIPS-46 test vector
// called out in the assignment spec.
constexpr Kat KATS[] = {
    {"FIPS-46 vector", 0x133457799BBCDFF1ULL, 0x0123456789ABCDEFULL,
                      0x85E813540F0AB405ULL},
    {"all-zero key/pt", 0x0000000000000000ULL, 0x0000000000000000ULL,
                       0x8CA64DE9C1B123A7ULL},
    {"all-one key/pt", 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
                      0x7359B2163E4EDC58ULL}
};

int run_host_kats() {
    int failed = 0;
    for (const auto& k : KATS) {
        uint64_t got = desrt::des::encrypt_block(k.key, k.plaintext);
        bool ok = (got == k.expected);
        std::printf("  [%s] %-15s  key=%016llX pt=%016llX ct=%016llX (want %016llX)\n",
                    ok ? "OK  " : "FAIL", k.name,
                    (unsigned long long)k.key,
                    (unsigned long long)k.plaintext,
                    (unsigned long long)got,
                    (unsigned long long)k.expected);
        if (!ok) failed++;
    }
    return failed;
}

double bench_host(uint64_t iters) {
    // We use a cheap dependency chain: feed each ciphertext into the next
    // encryption as the key. This prevents the compiler from hoisting the
    // call out of the loop, and produces a deterministic accumulator we can
    // print to keep the work observable.
    uint64_t key = 0x0123456789ABCDEFULL;
    uint64_t pt  = desrt::DEFAULT_PLAINTEXT;
    auto t0 = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < iters; i++) {
        key = desrt::des::encrypt_block(key, pt);
    }
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::printf("  host: %llu encryptions in %.3f s  ->  %.2f MH/s  (sink=%016llX)\n",
                (unsigned long long)iters, secs,
                (iters / secs) / 1.0e6,
                (unsigned long long)key);
    return iters / secs;
}

__global__ void bench_kernel(uint64_t* out_sinks, uint64_t per_thread,
                             uint64_t plaintext)
{
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    // Per-thread starting key derived from tid so threads do independent work.
    uint64_t key = 0x0123456789ABCDEFULL ^ (tid * 0x9E3779B97F4A7C15ULL);
    for (uint64_t i = 0; i < per_thread; i++) {
        key = desrt::des::encrypt_block(key, plaintext);
    }
    out_sinks[tid] = key;
}

double bench_gpu(int device, uint64_t threads, uint64_t per_thread) {
    CUDA_OK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_OK(cudaGetDeviceProperties(&prop, device));
    std::printf("  GPU %d: %s, %d SMs, %.1f GiB\n", device, prop.name,
                prop.multiProcessorCount,
                static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));

    uint64_t* d_sinks = nullptr;
    CUDA_OK(cudaMalloc(&d_sinks, threads * sizeof(uint64_t)));

    int block = 128;
    uint64_t grid = (threads + block - 1) / block;

    cudaEvent_t start, stop;
    CUDA_OK(cudaEventCreate(&start));
    CUDA_OK(cudaEventCreate(&stop));

    // Warm-up
    bench_kernel<<<grid, block>>>(d_sinks, /*per_thread=*/64, desrt::DEFAULT_PLAINTEXT);
    CUDA_OK(cudaDeviceSynchronize());

    CUDA_OK(cudaEventRecord(start));
    bench_kernel<<<grid, block>>>(d_sinks, per_thread, desrt::DEFAULT_PLAINTEXT);
    CUDA_OK(cudaEventRecord(stop));
    CUDA_OK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
    double secs = ms / 1000.0;

    std::vector<uint64_t> sinks(threads);
    CUDA_OK(cudaMemcpy(sinks.data(), d_sinks, threads * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));
    uint64_t fold = 0;
    for (uint64_t v : sinks) fold ^= v;

    CUDA_OK(cudaFree(d_sinks));
    CUDA_OK(cudaEventDestroy(start));
    CUDA_OK(cudaEventDestroy(stop));

    double ops = static_cast<double>(threads) * static_cast<double>(per_thread);
    double rate = ops / secs;
    std::printf("  GPU: %llu threads x %llu enc = %.3e enc in %.3f s  ->  %.2f MH/s  (fold=%016llX)\n",
                (unsigned long long)threads,
                (unsigned long long)per_thread,
                ops, secs, rate / 1.0e6,
                (unsigned long long)fold);
    return rate;
}

} // namespace

int cmd_bench(int argc, char** argv) {
    auto a = desrt::cli::wrap(argc, argv);
    if (a.has("--help", "-h")) {
        std::printf(
            "desrt bench [--host-iters N] [--gpu-threads N] [--gpu-iters N] [--gpu DEVICE]\n"
            "            [--skip-host] [--skip-gpu]\n");
        return 0;
    }
    const uint64_t host_iters = a.opt_u64("--host-iters", 200000ULL);
    const uint64_t gpu_threads = a.opt_u64("--gpu-threads", 65536ULL);
    const uint64_t gpu_iters = a.opt_u64("--gpu-iters", 4096ULL);
    const int      device   = a.opt_int("--gpu", 0);
    const bool     skip_host = a.has("--skip-host");
    const bool     skip_gpu  = a.has("--skip-gpu");

    std::printf("DES known-answer tests (host):\n");
    int failed = run_host_kats();
    if (failed) {
        std::fprintf(stderr, "\nFAIL: %d DES KAT(s) failed. Refusing to bench.\n", failed);
        return 3;
    }
    std::printf("\n");

    if (!skip_host) {
        std::printf("Host throughput:\n");
        bench_host(host_iters);
        std::printf("\n");
    }

    if (!skip_gpu) {
        int devs = 0;
        cudaError_t e = cudaGetDeviceCount(&devs);
        if (e != cudaSuccess || devs == 0) {
            std::fprintf(stderr, "No CUDA device available (%s). Skipping GPU bench.\n",
                         cudaGetErrorString(e));
            return 0;
        }
        std::printf("GPU throughput:\n");
        bench_gpu(device, gpu_threads, gpu_iters);
    }
    return 0;
}
