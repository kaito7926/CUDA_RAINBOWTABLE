// `desrt make-target` — pick random 8-char keys, encrypt the fixed plaintext
// under each, and emit a text file of (key, key_index, ciphertext) triples
// suitable for `desrt crack --target`.
//
// User-supplied --key may use any of [a-z0-9]; it is auto-canonicalised by
// `string_to_idx` into the 19-char DES-injective alphabet
// "abdfhjlnprtvxz02468". The emitted key column is the canonical form, and
// the emitted ciphertext is DES(canonical_key, plaintext) — i.e. exactly
// what the rainbow-table lookup will recover.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

#include "desrt/cli.h"
#include "desrt/common.h"
#include "desrt/des.h"

namespace {

void print_help() {
    std::printf(
        "desrt make-target [--out FILE] [--count N] [--seed N] [--plaintext 0xHEX]\n"
        "                  [--key KEY] [--key-index N]\n"
        "\n"
        "Emits one line per target:\n"
        "    <index>  <key>  <ciphertext_hex>\n"
        "If --key or --key-index is given, only that single target is written and\n"
        "--count is ignored.\n");
}

} // namespace

int cmd_make_target(int argc, char** argv) {
    auto a = desrt::cli::wrap(argc, argv);
    if (a.has("--help", "-h")) { print_help(); return 0; }

    const std::string out_path = a.opt_str("--out", "targets.txt");
    const uint64_t count       = a.opt_u64("--count", 8ULL);
    const uint64_t seed        = a.opt_u64("--seed",
        static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()));
    const uint64_t plaintext   = a.opt_u64("--plaintext", desrt::DEFAULT_PLAINTEXT);

    const std::string specific_key = a.opt_str("--key", "");
    const char* idx_str            = a.opt("--key-index", nullptr, nullptr);

    FILE* f = std::fopen(out_path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "make-target: cannot open %s\n", out_path.c_str());
        return 1;
    }

    auto emit = [&](uint64_t idx) {
        uint64_t k64 = desrt::idx_to_key(idx);
        uint64_t ct  = desrt::des::encrypt_block(k64, plaintext);
        char key_str[9];
        desrt::key_to_string(k64, key_str);
        key_str[8] = '\0';
        std::fprintf(f, "%llu\t%s\t%016llX\n",
                     (unsigned long long)idx, key_str,
                     (unsigned long long)ct);
    };

    if (!specific_key.empty() || idx_str) {
        uint64_t idx;
        if (!specific_key.empty()) {
            if (specific_key.size() != 8) {
                std::fprintf(stderr, "make-target: --key must be exactly 8 characters from [a-z0-9]\n");
                std::fclose(f);
                return 1;
            }
            idx = desrt::string_to_idx(specific_key.c_str());
            if (idx == UINT64_MAX) {
                std::fprintf(stderr, "make-target: --key must be exactly 8 characters from [a-z0-9]\n");
                std::fclose(f);
                return 1;
            }
        } else {
            idx = std::strtoull(idx_str, nullptr, 0);
            if (idx >= desrt::N_KEYSPACE) {
                std::fprintf(stderr, "make-target: --key-index out of range (N=%llu)\n",
                             (unsigned long long)desrt::N_KEYSPACE);
                std::fclose(f);
                return 1;
            }
        }
        emit(idx);
    } else {
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<uint64_t> dist(0, desrt::N_KEYSPACE - 1);
        for (uint64_t i = 0; i < count; i++) emit(dist(rng));
    }

    std::fclose(f);
    std::printf("desrt make-target: wrote %s (plaintext=0x%016llX)\n",
                out_path.c_str(), (unsigned long long)plaintext);
    return 0;
}
