// desrt — shared constants and small utilities used on both host and device.
#pragma once

#include <cstdint>
#include <cstddef>

#if defined(__CUDACC__)
  #define DESRT_HD __host__ __device__
  #define DESRT_INLINE __forceinline__
  // CUDA-only pragmas; expand to nothing on the host compiler so we don't
  // collect "unknown pragma" warnings when the DES header is included from
  // plain C++ translation units (cmd_make_target.cpp).
  #define DESRT_UNROLL   _Pragma("unroll")
  #define DESRT_NOUNROLL _Pragma("unroll 1")
#else
  #define DESRT_HD
  #define DESRT_INLINE inline
  #define DESRT_UNROLL
  #define DESRT_NOUNROLL
#endif

namespace desrt {

// Chosen-plaintext for the attack. Big-endian when written as the spec's
// 0x1122334455667788 — i.e. byte 0 (MSB) is 0x11.
constexpr uint64_t DEFAULT_PLAINTEXT = 0x1122334455667788ULL;

// Charset and key length. Keep these constexpr so the constants propagate.
//
// Charset is the 36-character lowercase-alphanumeric set
// "abcdefghijklmnopqrstuvwxyz0123456789".
//   digit  0..25 -> 'a'..'z'
//   digit 26..35 -> '0'..'9'
//
// N = 36^8 = 2,821,109,907,456 (~2^41.4), about 77x smaller than the 62^8
// alnum keyspace. Coverage and table-size figures in `desrt plan` follow
// directly from this.
constexpr int CHARSET_LEN = 36;
constexpr int KEY_LEN = 8;

constexpr uint64_t N_KEYSPACE =
    36ULL * 36ULL * 36ULL * 36ULL * 36ULL * 36ULL * 36ULL * 36ULL;

// LCG constants (Knuth's MMIX). Used to map chain_id -> startpoint index.
// Note: the LCG runs mod 2^64, and we then take mod N. This is not a
// permutation over [0, N), but for ~8M draws against N≈2.82e12, accidental
// startpoint collisions are still negligible.
constexpr uint64_t LCG_A = 6364136223846793005ULL;
constexpr uint64_t LCG_B = 1442695040888963407ULL;

// Record stored to disk (little-endian on x86_64).
#pragma pack(push, 1)
struct Record {
    uint64_t endpoint;
    uint64_t startpoint;
};
#pragma pack(pop)
static_assert(sizeof(Record) == 16, "Record must be packed at 16 bytes");

// ---------- splitmix64 ----------
// Standard splitmix64 from Steele/Lea, used here as a high-avalanche mixer.
DESRT_HD DESRT_INLINE uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// ---------- charset <-> index <-> packed key ----------
// The "key" returned is a 64-bit value with the FIRST character of the 8-char
// string in the MOST significant byte (byte 7), so the bit-1 (MSB) of the
// value is the MSB of the first character. This matches the byte order that
// DES PC-1 expects.
DESRT_HD DESRT_INLINE uint64_t idx_to_key(uint64_t idx) {
    uint64_t key = 0;
    // Iteration i=0 extracts the least significant base-36 digit, which
    // corresponds to the LAST character of the 8-char string.
    DESRT_UNROLL
    for (int i = 0; i < 8; i++) {
        uint32_t digit = static_cast<uint32_t>(idx % static_cast<uint64_t>(CHARSET_LEN));
        idx /= static_cast<uint64_t>(CHARSET_LEN);
        uint8_t c;
        if (digit < 26) c = static_cast<uint8_t>('a' + digit);
        else            c = static_cast<uint8_t>('0' + (digit - 26));
        // Place byte i at the i-th byte from LSB. i=0 -> LSB (last char),
        // i=7 -> MSB byte (first char).
        key |= (static_cast<uint64_t>(c) << (i * 8));
    }
    return key;
}

// Write the 8-char ASCII string for `key` into out[0..7] (no null terminator).
DESRT_HD DESRT_INLINE void key_to_string(uint64_t key, char out[8]) {
    DESRT_UNROLL
    for (int i = 0; i < 8; i++) {
        // out[0] is first character = byte 7 of key (MSB byte).
        out[i] = static_cast<char>((key >> ((7 - i) * 8)) & 0xFFu);
    }
}

// Returns UINT64_MAX if any character is outside the charset.
DESRT_HD DESRT_INLINE uint64_t string_to_idx(const char s[8]) {
    uint64_t idx = 0;
    DESRT_UNROLL
    for (int i = 0; i < 8; i++) {
        char c = s[i];
        uint32_t d;
        if (c >= 'a' && c <= 'z')      d = c - 'a';
        else if (c >= '0' && c <= '9') d = 26 + (c - '0');
        else                           return UINT64_MAX;
        idx = idx * static_cast<uint64_t>(CHARSET_LEN) + d;
    }
    return idx;
}

// ---------- reduction ----------
// reduce(ct, round, table_id) -> idx in [0, N).
// Mixes ct with round and table_id via splitmix64 so different rounds and
// tables yield uncorrelated reductions (the textbook "round-dependent
// reduction function" that prevents merges from collapsing chains).
DESRT_HD DESRT_INLINE uint64_t reduce_idx(
    uint64_t ct, uint32_t round, uint32_t table_id, uint64_t N)
{
    uint64_t m = ct;
    m ^= splitmix64(static_cast<uint64_t>(round) * 0x9E3779B97F4A7C15ULL
                  ^ static_cast<uint64_t>(table_id) * 0xC6BC279692B5C323ULL);
    return splitmix64(m) % N;
}

// LCG-based startpoint draw.
DESRT_HD DESRT_INLINE uint64_t startpoint_for(uint64_t chain_id, uint64_t N) {
    return ((LCG_A * chain_id) + LCG_B) % N;
}

} // namespace desrt
