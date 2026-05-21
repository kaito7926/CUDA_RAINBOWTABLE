// Textbook FIPS-46 DES implementation, callable from host and device.
//
// Notes:
//  * Bit numbering follows the FIPS-46 convention: bit 1 is the MOST
//    significant bit of the 64-bit data block. In our uint64_t the MSB is
//    bit position 63, so DES bit i (1-indexed) lives at (n - i) in the
//    underlying integer of width n.
//  * Parity bits (bits 8, 16, ..., 64) are NEVER used by PC-1, so the eight
//    LSBs of each input byte are discarded. This is why ASCII alnum keys
//    decoded back from a recovered DES key may differ in those bits from the
//    original 8-char input string.
//  * This is the correctness-first reference implementation. It runs at
//    "tens of MH/s per thread"-class speed; the kernel still parallelises
//    across many chains, so the build is slow but operational. A bitsliced
//    rewrite is a clearly-marked optimisation path (see docs/design.md).
#pragma once

#include <cstdint>
#include "desrt/common.h"

namespace desrt::des {

// ---------- FIPS-46 tables ----------

// Initial permutation.
inline constexpr uint8_t IP[64] = {
    58, 50, 42, 34, 26, 18, 10,  2,
    60, 52, 44, 36, 28, 20, 12,  4,
    62, 54, 46, 38, 30, 22, 14,  6,
    64, 56, 48, 40, 32, 24, 16,  8,
    57, 49, 41, 33, 25, 17,  9,  1,
    59, 51, 43, 35, 27, 19, 11,  3,
    61, 53, 45, 37, 29, 21, 13,  5,
    63, 55, 47, 39, 31, 23, 15,  7
};

// Final permutation (inverse of IP).
inline constexpr uint8_t FP[64] = {
    40,  8, 48, 16, 56, 24, 64, 32,
    39,  7, 47, 15, 55, 23, 63, 31,
    38,  6, 46, 14, 54, 22, 62, 30,
    37,  5, 45, 13, 53, 21, 61, 29,
    36,  4, 44, 12, 52, 20, 60, 28,
    35,  3, 43, 11, 51, 19, 59, 27,
    34,  2, 42, 10, 50, 18, 58, 26,
    33,  1, 41,  9, 49, 17, 57, 25
};

// E expansion (32 -> 48).
inline constexpr uint8_t E_TABLE[48] = {
    32,  1,  2,  3,  4,  5,
     4,  5,  6,  7,  8,  9,
     8,  9, 10, 11, 12, 13,
    12, 13, 14, 15, 16, 17,
    16, 17, 18, 19, 20, 21,
    20, 21, 22, 23, 24, 25,
    24, 25, 26, 27, 28, 29,
    28, 29, 30, 31, 32,  1
};

// P permutation inside f.
inline constexpr uint8_t P_TABLE[32] = {
    16,  7, 20, 21,
    29, 12, 28, 17,
     1, 15, 23, 26,
     5, 18, 31, 10,
     2,  8, 24, 14,
    32, 27,  3,  9,
    19, 13, 30,  6,
    22, 11,  4, 25
};

// PC-1 (64 -> 56).
inline constexpr uint8_t PC1[56] = {
    57, 49, 41, 33, 25, 17,  9,
     1, 58, 50, 42, 34, 26, 18,
    10,  2, 59, 51, 43, 35, 27,
    19, 11,  3, 60, 52, 44, 36,
    63, 55, 47, 39, 31, 23, 15,
     7, 62, 54, 46, 38, 30, 22,
    14,  6, 61, 53, 45, 37, 29,
    21, 13,  5, 28, 20, 12,  4
};

// PC-2 (56 -> 48).
inline constexpr uint8_t PC2[48] = {
    14, 17, 11, 24,  1,  5,
     3, 28, 15,  6, 21, 10,
    23, 19, 12,  4, 26,  8,
    16,  7, 27, 20, 13,  2,
    41, 52, 31, 37, 47, 55,
    30, 40, 51, 45, 33, 48,
    44, 49, 39, 56, 34, 53,
    46, 42, 50, 36, 29, 32
};

// Per-round left rotation count for C and D halves.
inline constexpr uint8_t SHIFTS[16] = {
    1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1
};

// S-boxes, each laid out as 64 entries in row-major order (row*16 + col),
// where row = (b1<<1)|b6 and col = (b2<<3)|(b3<<2)|(b4<<1)|b5 from the
// six-bit S-box input b1..b6 (MSB to LSB).
inline constexpr uint8_t S_BOX[8][64] = {
    { // S1
        14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7,
         0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8,
         4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0,
        15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13
    },
    { // S2
        15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10,
         3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5,
         0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15,
        13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9
    },
    { // S3
        10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8,
        13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1,
        13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7,
         1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12
    },
    { // S4
         7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15,
        13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9,
        10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4,
         3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14
    },
    { // S5
         2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9,
        14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6,
         4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14,
        11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3
    },
    { // S6
        12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11,
        10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8,
         9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6,
         4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13
    },
    { // S7
         4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1,
        13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6,
         1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2,
         6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12
    },
    { // S8
        13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7,
         1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2,
         7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8,
         2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11
    }
};

// ---------- helpers ----------

// Permute `in` (n_in valid bits) through `table` of length n_out. table[i] is
// 1-indexed and refers to a DES bit position (1 = MSB). The output uses the
// same convention: bit i+1 (DES) at uint64_t bit (n_out - 1 - i).
DESRT_HD DESRT_INLINE uint64_t permute(
    uint64_t in, const uint8_t* table, int n_in, int n_out)
{
    uint64_t out = 0;
    DESRT_NOUNROLL
    for (int i = 0; i < n_out; i++) {
        uint64_t bit = (in >> (n_in - table[i])) & 1ULL;
        out |= bit << (n_out - 1 - i);
    }
    return out;
}

// Left-rotate the low 28 bits.
DESRT_HD DESRT_INLINE uint32_t rotl28(uint32_t v, int s) {
    constexpr uint32_t M28 = (1u << 28) - 1u;
    v &= M28;
    return ((v << s) | (v >> (28 - s))) & M28;
}

// ---------- key schedule ----------

// Expand a 64-bit key into 16 subkeys (each in the low 48 bits of a uint64_t).
DESRT_HD DESRT_INLINE void key_schedule(uint64_t key, uint64_t subkeys[16]) {
    // PC-1: 64 -> 56 bits, the high 8 bits of the result are zero.
    uint64_t cd = permute(key, PC1, 64, 56);
    // Split into C (high 28 bits of the 56) and D (low 28 bits).
    uint32_t C = static_cast<uint32_t>((cd >> 28) & 0x0FFFFFFFu);
    uint32_t D = static_cast<uint32_t>(cd & 0x0FFFFFFFu);

    DESRT_NOUNROLL
    for (int i = 0; i < 16; i++) {
        C = rotl28(C, SHIFTS[i]);
        D = rotl28(D, SHIFTS[i]);
        uint64_t cdr = (static_cast<uint64_t>(C) << 28) | static_cast<uint64_t>(D);
        subkeys[i] = permute(cdr, PC2, 56, 48);
    }
}

// ---------- round function ----------

DESRT_HD DESRT_INLINE uint32_t feistel_f(uint32_t R, uint64_t subkey) {
    // E expansion: 32 -> 48 bits.
    uint64_t e = permute(static_cast<uint64_t>(R), E_TABLE, 32, 48);
    e ^= subkey;
    // Eight S-boxes, each consuming 6 bits and producing 4 bits.
    uint32_t s_out = 0;
    DESRT_UNROLL
    for (int i = 0; i < 8; i++) {
        // Take the (i+1)-th 6-bit group from the MSB end.
        uint32_t g = static_cast<uint32_t>((e >> (42 - 6 * i)) & 0x3Fu);
        uint32_t row = ((g >> 5) & 0x1u) << 1 | (g & 0x1u);   // b1 b6
        uint32_t col = (g >> 1) & 0xFu;                       // b2 b3 b4 b5
        uint32_t v = S_BOX[i][row * 16 + col];
        s_out |= v << (28 - 4 * i);
    }
    // P permutation: 32 -> 32 bits.
    return static_cast<uint32_t>(permute(static_cast<uint64_t>(s_out), P_TABLE, 32, 32));
}

// ---------- block encrypt ----------

// Encrypt one 64-bit block with a 64-bit key. Returns the ciphertext.
DESRT_HD DESRT_INLINE uint64_t encrypt_block(uint64_t key, uint64_t plaintext) {
    uint64_t subkeys[16];
    key_schedule(key, subkeys);

    uint64_t x = permute(plaintext, IP, 64, 64);
    uint32_t L = static_cast<uint32_t>((x >> 32) & 0xFFFFFFFFu);
    uint32_t R = static_cast<uint32_t>(x & 0xFFFFFFFFu);

    DESRT_NOUNROLL
    for (int i = 0; i < 16; i++) {
        uint32_t newR = L ^ feistel_f(R, subkeys[i]);
        L = R;
        R = newR;
    }
    // After 16 rounds the halves are swapped: pre-image is R || L.
    uint64_t preout = (static_cast<uint64_t>(R) << 32) | static_cast<uint64_t>(L);
    return permute(preout, FP, 64, 64);
}

// ---------- chain step ----------

// One iteration of the rainbow-table chain: key derived from idx, encrypt the
// fixed plaintext, reduce ciphertext back to an index.
DESRT_HD DESRT_INLINE uint64_t chain_step(
    uint64_t idx, uint32_t round, uint32_t table_id, uint64_t plaintext, uint64_t N)
{
    uint64_t key = desrt::base62_index_to_key(idx);
    uint64_t ct  = encrypt_block(key, plaintext);
    return desrt::reduce_idx(ct, round, table_id, N);
}

} // namespace desrt::des
