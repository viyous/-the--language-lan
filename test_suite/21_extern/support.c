/* 21_extern 的 C 侧实现：签名与 lanc 生成的 extern 声明一致。 */
#include <stdint.h>

uint32_t ext_popcount32(uint32_t x) {
    uint32_t n = 0;
    while (x) { n += x & 1u; x >>= 1; }
    return n;
}

int32_t ext_clamp(int32_t v, int32_t lo, int32_t hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

int32_t ext_sign(int32_t v) {
    if (v < 0) return -1;
    if (v > 0) return 1;
    return 0;
}

uint64_t ext_sum_bytes(const uint8_t *data, uint64_t n) {
    uint64_t s = 0;
    for (uint64_t i = 0; i < n; i++) s += data[i];
    return s;
}

uint32_t ext_hash(const uint8_t *data, uint64_t n) {
    uint32_t h = 2166136261u;
    for (uint64_t i = 0; i < n; i++) { h ^= data[i]; h *= 16777619u; }
    return h;
}

void ext_fill(uint8_t *dst, uint64_t n, uint8_t v) {
    for (uint64_t i = 0; i < n; i++) dst[i] = v;
}

uint8_t ext_max4(const uint8_t *a) {
    uint8_t m = a[0];
    for (int i = 1; i < 4; i++) if (a[i] > m) m = a[i];
    return m;
}

uint64_t ext_mix(uint64_t a, uint64_t b) {
    return (a << 32) ^ b;
}

int32_t ext_clamp100(int32_t v) {
    if (v < 0) return 0;
    if (v > 100) return 100;
    return v;
}
