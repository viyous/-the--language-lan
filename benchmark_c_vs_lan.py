"""Head-to-head performance benchmark: Lan vs idiomatic unsafe C, defensive C, and Lan --debug-bounds."""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from lanc import compile_source

LAN_KERNELS = r"""
enum SearchErr: uint16 { Missing = 65535 };

define dot_product(a: uint16[1024], b: uint16[1024]) -> uint32 {
    define acc: uint32 = 0;
    interation(i; 0; uint16; 1024; 1) {
        acc = (acc + (a[i] as uint32) * (b[i] as uint32)) as: wrap uint32;
    }
    return acc;
}

define insertion_sort(a: uint16[64]) -> void {
    define i: uint8 = 1;
    while (i < 64) {
        define key: uint16 = a[i];
        define j: uint8 = i;
        define moving: boolean = true;
        while (moving && j > 0) {
            define prev: uint8 = j - 1;
            if (a[prev] > key) {
                a[j] = a[prev];
                j = prev;
            } else {
                moving = false;
            }
        }
        a[j] = key;
        i += 1;
    }
}

define matmul_8x8(out: uint32[64], a: uint16[0, 255][64], b: uint16[0, 255][64]) -> void {
    define i: uint16 = 0;
    while (i < 8) {
        define row_off: uint16 = i * 8;
        define j: uint16 = 0;
        while (j < 8) {
            define acc: uint32 = 0;
            define k: uint16 = 0;
            while (k < 8) {
                define aik: uint32 = a[row_off + k];
                define bkj: uint32 = b[k * 8 + j];
                acc = (acc + aik * bkj) as: wrap uint32;
                k += 1;
            }
            out[row_off + j] = acc;
            j += 1;
        }
        i += 1;
    }
}

define slice_copy_disjoint(buf: uint8[2048], src_off: uint16[0, 1024], dst_off: uint16[0, 1024], n: uint16[1, 1024]) -> uint32 {
    if (src_off + n <= dst_off && dst_off + n <= 2048) {
        buf[dst_off, dst_off + n) = buf[src_off, src_off + n) as: copy;
        return buf[dst_off] + buf[dst_off + n - 1];
    }
    return 0;
}

define binary_search(arr: uint32[256], target: uint32) -> uint16[0, 255] | SearchErr {
    define lo: uint16[0, 256] = 0;
    define hi: uint16[0, 256] = 256;
    define found: uint16[0, 255] | SearchErr = SearchErr.Missing;
    define going: boolean = true;
    while (going && lo < hi) {
        define mid: uint16 = (lo + hi) >> 1;
        define val: uint32 = arr[mid];
        if (val == target) {
            found = mid;
            going = false;
        } else if (val < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return found;
}

define bitfield_be(raw: uint64) -> uint32 {
    define bits: bit[64] = raw as: view bit[64];
    define hdr = bits[
        seq: uint32@bit(32, 0, endian: be),
        plen: uint16@bit(16, 32, endian: be),
        tag: uint16@bit(16, 48, endian: be)
    ];
    define sum: uint32 = (hdr.tag as uint32) + (hdr.plen as uint32);
    return (sum ^ hdr.seq) as: wrap uint32;
}

define clamp100(x: uint16) -> uint8[0, 100]
    property: forall(x) => clamp100(clamp100(x)) == clamp100(x) {
    if (x > 100) { return 100; }
    return x;
}

define consumer_narrow(v: uint8[0, 100]) -> uint16 {
    if (v > 200) {
        return 9999;
    }
    return (v as uint16) * 3 + ((v as uint16) >> 1);
}

define clamp_pipeline(x: uint16) -> uint16 {
    define c: uint8[0, 100] = clamp100(clamp100(x));
    return consumer_narrow(c);
}
"""

LAN_CROSS_LIB = r"""
define ext_clamp100(x: uint16) -> uint8[0, 100]
    property: forall(x) => ext_clamp100(ext_clamp100(x)) == ext_clamp100(x) {
    if (x > 100) { return 100; }
    return x;
}

export { ext_clamp100 };
"""

LAN_CROSS_CALLER = r"""
import { ext_clamp100 } from "./cross_lib.table";

define cross_clamp_double(x: uint16) -> uint16 {
    define c: uint8[0, 100] = ext_clamp100(ext_clamp100(x));
    if (c > 200) {
        return 9999;
    }
    return (c as uint16) * 3 + ((c as uint16) >> 1);
}
"""

C_COMMON_HELPERS = r"""
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
"""

C_FAST_SOURCE = C_COMMON_HELPERS + r"""
uint32_t c_fast_dot_product(const uint16_t *a, const uint16_t *b) {
    uint32_t acc = 0;
    for (size_t i = 0; i < 1024; ++i) {
        acc += (uint32_t)a[i] * (uint32_t)b[i];
    }
    return acc;
}

void c_fast_insertion_sort(uint16_t *a) {
    for (size_t i = 1; i < 64; ++i) {
        uint16_t key = a[i];
        size_t j = i;
        while (j > 0 && a[j - 1] > key) {
            a[j] = a[j - 1];
            --j;
        }
        a[j] = key;
    }
}

void c_fast_matmul_8x8(uint32_t *out, const uint16_t *a, const uint16_t *b) {
    for (size_t i = 0; i < 8; ++i) {
        size_t row_off = i * 8;
        for (size_t j = 0; j < 8; ++j) {
            uint32_t acc = 0;
            for (size_t k = 0; k < 8; ++k) {
                acc += (uint32_t)a[row_off + k] * (uint32_t)b[k * 8 + j];
            }
            out[row_off + j] = acc;
        }
    }
}

uint32_t c_fast_slice_copy_disjoint(uint8_t *buf, uint16_t src_off, uint16_t dst_off, uint16_t n) {
    if ((uint32_t)src_off + n <= dst_off && (uint32_t)dst_off + n <= 2048u) {
        memcpy(buf + dst_off, buf + src_off, n);
        return (uint32_t)buf[dst_off] + (uint32_t)buf[dst_off + n - 1];
    }
    return 0;
}

uint16_t c_fast_binary_search(const uint32_t *arr, uint32_t target) {
    uint16_t lo = 0;
    uint16_t hi = 256;
    while (lo < hi) {
        uint16_t mid = (uint16_t)(((uint32_t)lo + (uint32_t)hi) >> 1);
        uint32_t val = arr[mid];
        if (val == target) {
            return mid;
        } else if (val < target) {
            lo = (uint16_t)(mid + 1);
        } else {
            hi = mid;
        }
    }
    return 65535u;
}

static inline uint16_t read_u16_be(const uint8_t *p) {
    return (uint16_t)(((uint16_t)p[0] << 8) | (uint16_t)p[1]);
}

static inline uint32_t read_u32_be(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

uint32_t c_fast_bitfield_be(uint64_t raw) {
    uint8_t bytes[8];
    memcpy(bytes, &raw, 8);
    uint32_t seq = read_u32_be(bytes);
    uint16_t len = read_u16_be(bytes + 4);
    uint16_t tag = read_u16_be(bytes + 6);
    uint32_t sum = (uint32_t)tag + (uint32_t)len;
    return sum ^ seq;
}

static uint8_t c_fast_clamp100(uint16_t x) {
    return x > 100 ? 100 : (uint8_t)x;
}

static uint16_t c_fast_consumer_narrow(uint8_t v) {
    if (v > 200) return 9999;
    return (uint16_t)v * 3u + (uint16_t)(v >> 1);
}

uint16_t c_fast_clamp_pipeline(uint16_t x) {
    uint8_t c = c_fast_clamp100(c_fast_clamp100(x));
    return c_fast_consumer_narrow(c);
}

/* Cross-TU function defined in bench_c_cross_lib.c */
uint8_t c_ext_clamp100(uint16_t x);

uint16_t c_fast_cross_clamp_double(uint16_t x) {
    uint8_t c = c_ext_clamp100(c_ext_clamp100(x));
    if (c > 200) return 9999;
    return (uint16_t)c * 3u + (uint16_t)(c >> 1);
}
"""

C_CROSS_LIB_SOURCE = r"""
#include <stdint.h>
uint8_t c_ext_clamp100(uint16_t x) {
    return x > 100 ? 100 : (uint8_t)x;
}
"""

C_SAFE_SOURCE = C_COMMON_HELPERS + r"""
static inline uint64_t safe_idx(int64_t i, size_t len) {
    if (i < 0 || (uint64_t)i >= len) abort();
    return (uint64_t)i;
}

uint32_t c_safe_dot_product(const uint16_t *a, const uint16_t *b) {
    if (!a || !b) abort();
    uint32_t acc = 0;
    for (size_t i = 0; i < 1024; ++i) {
        uint64_t idx = safe_idx((int64_t)i, 1024);
        acc += (uint32_t)a[idx] * (uint32_t)b[idx];
    }
    return acc;
}

void c_safe_insertion_sort(uint16_t *a) {
    if (!a) abort();
    for (size_t i = 1; i < 64; ++i) {
        uint16_t key = a[safe_idx((int64_t)i, 64)];
        size_t j = i;
        while (j > 0 && a[safe_idx((int64_t)(j - 1), 64)] > key) {
            a[safe_idx((int64_t)j, 64)] = a[safe_idx((int64_t)(j - 1), 64)];
            --j;
        }
        a[safe_idx((int64_t)j, 64)] = key;
    }
}

void c_safe_matmul_8x8(uint32_t *out, const uint16_t *a, const uint16_t *b) {
    if (!out || !a || !b) abort();
    for (size_t i = 0; i < 8; ++i) {
        size_t row_off = i * 8;
        for (size_t j = 0; j < 8; ++j) {
            uint32_t acc = 0;
            for (size_t k = 0; k < 8; ++k) {
                uint16_t av = a[safe_idx((int64_t)(row_off + k), 64)];
                uint16_t bv = b[safe_idx((int64_t)(k * 8 + j), 64)];
                if (av > 255 || bv > 255) abort();
                acc += (uint32_t)av * (uint32_t)bv;
            }
            out[safe_idx((int64_t)(row_off + j), 64)] = acc;
        }
    }
}

uint32_t c_safe_slice_copy_disjoint(uint8_t *buf, uint16_t src_off, uint16_t dst_off, uint16_t n) {
    if (!buf || n < 1 || n > 1024 || src_off > 1024 || dst_off > 1024) abort();
    if ((uint32_t)src_off + n <= dst_off && (uint32_t)dst_off + n <= 2048u) {
        if ((uint64_t)dst_off + n > 2048u || (uint64_t)src_off + n > 2048u) abort();
        memmove(buf + dst_off, buf + src_off, n);
        return (uint32_t)buf[safe_idx(dst_off, 2048)] + (uint32_t)buf[safe_idx(dst_off + n - 1, 2048)];
    }
    return 0;
}

uint16_t c_safe_binary_search(const uint32_t *arr, uint32_t target) {
    if (!arr) abort();
    uint16_t lo = 0;
    uint16_t hi = 256;
    while (lo < hi) {
        uint16_t mid = (uint16_t)(((uint32_t)lo + (uint32_t)hi) >> 1);
        uint32_t val = arr[safe_idx(mid, 256)];
        if (val == target) {
            return mid;
        } else if (val < target) {
            lo = (uint16_t)(mid + 1);
        } else {
            hi = mid;
        }
    }
    return 65535u;
}

uint32_t c_safe_bitfield_be(uint64_t raw) {
    uint8_t bytes[8];
    memcpy(bytes, &raw, 8);
    uint32_t seq = ((uint32_t)bytes[safe_idx(0, 8)] << 24) | ((uint32_t)bytes[safe_idx(1, 8)] << 16) |
                   ((uint32_t)bytes[safe_idx(2, 8)] << 8) | (uint32_t)bytes[safe_idx(3, 8)];
    uint16_t len = (uint16_t)(((uint16_t)bytes[safe_idx(4, 8)] << 8) | (uint16_t)bytes[safe_idx(5, 8)]);
    uint16_t tag = (uint16_t)(((uint16_t)bytes[safe_idx(6, 8)] << 8) | (uint16_t)bytes[safe_idx(7, 8)]);
    uint32_t sum = (uint32_t)tag + (uint32_t)len;
    return sum ^ seq;
}

static uint8_t c_safe_clamp100(uint16_t x) {
    return x > 100 ? 100 : (uint8_t)x;
}

static uint16_t c_safe_consumer_narrow(uint8_t v) {
    if (v > 100) abort();
    if (v > 200) return 9999;
    return (uint16_t)v * 3u + (uint16_t)(v >> 1);
}

uint16_t c_safe_clamp_pipeline(uint16_t x) {
    uint8_t c = c_safe_clamp100(c_safe_clamp100(x));
    return c_safe_consumer_narrow(c);
}

uint8_t c_ext_clamp100(uint16_t x);

uint16_t c_safe_cross_clamp_double(uint16_t x) {
    uint8_t c = c_ext_clamp100(c_ext_clamp100(x));
    if (c > 100) abort();
    if (c > 200) return 9999;
    return (uint16_t)c * 3u + (uint16_t)(c >> 1);
}
"""

C_DRIVER_TEMPLATE = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
static double now_sec(void) {
    static LARGE_INTEGER freq;
    LARGE_INTEGER cur;
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cur);
    return (double)cur.QuadPart / (double)freq.QuadPart;
}
#else
#include <time.h>
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

static volatile uint64_t g_sink = 0;

/* Symbols from Lan, Lan --debug-bounds, C-fast, C-safe */
uint32_t lan_fn_dot_product(uint16_t *a, uint16_t *b);
void lan_fn_insertion_sort(uint16_t *a);
void lan_fn_matmul_8x8(uint32_t *out, uint16_t *a, uint16_t *b);
uint32_t lan_fn_slice_copy_disjoint(uint8_t *buf, uint16_t src_off, uint16_t dst_off, uint16_t n);
uint16_t lan_fn_binary_search(uint32_t *arr, uint32_t target);
uint32_t lan_fn_bitfield_be(uint64_t raw);
uint16_t lan_fn_clamp_pipeline(uint16_t x);
uint16_t lan_fn_cross_clamp_double(uint16_t x);

uint32_t dbg_lan_fn_dot_product(uint16_t *a, uint16_t *b);
void dbg_lan_fn_insertion_sort(uint16_t *a);
void dbg_lan_fn_matmul_8x8(uint32_t *out, uint16_t *a, uint16_t *b);
uint32_t dbg_lan_fn_slice_copy_disjoint(uint8_t *buf, uint16_t src_off, uint16_t dst_off, uint16_t n);
uint16_t dbg_lan_fn_binary_search(uint32_t *arr, uint32_t target);
uint32_t dbg_lan_fn_bitfield_be(uint64_t raw);
uint16_t dbg_lan_fn_clamp_pipeline(uint16_t x);
uint16_t dbg_lan_fn_cross_clamp_double(uint16_t x);

uint32_t c_fast_dot_product(const uint16_t *a, const uint16_t *b);
void c_fast_insertion_sort(uint16_t *a);
void c_fast_matmul_8x8(uint32_t *out, const uint16_t *a, const uint16_t *b);
uint32_t c_fast_slice_copy_disjoint(uint8_t *buf, uint16_t src_off, uint16_t dst_off, uint16_t n);
uint16_t c_fast_binary_search(const uint32_t *arr, uint32_t target);
uint32_t c_fast_bitfield_be(uint64_t raw);
uint16_t c_fast_clamp_pipeline(uint16_t x);
uint16_t c_fast_cross_clamp_double(uint16_t x);

uint32_t c_safe_dot_product(const uint16_t *a, const uint16_t *b);
void c_safe_insertion_sort(uint16_t *a);
void c_safe_matmul_8x8(uint32_t *out, const uint16_t *a, const uint16_t *b);
uint32_t c_safe_slice_copy_disjoint(uint8_t *buf, uint16_t src_off, uint16_t dst_off, uint16_t n);
uint16_t c_safe_binary_search(const uint32_t *arr, uint32_t target);
uint32_t c_safe_bitfield_be(uint64_t raw);
uint16_t c_safe_clamp_pipeline(uint16_t x);
uint16_t c_safe_cross_clamp_double(uint16_t x);

static uint64_t run_dot(int impl, long iters) {
    static uint16_t a[1024], b[1024];
    for (int i = 0; i < 1024; ++i) {
        a[i] = (uint16_t)((i * 17 + 3) & 1023);
        b[i] = (uint16_t)((i * 31 + 7) & 1023);
    }
    uint64_t acc = 0;
    for (long it = 0; it < iters; ++it) {
        uint32_t r = 0;
        if (impl == 0) r = lan_fn_dot_product(a, b);
        else if (impl == 1) r = dbg_lan_fn_dot_product(a, b);
        else if (impl == 2) r = c_fast_dot_product(a, b);
        else r = c_safe_dot_product(a, b);
        acc += r;
        a[it & 1023] = (uint16_t)((a[it & 1023] + 1) & 1023);
    }
    g_sink ^= acc;
    return acc;
}

static uint64_t run_sort(int impl, long iters) {
    uint16_t seed[64], buf[64];
    for (int i = 0; i < 64; ++i) seed[i] = (uint16_t)((64 - i) * 37 ^ (i * 13));
    uint64_t acc = 0;
    for (long it = 0; it < iters; ++it) {
        memcpy(buf, seed, sizeof(buf));
        buf[0] = (uint16_t)(it & 0xffff);
        if (impl == 0) lan_fn_insertion_sort(buf);
        else if (impl == 1) dbg_lan_fn_insertion_sort(buf);
        else if (impl == 2) c_fast_insertion_sort(buf);
        else c_safe_insertion_sort(buf);
        acc += buf[0] + buf[31] * 3u + buf[63] * 7u;
    }
    g_sink ^= acc;
    return acc;
}

static uint64_t run_matmul(int impl, long iters) {
    uint16_t a[64], b[64];
    uint32_t out[64];
    for (int i = 0; i < 64; ++i) {
        a[i] = (uint16_t)((i * 11 + 5) & 255);
        b[i] = (uint16_t)((i * 19 + 9) & 255);
    }
    uint64_t acc = 0;
    for (long it = 0; it < iters; ++it) {
        a[it & 63] = (uint16_t)((a[it & 63] + 1) & 255);
        if (impl == 0) lan_fn_matmul_8x8(out, a, b);
        else if (impl == 1) dbg_lan_fn_matmul_8x8(out, a, b);
        else if (impl == 2) c_fast_matmul_8x8(out, a, b);
        else c_safe_matmul_8x8(out, a, b);
        acc += out[0] + out[27] + out[63];
    }
    g_sink ^= acc;
    return acc;
}

static uint64_t run_copy(int impl, long iters) {
    static uint8_t buf[2048];
    for (int i = 0; i < 2048; ++i) buf[i] = (uint8_t)(i * 7 + 1);
    uint64_t acc = 0;
    for (long it = 0; it < iters; ++it) {
        uint16_t n = (uint16_t)(512 + (it & 511));
        uint32_t r = 0;
        if (impl == 0) r = lan_fn_slice_copy_disjoint(buf, 0, 1024, n);
        else if (impl == 1) r = dbg_lan_fn_slice_copy_disjoint(buf, 0, 1024, n);
        else if (impl == 2) r = c_fast_slice_copy_disjoint(buf, 0, 1024, n);
        else r = c_safe_slice_copy_disjoint(buf, 0, 1024, n);
        acc += r;
        buf[it & 511] ^= (uint8_t)it;
    }
    g_sink ^= acc;
    return acc;
}

static uint64_t run_bsearch(int impl, long iters) {
    uint32_t arr[256];
    for (int i = 0; i < 256; ++i) arr[i] = (uint32_t)(i * 17 + 5);
    uint64_t acc = 0;
    for (long it = 0; it < iters; ++it) {
        uint32_t target = (uint32_t)(((it * 73) % 270) * 17 + 5);
        uint16_t r = 0;
        if (impl == 0) r = lan_fn_binary_search(arr, target);
        else if (impl == 1) r = dbg_lan_fn_binary_search(arr, target);
        else if (impl == 2) r = c_fast_binary_search(arr, target);
        else r = c_safe_binary_search(arr, target);
        acc += r;
    }
    g_sink ^= acc;
    return acc;
}

static uint64_t run_bitfield(int impl, long iters) {
    uint64_t acc = 0;
    uint64_t x = 0x123456789abcdef0ULL;
    for (long it = 0; it < iters; ++it) {
        uint32_t r = 0;
        if (impl == 0) r = lan_fn_bitfield_be(x);
        else if (impl == 1) r = dbg_lan_fn_bitfield_be(x);
        else if (impl == 2) r = c_fast_bitfield_be(x);
        else r = c_safe_bitfield_be(x);
        acc += r;
        x = x * 6364136223846793005ULL + 1ULL;
    }
    g_sink ^= acc;
    return acc;
}

static uint64_t run_clamp(int impl, long iters) {
    uint64_t acc = 0;
    for (long it = 0; it < iters; ++it) {
        uint16_t x = (uint16_t)((it * 29) & 1023);
        uint16_t r = 0;
        if (impl == 0) r = lan_fn_clamp_pipeline(x);
        else if (impl == 1) r = dbg_lan_fn_clamp_pipeline(x);
        else if (impl == 2) r = c_fast_clamp_pipeline(x);
        else r = c_safe_clamp_pipeline(x);
        acc += r;
    }
    g_sink ^= acc;
    return acc;
}

static uint64_t run_cross(int impl, long iters) {
    uint64_t acc = 0;
    for (long it = 0; it < iters; ++it) {
        uint16_t x = (uint16_t)((it * 29) & 1023);
        uint16_t r = 0;
        if (impl == 0) r = lan_fn_cross_clamp_double(x);
        else if (impl == 1) r = dbg_lan_fn_cross_clamp_double(x);
        else if (impl == 2) r = c_fast_cross_clamp_double(x);
        else r = c_safe_cross_clamp_double(x);
        acc += r;
    }
    g_sink ^= acc;
    return acc;
}

typedef uint64_t (*bench_fn)(int, long);

int main(int argc, char **argv) {
    int rounds = argc > 1 ? atoi(argv[1]) : 5;
    long scale = argc > 2 ? atol(argv[2]) : 1;
    struct {
        const char *name;
        bench_fn fn;
        long base_iters;
    } cases[] = {
        {"dot_product_1024",    run_dot,      300000L},
        {"insertion_sort_64",   run_sort,     400000L},
        {"matmul_8x8",          run_matmul,   600000L},
        {"slice_copy_disjoint", run_copy,     3000000L},
        {"binary_search_256",   run_bsearch,  3000000L},
        {"bitfield_be_64",      run_bitfield, 12000000L},
        {"clamp_pipeline",      run_clamp,    15000000L},
        {"cross_tu_idempotent", run_cross,    15000000L},
    };
    int ncases = (int)(sizeof(cases) / sizeof(cases[0]));
    for (int i = 0; i < ncases; ++i) {
        long iters = cases[i].base_iters * scale;
        uint64_t sums[4] = {0};
        for (int impl = 0; impl < 4; ++impl) {
            sums[impl] = cases[i].fn(impl, iters / 10 > 0 ? iters / 10 : 1);
        }
        if (sums[0] != sums[1] || sums[0] != sums[2] || sums[0] != sums[3]) {
            fprintf(stderr, "CHECKSUM MISMATCH on %s: lan=%llu dbg=%llu c_fast=%llu c_safe=%llu\n",
                    cases[i].name, (unsigned long long)sums[0], (unsigned long long)sums[1],
                    (unsigned long long)sums[2], (unsigned long long)sums[3]);
            return 2;
        }
        for (int r = 0; r < rounds; ++r) {
            for (int offset = 0; offset < 4; ++offset) {
                int impl = (r + offset) & 3;
                double t0 = now_sec();
                uint64_t chk = cases[i].fn(impl, iters);
                double dt = now_sec() - t0;
                printf("RESULT %s %d %d %.9f %llu %ld\n",
                       cases[i].name, impl, r, dt, (unsigned long long)chk, iters);
            }
        }
    }
    return 0;
}
"""


def find_c_compiler(preferred=None):
    if preferred:
        return preferred
    for cand in ("zig", "clang", "gcc"):
        found = shutil.which(cand)
        if found:
            return "zig cc" if cand == "zig" else found
    raise RuntimeError("No C compiler (zig, clang, or gcc) found on PATH")


def prefix_lan_debug_symbols(c_source: str) -> str:
    """Rename Lan C symbols to dbg_* so static and --debug-bounds link side-by-side."""
    c_source = re.sub(r"\blan_fn_", "dbg_lan_fn_", c_source)
    c_source = re.sub(r"\blan_api_", "dbg_lan_api_", c_source)
    c_source = re.sub(r"\blh_", "dbg_lh_", c_source)
    c_source = re.sub(r"\blan_array_", "dbg_lan_array_", c_source)
    c_source = re.sub(r"\bLAN_ARRAY_", "DBG_LAN_ARRAY_", c_source)
    return c_source


def run_lanc_file(source_path: Path, output_c: Path, *, library=False, debug_bounds=False):
    lanc_py = Path(__file__).resolve().parent / "lanc.py"
    cmd = [sys.executable, "-B", str(lanc_py), str(source_path), "-o", str(output_c)]
    if library:
        cmd.append("--library")
    if debug_bounds:
        cmd.append("--debug-bounds")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"lanc failed for {source_path.name}:\n{res.stderr}\n{res.stdout}")


def compile_c_obj(compiler_cmd, src_path: Path, obj_path: Path, flags: list[str]):
    cmd = [*compiler_cmd.split(), "-std=gnu11", *flags, "-c", str(src_path), "-o", str(obj_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"C compile failed for {src_path.name}:\n{res.stderr}\n{res.stdout}")


def link_exe(compiler_cmd, objs: list[Path], exe_path: Path, flags: list[str]):
    cmd = [*compiler_cmd.split(), *flags, *(str(o) for o in objs), "-o", str(exe_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Link failed:\n{res.stderr}\n{res.stdout}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", help="C compiler command (e.g. 'zig cc', 'clang', 'gcc')")
    parser.add_argument("--opt", default="O2", choices=("O1", "O2", "O3"))
    parser.add_argument("--lto", action="store_true", help="Enable -flto across translation units")
    parser.add_argument("--rounds", type=int, default=7, help="Timed rounds per benchmark")
    parser.add_argument("--scale", type=int, default=1, help="Iteration multiplier")
    parser.add_argument("--compile-only", action="store_true",
                        help="Only compile Lan -> C (for sandbox check without external C toolchain)")
    parser.add_argument("--keep-dir", type=Path, help="Keep generated C/obj/exe files in this directory")
    args = parser.parse_args()

    ctx = tempfile.TemporaryDirectory() if args.keep_dir is None else None
    workdir = Path(ctx.name) if ctx else args.keep_dir
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        t0 = time.perf_counter()
        lan_c, _ = compile_source(LAN_KERNELS, debug_bounds=False)
        t_lan_compile = time.perf_counter() - t0

        t1 = time.perf_counter()
        lan_dbg_c, _ = compile_source(LAN_KERNELS, debug_bounds=True)
        lan_dbg_c = prefix_lan_debug_symbols(lan_dbg_c)
        t_dbg_compile = time.perf_counter() - t1

        cross_lib_lang = workdir / "cross_lib.lang"
        cross_caller_lang = workdir / "cross_caller.lang"
        cross_lib_lang.write_text(LAN_CROSS_LIB, encoding="utf-8")
        cross_caller_lang.write_text(LAN_CROSS_CALLER, encoding="utf-8")

        cross_lib_c = workdir / "cross_lib.c"
        cross_caller_c = workdir / "cross_caller.c"
        run_lanc_file(cross_lib_lang, cross_lib_c, library=True, debug_bounds=False)
        run_lanc_file(cross_caller_lang, cross_caller_c, debug_bounds=False)

        dbg_dir = workdir / "dbg_cross"
        dbg_dir.mkdir(exist_ok=True)
        (dbg_dir / "cross_lib.lang").write_text(LAN_CROSS_LIB, encoding="utf-8")
        (dbg_dir / "cross_caller.lang").write_text(LAN_CROSS_CALLER, encoding="utf-8")
        dbg_lib_c = dbg_dir / "cross_lib.c"
        dbg_caller_c = dbg_dir / "cross_caller.c"
        run_lanc_file(dbg_dir / "cross_lib.lang", dbg_lib_c, library=True, debug_bounds=True)
        run_lanc_file(dbg_dir / "cross_caller.lang", dbg_caller_c, debug_bounds=True)
        dbg_lib_c.write_text(prefix_lan_debug_symbols(dbg_lib_c.read_text(encoding="utf-8")), encoding="utf-8")
        dbg_caller_c.write_text(prefix_lan_debug_symbols(dbg_caller_c.read_text(encoding="utf-8")), encoding="utf-8")

        print(f"Lan static compile : {t_lan_compile * 1000:.1f} ms ({len(lan_c):,} bytes C)")
        print(f"Lan debug compile  : {t_dbg_compile * 1000:.1f} ms ({len(lan_dbg_c):,} bytes C)")
        if args.compile_only:
            return 0

        cc = find_c_compiler(args.cc)
        cflags = [f"-{args.opt}"] + (["-flto"] if args.lto else [])
        lflags = ["-flto"] if args.lto else []
        ext = ".exe" if platform.system() == "Windows" else ""

        files = {
            "lan": (workdir / "bench_lan.c", lan_c),
            "lan_dbg": (workdir / "bench_lan_dbg.c", lan_dbg_c),
            "lan_cross_lib": (cross_lib_c, cross_lib_c.read_text(encoding="utf-8")),
            "lan_cross_caller": (cross_caller_c, cross_caller_c.read_text(encoding="utf-8")),
            "dbg_cross_lib": (dbg_lib_c, dbg_lib_c.read_text(encoding="utf-8")),
            "dbg_cross_caller": (dbg_caller_c, dbg_caller_c.read_text(encoding="utf-8")),
            "c_fast": (workdir / "bench_c_fast.c", C_FAST_SOURCE),
            "c_cross_lib": (workdir / "bench_c_cross_lib.c", C_CROSS_LIB_SOURCE),
            "c_safe": (workdir / "bench_c_safe.c", C_SAFE_SOURCE),
            "driver": (workdir / "bench_driver.c", C_DRIVER_TEMPLATE),
        }
        objs = {}
        t_cc_start = time.perf_counter()
        for key, (path, content) in files.items():
            path.write_text(content, encoding="utf-8")
            obj = path.with_suffix(".o")
            compile_c_obj(cc, path, obj, cflags)
            objs[key] = obj
        exe = workdir / f"bench_runner{ext}"
        link_exe(cc, list(objs.values()), exe, lflags)
        t_cc = time.perf_counter() - t_cc_start

        print(f"C compiler         : {cc} ({' '.join(cflags)}), build={t_cc * 1000:.1f} ms")
        print(
            "Object sizes (bytes): "
            + ", ".join(f"{k}={objs[k].stat().st_size:,}" for k in ("lan", "lan_dbg", "c_fast", "c_safe"))
        )

        run = subprocess.run([str(exe), str(args.rounds), str(args.scale)], capture_output=True, text=True)
        if run.returncode != 0:
            raise RuntimeError(f"Benchmark binary failed ({run.returncode}):\n{run.stderr}\n{run.stdout}")

        timings = {}
        iters_map = {}
        checksums = {}
        for line in run.stdout.splitlines():
            parts = line.split()
            if len(parts) == 7 and parts[0] == "RESULT":
                _, name, impl_s, _, dt_s, chk, it_s = parts
                impl = int(impl_s)
                timings.setdefault((name, impl), []).append(float(dt_s))
                iters_map[name] = int(it_s)
                checksums.setdefault(name, set()).add(chk)

        print("\n=== Runtime Comparison (median of %d rounds, lower is better) ===" % args.rounds)
        header = (
            f"{'Benchmark':22s} | {'Iters':>10s} | "
            f"{'Lan(ms)':>9s} | {'C-fast(ms)':>10s} | {'C-safe(ms)':>10s} | {'Lan-dbg(ms)':>11s} | "
            f"{'Lan/C-fast':>10s} | {'C-safe/Lan':>10s}"
        )
        print(header)
        print("-" * len(header))
        ratios_fast, ratios_safe = [], []
        for name in iters_map:
            assert len(checksums[name]) == 1, f"Checksum mismatch for {name}: {checksums[name]}"
            med = {i: statistics.median(timings[name, i]) * 1000.0 for i in range(4)}
            r_fast = med[0] / med[2] if med[2] > 0 else 1.0
            r_safe = med[3] / med[0] if med[0] > 0 else 1.0
            ratios_fast.append(r_fast)
            ratios_safe.append(r_safe)
            print(
                f"{name:22s} | {iters_map[name]:10,d} | "
                f"{med[0]:9.2f} | {med[2]:10.2f} | {med[3]:10.2f} | {med[1]:11.2f} | "
                f"{r_fast:9.2f}x | {r_safe:9.2f}x"
            )
        print("-" * len(header))
        print(
            f"Geometric mean ratio: Lan / C-fast = {statistics.geometric_mean(ratios_fast):.2f}x  |  "
            f"C-safe / Lan = {statistics.geometric_mean(ratios_safe):.2f}x"
        )
        print("Checksums verified identical across all 4 variants for every benchmark.")
    finally:
        if ctx is not None:
            ctx.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
