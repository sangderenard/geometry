#include "assembly_backend/core.h"
#include "assembly_backend/simd/memory_intrinsics.h"
#include "assembly_backend/simd/simd_dispatch.h"
#include <stdlib.h>
#include <immintrin.h> // Include AVX2 intrinsics
#include "assembly_backend/simd/simd_types.h" // Include SIMD type definitions
#include "geometry/types.h"              // for boolean type
#if defined(_MSC_VER)
  #include <malloc.h>
#endif

void* simd_aligned_alloc(size_t size) {
#if defined(_MSC_VER)
    // Windows aligned alloc
    return _aligned_malloc(size, 64);
#elif defined(__APPLE__) || defined(__unix__)
    void* ptr = NULL;
    if (posix_memalign(&ptr, 64, size) != 0) return NULL;
    return ptr;
#else
    // Fallback to malloc
    return malloc(size);
#endif
}

void simd_aligned_free(void* ptr) {
    if (!ptr) return;
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static size_t hamming_distance(const uint8_t* a, const uint8_t* b, size_t len) {
    size_t diff = 0;
    size_t i = 0;
#if defined(__AVX2__)
    const size_t VEC = 32;
    for (; i + VEC <= len; i += VEC) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i cmp = _mm256_cmpeq_epi8(va, vb);
        unsigned mask = (unsigned)_mm256_movemask_epi8(cmp);
        diff += VEC - (size_t)__builtin_popcount(mask);
    }
#elif defined(__SSE2__)
    const size_t VEC = 16;
    for (; i + VEC <= len; i += VEC) {
        __m128i va = _mm_loadu_si128((const __m128i*)(a + i));
        __m128i vb = _mm_loadu_si128((const __m128i*)(b + i));
        __m128i cmp = _mm_cmpeq_epi8(va, vb);
        int mask = _mm_movemask_epi8(cmp);
        diff += VEC - (size_t)__builtin_popcount((unsigned)mask);
    }
#endif
    for (; i < len; ++i) {
        if (a[i] != b[i]) diff++;
    }
    return diff;
}

size_t simd_byte_diff_count(const uint8_t* a, const uint8_t* b, size_t len) {
    return hamming_distance(a, b, len);
}
// Scatter a single byte value to multiple arbitrary memory locations (fallback)
// ptrs: array of pointers to memory locations
void simd_scatter_byte_c_backup(uint8_t value, void** ptrs, size_t count) {
    if (!ptrs || count == 0) return;
    // Note: AVX2/SSE2 do not support scatter to arbitrary addresses; use simple loop
    for (size_t i = 0; i < count; ++i) {
        uint8_t* dest = (uint8_t*)ptrs[i];
        if (dest) *dest = value;
    }
}
// Primary scatter: try a contiguous "blit" via simd_memset, else fallback
// ptrs: array of pointers to memory locations
void simd_scatter_byte(uint8_t value, void** ptrs, size_t count) {
    if (!ptrs || count == 0) return;
    #if defined(__AVX2__) || defined(__SSE2__)
    uint8_t* base = (uint8_t*)ptrs[0];
    boolean contiguous = true;
    for (size_t i = 1; i < count; ++i) {
        if ((uint8_t*)ptrs[i] != base + i) {
            contiguous = false;
            break;
        }
    }
    if (contiguous) {
        // contiguous span: one SIMD memset
        simd_memset_impl(base, (int)value, count);
        return;
    }
#endif
    // non-contiguous or no SIMD: fallback to C loop
    simd_scatter_byte_c_backup(value, ptrs, count);
}



// Fill a contiguous array of 32-bit values with a constant via SIMD lanes
void simd_fill_u32(uint32_t value, uint32_t* dst, size_t count) {
    if (!dst || count == 0) return;
#if defined(__AVX2__)
    size_t i = 0;
    const size_t LANES = 8; // 8 lanes of 32 bits = 32 bytes
    __m256i v = _mm256_set1_epi32((int)value);
    for (; i + LANES <= count; i += LANES) {
        _mm256_storeu_si256((__m256i*)(dst + i), v);
    }
    for (; i < count; ++i) {
        dst[i] = value;
    }
#elif defined(__SSE2__)
    size_t i = 0;
    const size_t LANES = 4; // 4 lanes of 32 bits = 16 bytes
    __m128i v = _mm_set1_epi32((int)value);
    for (; i + LANES <= count; i += LANES) {
        _mm_storeu_si128((__m128i*)(dst + i), v);
    }
    for (; i < count; ++i) {
        dst[i] = value;
    }
#else
    for (size_t i = 0; i < count; ++i) {
        dst[i] = value;
    }
#endif
}

// Fill dst buffer (total_bytes) by repeating an arbitrary pattern
void simd_fill_pattern(const void* pattern, size_t pattern_size, void* dst, size_t total_bytes) {
    if (!pattern || !dst || pattern_size == 0 || total_bytes == 0) return;
    uint8_t* out = (uint8_t*)dst;
    size_t written = 0;
    // Copy full pattern blocks
    while (written + pattern_size <= total_bytes) {
        simd_memcpy_impl(out + written, pattern, pattern_size);
        written += pattern_size;
    }
    // Copy remaining tail
    size_t tail = total_bytes - written;
    if (tail > 0) {
        // pattern may be larger than tail, so we copy only tail bytes
        memcpy(out + written, pattern, tail);
    }
}
