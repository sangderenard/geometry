#include "assembly_backend/simd/memory_intrinsics.h"
#include "assembly_backend/simd/simd_dispatch.h"
#include <immintrin.h>
#include <string.h>

void* simd_aligned_alloc(size_t size) {
    void* ptr = _mm_malloc(size, 64);
    if (ptr) simd_memset_impl(ptr, 0, size);
    return ptr;
}

void simd_aligned_free(void* ptr) {
    if (ptr) _mm_free(ptr);
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
