#include "assembly_backend/simd/memory_intrinsics.h"
#include "assembly_backend/simd/simd_dispatch.h"
#include <stdlib.h>
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
