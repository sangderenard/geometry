#include "geometry/simd_ops.h"

#include <string.h> // memcpy, memset

#if defined(__AVX2__)
#include <immintrin.h>

// --- AVX2 Implementations ---

void simd_add_u8_saturate(uint8_t* dst, const uint8_t* src, size_t count) {
    size_t i = 0;
    const size_t VEC_SIZE = 32;

    for (; i + VEC_SIZE <= count; i += VEC_SIZE) {
        __m256i v_dst = _mm256_loadu_si256((__m256i*)(dst + i));
        __m256i v_src = _mm256_loadu_si256((__m256i*)(src + i));
        __m256i v_result = _mm256_adds_epu8(v_dst, v_src);
        _mm256_storeu_si256((__m256i*)(dst + i), v_result);
    }

    for (; i < count; ++i) {
        unsigned int sum = dst[i] + src[i];
        dst[i] = (sum > 255) ? 255 : (uint8_t)sum;
    }
}

void simd_add_u16_saturate(uint16_t* dst, const uint16_t* src, size_t count) {
    size_t i = 0;
    const size_t VEC_SIZE = 16;

    for (; i + VEC_SIZE <= count; i += VEC_SIZE) {
        __m256i v_dst = _mm256_loadu_si256((__m256i*)(dst + i));
        __m256i v_src = _mm256_loadu_si256((__m256i*)(src + i));
        __m256i v_result = _mm256_adds_epu16(v_dst, v_src);
        _mm256_storeu_si256((__m256i*)(dst + i), v_result);
    }

    for (; i < count; ++i) {
        unsigned int sum = dst[i] + src[i];
        dst[i] = (sum > 65535) ? 65535 : (uint16_t)sum;
    }
}

void simd_add_s8_saturate(int8_t* dst, const int8_t* src, size_t count) {
    size_t i = 0;
    const size_t VEC_SIZE = 32;

    for (; i + VEC_SIZE <= count; i += VEC_SIZE) {
        __m256i v_dst = _mm256_loadu_si256((__m256i*)(dst + i));
        __m256i v_src = _mm256_loadu_si256((__m256i*)(src + i));
        __m256i v_result = _mm256_adds_epi8(v_dst, v_src);
        _mm256_storeu_si256((__m256i*)(dst + i), v_result);
    }

    for (; i < count; ++i) {
        int sum = dst[i] + src[i];
        dst[i] = (sum > 127) ? 127 : ((sum < -128) ? -128 : (int8_t)sum);
    }
}

void simd_add_f32(float* dst, const float* src, size_t count) {
    size_t i = 0;
    const size_t VEC_SIZE = 8;

    for (; i + VEC_SIZE <= count; i += VEC_SIZE) {
        __m256 v_dst = _mm256_loadu_ps(dst + i);
        __m256 v_src = _mm256_loadu_ps(src + i);
        __m256 v_result = _mm256_add_ps(v_dst, v_src);
        _mm256_storeu_ps(dst + i, v_result);
    }

    for (; i < count; ++i) dst[i] += src[i];
}

void simd_add_f64(double* dst, const double* src, size_t count) {
    size_t i = 0;
    const size_t VEC_SIZE = 4;

    for (; i + VEC_SIZE <= count; i += VEC_SIZE) {
        __m256d v_dst = _mm256_loadu_pd(dst + i);
        __m256d v_src = _mm256_loadu_pd(src + i);
        __m256d v_result = _mm256_add_pd(v_dst, v_src);
        _mm256_storeu_pd(dst + i, v_result);
    }

    for (; i < count; ++i) dst[i] += src[i];
}

void simd_and_u8(uint8_t* dst, const uint8_t* src, size_t count) {
    size_t i = 0;
    const size_t VEC_SIZE = 32;

    for (; i + VEC_SIZE <= count; i += VEC_SIZE) {
        __m256i v_dst = _mm256_loadu_si256((__m256i*)(dst + i));
        __m256i v_src = _mm256_loadu_si256((__m256i*)(src + i));
        __m256i v_result = _mm256_and_si256(v_dst, v_src);
        _mm256_storeu_si256((__m256i*)(dst + i), v_result);
    }

    for (; i < count; ++i) dst[i] &= src[i];
}

void simd_or_u8(uint8_t* dst, const uint8_t* src, size_t count) {
    size_t i = 0;
    const size_t VEC_SIZE = 32;

    for (; i + VEC_SIZE <= count; i += VEC_SIZE) {
        __m256i v_dst = _mm256_loadu_si256((__m256i*)(dst + i));
        __m256i v_src = _mm256_loadu_si256((__m256i*)(src + i));
        __m256i v_result = _mm256_or_si256(v_dst, v_src);
        _mm256_storeu_si256((__m256i*)(dst + i), v_result);
    }

    for (; i < count; ++i) dst[i] |= src[i];
}

void simd_xor_u8(uint8_t* dst, const uint8_t* src, size_t count) {
    size_t i = 0;
    const size_t VEC_SIZE = 32;

    for (; i + VEC_SIZE <= count; i += VEC_SIZE) {
        __m256i v_dst = _mm256_loadu_si256((__m256i*)(dst + i));
        __m256i v_src = _mm256_loadu_si256((__m256i*)(src + i));
        __m256i v_result = _mm256_xor_si256(v_dst, v_src);
        _mm256_storeu_si256((__m256i*)(dst + i), v_result);
    }

    for (; i < count; ++i) dst[i] ^= src[i];
}

void simd_memcpy(void* dst, const void* src, size_t count) {
    uint8_t* d = (uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    size_t i = 0;
    const size_t VEC_SIZE = 32;

    for (; i + VEC_SIZE <= count; i += VEC_SIZE) {
        _mm256_storeu_si256((__m256i*)(d + i), _mm256_loadu_si256((__m256i*)(s + i)));
    }

    for (; i < count; ++i) d[i] = s[i];
}

void simd_memset(void* dst, int value, size_t count) {
    uint8_t* d = (uint8_t*)dst;
    size_t i = 0;
    const size_t VEC_SIZE = 32;
    __m256i v_val = _mm256_set1_epi8((uint8_t)value);

    for (; i + VEC_SIZE <= count; i += VEC_SIZE) {
        _mm256_storeu_si256((__m256i*)(d + i), v_val);
    }

    for (; i < count; ++i) d[i] = (uint8_t)value;
}

#else // --- C Fallbacks (non-AVX2) ---

void simd_add_u8_saturate(uint8_t* dst, const uint8_t* src, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        unsigned int sum = dst[i] + src[i];
        dst[i] = (sum > 255) ? 255 : (uint8_t)sum;
    }
}

void simd_add_u16_saturate(uint16_t* dst, const uint16_t* src, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        unsigned int sum = dst[i] + src[i];
        dst[i] = (sum > 65535) ? 65535 : (uint16_t)sum;
    }
}

void simd_add_s8_saturate(int8_t* dst, const int8_t* src, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        int sum = dst[i] + src[i];
        dst[i] = (sum > 127) ? 127 : ((sum < -128) ? -128 : (int8_t)sum);
    }
}

void simd_add_f32(float* dst, const float* src, size_t count) {
    for (size_t i = 0; i < count; ++i) dst[i] += src[i];
}

void simd_add_f64(double* dst, const double* src, size_t count) {
    for (size_t i = 0; i < count; ++i) dst[i] += src[i];
}

void simd_and_u8(uint8_t* dst, const uint8_t* src, size_t count) {
    for (size_t i = 0; i < count; ++i) dst[i] &= src[i];
}

void simd_or_u8(uint8_t* dst, const uint8_t* src, size_t count) {
    for (size_t i = 0; i < count; ++i) dst[i] |= src[i];
}

void simd_xor_u8(uint8_t* dst, const uint8_t* src, size_t count) {
    for (size_t i = 0; i < count; ++i) dst[i] ^= src[i];
}

void simd_memcpy(void* dst, const void* src, size_t count) {
    memcpy(dst, src, count);
}

void simd_memset(void* dst, int value, size_t count) {
    memset(dst, value, count);
}

#endif // __AVX2__
