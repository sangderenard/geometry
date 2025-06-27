#if defined(__AVX2__)
#include <immintrin.h>
#include "assembly_backend/simd/simd_impl_common.h"
#include "assembly_backend/simd/simd_dispatch.h"

void simd_add_u8_avx2(simd_u8* dst, const simd_u8* src, size_t count) {
    size_t i = 0;
    const size_t VEC = 32;
    for (; i + VEC <= count; i += VEC) {
        __m256i v_dst = _mm256_loadu_si256((const __m256i*)(dst + i));
        __m256i v_src = _mm256_loadu_si256((const __m256i*)(src + i));
        __m256i v_res = _mm256_adds_epu8(v_dst, v_src);
        _mm256_storeu_si256((__m256i*)(dst + i), v_res);
    }
    for (; i < count; ++i) {
        unsigned int sum = dst[i] + src[i];
        dst[i] = (sum > 255) ? 255 : (simd_u8)sum;
    }
}

void simd_add_u16_avx2(simd_u16* dst, const simd_u16* src, size_t count) {
    size_t i = 0;
    const size_t VEC = 16;
    for (; i + VEC <= count; i += VEC) {
        __m256i v_dst = _mm256_loadu_si256((const __m256i*)(dst + i));
        __m256i v_src = _mm256_loadu_si256((const __m256i*)(src + i));
        __m256i v_res = _mm256_adds_epu16(v_dst, v_src);
        _mm256_storeu_si256((__m256i*)(dst + i), v_res);
    }
    for (; i < count; ++i) {
        unsigned int sum = dst[i] + src[i];
        dst[i] = (sum > 65535) ? 65535 : (simd_u16)sum;
    }
}

void simd_add_s8_avx2(simd_s8* dst, const simd_s8* src, size_t count) {
    size_t i = 0;
    const size_t VEC = 32;
    for (; i + VEC <= count; i += VEC) {
        __m256i v_dst = _mm256_loadu_si256((const __m256i*)(dst + i));
        __m256i v_src = _mm256_loadu_si256((const __m256i*)(src + i));
        __m256i v_res = _mm256_adds_epi8(v_dst, v_src);
        _mm256_storeu_si256((__m256i*)(dst + i), v_res);
    }
    for (; i < count; ++i) {
        int sum = dst[i] + src[i];
        dst[i] = (sum > 127) ? 127 : ((sum < -128) ? -128 : (simd_s8)sum);
    }
}

void simd_add_f32_avx2(simd_f32* dst, const simd_f32* src, size_t count) {
    size_t i = 0;
    const size_t VEC = 8;
    for (; i + VEC <= count; i += VEC) {
        __m256 v_dst = _mm256_loadu_ps(dst + i);
        __m256 v_src = _mm256_loadu_ps(src + i);
        __m256 v_res = _mm256_add_ps(v_dst, v_src);
        _mm256_storeu_ps(dst + i, v_res);
    }
    for (; i < count; ++i) dst[i] += src[i];
}

void simd_add_f64_avx2(simd_f64* dst, const simd_f64* src, size_t count) {
    size_t i = 0;
    const size_t VEC = 4;
    for (; i + VEC <= count; i += VEC) {
        __m256d v_dst = _mm256_loadu_pd(dst + i);
        __m256d v_src = _mm256_loadu_pd(src + i);
        __m256d v_res = _mm256_add_pd(v_dst, v_src);
        _mm256_storeu_pd(dst + i, v_res);
    }
    for (; i < count; ++i) dst[i] += src[i];
}

void simd_and_u8_avx2(simd_u8* dst, const simd_u8* src, size_t count) {
    size_t i = 0;
    const size_t VEC = 32;
    for (; i + VEC <= count; i += VEC) {
        __m256i v_dst = _mm256_loadu_si256((const __m256i*)(dst + i));
        __m256i v_src = _mm256_loadu_si256((const __m256i*)(src + i));
        __m256i v_res = _mm256_and_si256(v_dst, v_src);
        _mm256_storeu_si256((__m256i*)(dst + i), v_res);
    }
    for (; i < count; ++i) dst[i] &= src[i];
}

void simd_or_u8_avx2(simd_u8* dst, const simd_u8* src, size_t count) {
    size_t i = 0;
    const size_t VEC = 32;
    for (; i + VEC <= count; i += VEC) {
        __m256i v_dst = _mm256_loadu_si256((const __m256i*)(dst + i));
        __m256i v_src = _mm256_loadu_si256((const __m256i*)(src + i));
        __m256i v_res = _mm256_or_si256(v_dst, v_src);
        _mm256_storeu_si256((__m256i*)(dst + i), v_res);
    }
    for (; i < count; ++i) dst[i] |= src[i];
}

void simd_xor_u8_avx2(simd_u8* dst, const simd_u8* src, size_t count) {
    size_t i = 0;
    const size_t VEC = 32;
    for (; i + VEC <= count; i += VEC) {
        __m256i v_dst = _mm256_loadu_si256((const __m256i*)(dst + i));
        __m256i v_src = _mm256_loadu_si256((const __m256i*)(src + i));
        __m256i v_res = _mm256_xor_si256(v_dst, v_src);
        _mm256_storeu_si256((__m256i*)(dst + i), v_res);
    }
    for (; i < count; ++i) dst[i] ^= src[i];
}

void simd_memcpy_avx2(void* dst, const void* src, size_t count) {
    simd_u8* d = (simd_u8*)dst;
    const simd_u8* s = (const simd_u8*)src;
    size_t i = 0;
    const size_t VEC = 32;
    for (; i + VEC <= count; i += VEC) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(s + i));
        _mm256_storeu_si256((__m256i*)(d + i), v);
    }
    for (; i < count; ++i) d[i] = s[i];
}

void simd_memset_avx2(void* dst, int value, size_t count) {
    simd_u8* d = (simd_u8*)dst;
    size_t i = 0;
    const size_t VEC = 32;
    __m256i v = _mm256_set1_epi8((simd_u8)value);
    for (; i + VEC <= count; i += VEC) {
        _mm256_storeu_si256((__m256i*)(d + i), v);
    }
    for (; i < count; ++i) d[i] = (simd_u8)value;
}

#endif // __AVX2__
