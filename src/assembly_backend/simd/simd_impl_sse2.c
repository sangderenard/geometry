#if defined(__SSE2__)
#include <emmintrin.h>
#include "assembly_backend/simd/simd_impl_common.h"
#include "assembly_backend/simd/simd_dispatch.h"

void simd_add_u8_sse2(simd_u8* dst, const simd_u8* src, size_t count) {
    size_t i = 0;
    const size_t VEC = 16;
    for (; i + VEC <= count; i += VEC) {
        __m128i v_dst = _mm_loadu_si128((const __m128i*)(dst + i));
        __m128i v_src = _mm_loadu_si128((const __m128i*)(src + i));
        __m128i v_res = _mm_adds_epu8(v_dst, v_src);
        _mm_storeu_si128((__m128i*)(dst + i), v_res);
    }
    for (; i < count; ++i) {
        unsigned int sum = dst[i] + src[i];
        dst[i] = (sum > 255) ? 255 : (simd_u8)sum;
    }
}

void simd_add_u16_sse2(simd_u16* dst, const simd_u16* src, size_t count) {
    size_t i = 0;
    const size_t VEC = 8;
    for (; i + VEC <= count; i += VEC) {
        __m128i v_dst = _mm_loadu_si128((const __m128i*)(dst + i));
        __m128i v_src = _mm_loadu_si128((const __m128i*)(src + i));
        __m128i v_res = _mm_adds_epu16(v_dst, v_src);
        _mm_storeu_si128((__m128i*)(dst + i), v_res);
    }
    for (; i < count; ++i) {
        unsigned int sum = dst[i] + src[i];
        dst[i] = (sum > 65535) ? 65535 : (simd_u16)sum;
    }
}

void simd_add_s8_sse2(simd_s8* dst, const simd_s8* src, size_t count) {
    size_t i = 0;
    const size_t VEC = 16;
    for (; i + VEC <= count; i += VEC) {
        __m128i v_dst = _mm_loadu_si128((const __m128i*)(dst + i));
        __m128i v_src = _mm_loadu_si128((const __m128i*)(src + i));
        __m128i v_res = _mm_adds_epi8(v_dst, v_src);
        _mm_storeu_si128((__m128i*)(dst + i), v_res);
    }
    for (; i < count; ++i) {
        int sum = dst[i] + src[i];
        dst[i] = (sum > 127) ? 127 : ((sum < -128) ? -128 : (simd_s8)sum);
    }
}

void simd_add_f32_sse2(simd_f32* dst, const simd_f32* src, size_t count) {
    size_t i = 0;
    const size_t VEC = 4;
    for (; i + VEC <= count; i += VEC) {
        __m128 v_dst = _mm_loadu_ps(dst + i);
        __m128 v_src = _mm_loadu_ps(src + i);
        __m128 v_res = _mm_add_ps(v_dst, v_src);
        _mm_storeu_ps(dst + i, v_res);
    }
    for (; i < count; ++i) dst[i] += src[i];
}

void simd_add_f64_sse2(simd_f64* dst, const simd_f64* src, size_t count) {
    size_t i = 0;
    const size_t VEC = 2;
    for (; i + VEC <= count; i += VEC) {
        __m128d v_dst = _mm_loadu_pd(dst + i);
        __m128d v_src = _mm_loadu_pd(src + i);
        __m128d v_res = _mm_add_pd(v_dst, v_src);
        _mm_storeu_pd(dst + i, v_res);
    }
    for (; i < count; ++i) dst[i] += src[i];
}

void simd_and_u8_sse2(simd_u8* dst, const simd_u8* src, size_t count) {
    size_t i = 0;
    const size_t VEC = 16;
    for (; i + VEC <= count; i += VEC) {
        __m128i v_dst = _mm_loadu_si128((const __m128i*)(dst + i));
        __m128i v_src = _mm_loadu_si128((const __m128i*)(src + i));
        __m128i v_res = _mm_and_si128(v_dst, v_src);
        _mm_storeu_si128((__m128i*)(dst + i), v_res);
    }
    for (; i < count; ++i) dst[i] &= src[i];
}

void simd_or_u8_sse2(simd_u8* dst, const simd_u8* src, size_t count) {
    size_t i = 0;
    const size_t VEC = 16;
    for (; i + VEC <= count; i += VEC) {
        __m128i v_dst = _mm_loadu_si128((const __m128i*)(dst + i));
        __m128i v_src = _mm_loadu_si128((const __m128i*)(src + i));
        __m128i v_res = _mm_or_si128(v_dst, v_src);
        _mm_storeu_si128((__m128i*)(dst + i), v_res);
    }
    for (; i < count; ++i) dst[i] |= src[i];
}

void simd_xor_u8_sse2(simd_u8* dst, const simd_u8* src, size_t count) {
    size_t i = 0;
    const size_t VEC = 16;
    for (; i + VEC <= count; i += VEC) {
        __m128i v_dst = _mm_loadu_si128((const __m128i*)(dst + i));
        __m128i v_src = _mm_loadu_si128((const __m128i*)(src + i));
        __m128i v_res = _mm_xor_si128(v_dst, v_src);
        _mm_storeu_si128((__m128i*)(dst + i), v_res);
    }
    for (; i < count; ++i) dst[i] ^= src[i];
}

void simd_memcpy_sse2(void* dst, const void* src, size_t count) {
    simd_u8* d = (simd_u8*)dst;
    const simd_u8* s = (const simd_u8*)src;
    size_t i = 0;
    const size_t VEC = 16;
    for (; i + VEC <= count; i += VEC) {
        __m128i v = _mm_loadu_si128((const __m128i*)(s + i));
        _mm_storeu_si128((__m128i*)(d + i), v);
    }
    for (; i < count; ++i) d[i] = s[i];
}

void simd_memset_sse2(void* dst, int value, size_t count) {
    simd_u8* d = (simd_u8*)dst;
    size_t i = 0;
    const size_t VEC = 16;
    __m128i v = _mm_set1_epi8((simd_u8)value);
    for (; i + VEC <= count; i += VEC) {
        _mm_storeu_si128((__m128i*)(d + i), v);
    }
    for (; i < count; ++i) d[i] = (simd_u8)value;
}

#endif // __SSE2__
