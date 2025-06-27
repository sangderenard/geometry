#if defined(__AVX2__)
#include <immintrin.h>
#include "simd_dispatch.h"

void simd_add_u8_avx2(uint8_t* dst, const uint8_t* src, size_t count) {
    size_t i = 0;
    const size_t VEC = 32;
    for (; i + VEC <= count; i += VEC) {
        __m256i v_dst = _mm256_loadu_si256((__m256i*)(dst + i));
        __m256i v_src = _mm256_loadu_si256((__m256i*)(src + i));
        __m256i v_res = _mm256_adds_epu8(v_dst, v_src);
        _mm256_storeu_si256((__m256i*)(dst + i), v_res);
    }
    for (; i < count; ++i) {
        unsigned int sum = dst[i] + src[i];
        dst[i] = (sum > 255) ? 255 : (uint8_t)sum;
    }
}
#endif