#if defined(__SSE2__)
#include <emmintrin.h>
#include "simd_dispatch.h"

void simd_add_u8_sse2(uint8_t* dst, const uint8_t* src, size_t count) {
    size_t i = 0;
    const size_t VEC = 16;
    for (; i + VEC <= count; i += VEC) {
        __m128i v_dst = _mm_loadu_si128((__m128i*)(dst + i));
        __m128i v_src = _mm_loadu_si128((__m128i*)(src + i));
        __m128i v_res = _mm_adds_epu8(v_dst, v_src);
        _mm_storeu_si128((__m128i*)(dst + i), v_res);
    }
    for (; i < count; ++i) {
        unsigned int sum = dst[i] + src[i];
        dst[i] = (sum > 255) ? 255 : (uint8_t)sum;
    }
}
#endif