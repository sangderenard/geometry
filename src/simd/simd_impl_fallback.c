#include "simd_dispatch.h"

void simd_add_u8_fallback(uint8_t* dst, const uint8_t* src, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        unsigned int sum = dst[i] + src[i];
        dst[i] = (sum > 255) ? 255 : (uint8_t)sum;
    }
}