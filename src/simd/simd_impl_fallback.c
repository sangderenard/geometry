#include "simd_impl_common.h"
#include "simd_dispatch.h"

void simd_add_u8_fallback(simd_u8* dst, const simd_u8* src, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        unsigned int sum = dst[i] + src[i];
        dst[i] = (sum > 255) ? 255 : (simd_u8)sum;
    }
}

void simd_add_u16_fallback(simd_u16* dst, const simd_u16* src, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        unsigned int sum = dst[i] + src[i];
        dst[i] = (sum > 65535) ? 65535 : (simd_u16)sum;
    }
}

void simd_add_s8_fallback(simd_s8* dst, const simd_s8* src, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        int sum = dst[i] + src[i];
        dst[i] = (sum > 127) ? 127 : ((sum < -128) ? -128 : (simd_s8)sum);
    }
}

void simd_add_f32_fallback(simd_f32* dst, const simd_f32* src, size_t count) {
    for (size_t i = 0; i < count; ++i) dst[i] += src[i];
}

void simd_add_f64_fallback(simd_f64* dst, const simd_f64* src, size_t count) {
    for (size_t i = 0; i < count; ++i) dst[i] += src[i];
}

void simd_and_u8_fallback(simd_u8* dst, const simd_u8* src, size_t count) {
    for (size_t i = 0; i < count; ++i) dst[i] &= src[i];
}

void simd_or_u8_fallback(simd_u8* dst, const simd_u8* src, size_t count) {
    for (size_t i = 0; i < count; ++i) dst[i] |= src[i];
}

void simd_xor_u8_fallback(simd_u8* dst, const simd_u8* src, size_t count) {
    for (size_t i = 0; i < count; ++i) dst[i] ^= src[i];
}

void simd_memcpy_fallback(void* dst, const void* src, size_t count) {
    memcpy(dst, src, count);
}

void simd_memset_fallback(void* dst, int value, size_t count) {
    memset(dst, value, count);
}
