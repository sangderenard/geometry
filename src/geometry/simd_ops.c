#include "geometry/simd_ops.h"
#include "simd/simd_dispatch.h"

static int initialized = 0;
static void ensure_init(void) {
    if (!initialized) {
        simd_init_dispatch();
        initialized = 1;
    }
}

void simd_add_u8_saturate(simd_u8* dst, const simd_u8* src, size_t count) {
    ensure_init();
    simd_add_u8_impl(dst, src, count);
}

void simd_add_u16_saturate(simd_u16* dst, const simd_u16* src, size_t count) {
    ensure_init();
    simd_add_u16_impl(dst, src, count);
}

void simd_add_s8_saturate(simd_s8* dst, const simd_s8* src, size_t count) {
    ensure_init();
    simd_add_s8_impl(dst, src, count);
}

void simd_add_f32(simd_f32* dst, const simd_f32* src, size_t count) {
    ensure_init();
    simd_add_f32_impl(dst, src, count);
}

void simd_add_f64(simd_f64* dst, const simd_f64* src, size_t count) {
    ensure_init();
    simd_add_f64_impl(dst, src, count);
}

void simd_and_u8(simd_u8* dst, const simd_u8* src, size_t count) {
    ensure_init();
    simd_and_u8_impl(dst, src, count);
}

void simd_or_u8(simd_u8* dst, const simd_u8* src, size_t count) {
    ensure_init();
    simd_or_u8_impl(dst, src, count);
}

void simd_xor_u8(simd_u8* dst, const simd_u8* src, size_t count) {
    ensure_init();
    simd_xor_u8_impl(dst, src, count);
}

void simd_memcpy(void* dst, const void* src, size_t count) {
    ensure_init();
    simd_memcpy_impl(dst, src, count);
}

void simd_memset(void* dst, int value, size_t count) {
    ensure_init();
    simd_memset_impl(dst, value, count);
}
