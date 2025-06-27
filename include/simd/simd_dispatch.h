#ifndef SIMD_DISPATCH_H
#define SIMD_DISPATCH_H

#include "simd_types.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*simd_add_u8_fn)(simd_u8* dst, const simd_u8* src, size_t count);
extern simd_add_u8_fn simd_add_u8_impl;

typedef void (*simd_add_u16_fn)(simd_u16* dst, const simd_u16* src, size_t count);
extern simd_add_u16_fn simd_add_u16_impl;

typedef void (*simd_add_s8_fn)(simd_s8* dst, const simd_s8* src, size_t count);
extern simd_add_s8_fn simd_add_s8_impl;

typedef void (*simd_add_f32_fn)(simd_f32* dst, const simd_f32* src, size_t count);
extern simd_add_f32_fn simd_add_f32_impl;

typedef void (*simd_add_f64_fn)(simd_f64* dst, const simd_f64* src, size_t count);
extern simd_add_f64_fn simd_add_f64_impl;

typedef void (*simd_and_u8_fn)(simd_u8* dst, const simd_u8* src, size_t count);
extern simd_and_u8_fn simd_and_u8_impl;

typedef void (*simd_or_u8_fn)(simd_u8* dst, const simd_u8* src, size_t count);
extern simd_or_u8_fn simd_or_u8_impl;

typedef void (*simd_xor_u8_fn)(simd_u8* dst, const simd_u8* src, size_t count);
extern simd_xor_u8_fn simd_xor_u8_impl;

typedef void (*simd_memcpy_fn)(void* dst, const void* src, size_t num_bytes);
extern simd_memcpy_fn simd_memcpy_impl;

typedef void (*simd_memset_fn)(void* dst, int value, size_t num_bytes);
extern simd_memset_fn simd_memset_impl;

void simd_init_dispatch(void);

#ifdef __cplusplus
}
#endif

#endif // SIMD_DISPATCH_H
