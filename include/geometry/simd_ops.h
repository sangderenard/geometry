#ifndef GEOMETRY_SIMD_OPS_H
#define GEOMETRY_SIMD_OPS_H

#include <stddef.h>
#include "assembly_backend/simd/simd_types.h"

#ifdef __cplusplus
extern "C" {
#endif

void simd_add_u8_saturate(simd_u8* dst, const simd_u8* src, size_t count);
void simd_add_u16_saturate(simd_u16* dst, const simd_u16* src, size_t count);
void simd_add_s8_saturate(simd_s8* dst, const simd_s8* src, size_t count);
void simd_add_f32(simd_f32* dst, const simd_f32* src, size_t count);
void simd_add_f64(simd_f64* dst, const simd_f64* src, size_t count);
void simd_and_u8(simd_u8* dst, const simd_u8* src, size_t count);
void simd_or_u8(simd_u8* dst, const simd_u8* src, size_t count);
void simd_xor_u8(simd_u8* dst, const simd_u8* src, size_t count);
void simd_memcpy(void* dst, const void* src, size_t num_bytes);
void simd_memset(void* dst, int value, size_t num_bytes);

#ifdef __cplusplus
}
#endif

#endif // GEOMETRY_SIMD_OPS_H
