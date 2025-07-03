#ifndef ASSEMBLY_BACKEND_CORE_H
#define ASSEMBLY_BACKEND_CORE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void assembly_backend_core_init(void);

void* mg_alloc(size_t size);
void mg_free(void* ptr);

void* mg_link_factory(long long id, size_t wrapper_size,size_t size, int count, void* template, void* field_functions);

/* ------------------------------------------------------------------------- */
/*  SIMD Intrinsics Library                                                  */
/* ------------------------------------------------------------------------- */
/*  Core exposes a small library of portable vector operations and memory    */
/*  helpers.  memory_ops can call these function pointers to access the      */
/*  accelerated routines selected by simd_init_dispatch().                   */
/*                                                                           */
/*    simd_add_u8_impl(dst, src, count)    - saturating add uint8 vectors     */
/*    simd_add_u16_impl(dst, src, count)   - saturating add uint16 vectors    */
/*    simd_add_s8_impl(dst, src, count)    - saturating add int8 vectors      */
/*    simd_add_f32_impl(dst, src, count)   - add float32 vectors              */
/*    simd_add_f64_impl(dst, src, count)   - add float64 vectors              */
/*    simd_and_u8_impl(dst, src, count)    - bitwise AND uint8 vectors        */
/*    simd_or_u8_impl(dst, src, count)     - bitwise OR uint8 vectors         */
/*    simd_xor_u8_impl(dst, src, count)    - bitwise XOR uint8 vectors        */
/*    simd_memcpy_impl(dst, src, bytes)    - accelerated memcpy              */
/*    simd_memset_impl(dst, val, bytes)    - accelerated memset              */
/*    simd_byte_diff_count(a, b, len)      - Hamming distance of byte arrays  */
/*    simd_aligned_alloc(size)             - 64-byte aligned allocation       */
/*    simd_aligned_free(ptr)               - release aligned memory           */
/*                                                                           */
/*  Call simd_init_dispatch() during initialization to select the best SIMD  */
/*  backend automatically.                                                   */
/* ------------------------------------------------------------------------- */

#ifdef __cplusplus
}
#endif

#endif /* ASSEMBLY_BACKEND_CORE_H */
