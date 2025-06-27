#ifndef ASSEMBLY_BACKEND_CORE_H
#define ASSEMBLY_BACKEND_CORE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SIMD_TYPES_H
#define SIMD_TYPES_H

// ----------------------
// ARM NEON
// ----------------------
#if defined(__ARM_NEON)
  #include <arm_neon.h>

  #define SIMD_ARCH_NEON 1

  // Unsigned integer types
  typedef uint8x8_t    simd_u8x8;
  typedef uint8x16_t   simd_u8x16;
  typedef uint16x4_t   simd_u16x4;
  typedef uint16x8_t   simd_u16x8;
  typedef uint32x2_t   simd_u32x2;
  typedef uint32x4_t   simd_u32x4;
  typedef uint64x1_t   simd_u64x1;
  typedef uint64x2_t   simd_u64x2;

  // Signed integer types
  typedef int8x8_t     simd_i8x8;
  typedef int8x16_t    simd_i8x16;
  typedef int16x4_t    simd_i16x4;
  typedef int16x8_t    simd_i16x8;
  typedef int32x2_t    simd_i32x2;
  typedef int32x4_t    simd_i32x4;
  typedef int64x1_t    simd_i64x1;
  typedef int64x2_t    simd_i64x2;

  // Floating point types
  typedef float32x2_t  simd_f32x2;
  typedef float32x4_t  simd_f32x4;
  typedef float64x1_t  simd_f64x1;
  typedef float64x2_t  simd_f64x2;

// ----------------------
// x86 AVX2
// ----------------------
#elif defined(__AVX2__)
  #include <immintrin.h>

  #define SIMD_ARCH_AVX2 1

  // Unsigned integer types
  typedef __m128i simd_u8x16;
  typedef __m256i simd_u8x32;
  typedef __m128i simd_u16x8;
  typedef __m256i simd_u16x16;
  typedef __m128i simd_u32x4;
  typedef __m256i simd_u32x8;
  typedef __m128i simd_u64x2;
  typedef __m256i simd_u64x4;

  // Signed integer types
  typedef __m128i simd_i8x16;
  typedef __m256i simd_i8x32;
  typedef __m128i simd_i16x8;
  typedef __m256i simd_i16x16;
  typedef __m128i simd_i32x4;
  typedef __m256i simd_i32x8;
  typedef __m128i simd_i64x2;
  typedef __m256i simd_i64x4;

  // Floating point types
  typedef __m128  simd_f32x4;
  typedef __m256  simd_f32x8;
  typedef __m128d simd_f64x2;
  typedef __m256d simd_f64x4;

// ----------------------
// x86 SSE2
// ----------------------
#elif defined(__SSE2__)
  #include <emmintrin.h>

  #define SIMD_ARCH_SSE2 1

  // Unsigned integer types
  typedef __m128i simd_u8x16;
  typedef __m128i simd_u16x8;
  typedef __m128i simd_u32x4;
  typedef __m128i simd_u64x2;

  // Signed integer types
  typedef __m128i simd_i8x16;
  typedef __m128i simd_i16x8;
  typedef __m128i simd_i32x4;
  typedef __m128i simd_i64x2;

  // Floating point types
  typedef __m128  simd_f32x4;
  typedef __m128d simd_f64x2;

// ----------------------
// Unsupported
// ----------------------
#else
  #error "No supported SIMD ISA detected"
#endif

#endif // SIMD_TYPES_H


void assembly_backend_core_init(void);

void* mg_alloc(size_t size);
void mg_free(void* ptr);

void* mg_link_factory(long long id, size_t size, int count, void* template, void* field_functions);

#ifdef __cplusplus
}
#endif

#endif /* ASSEMBLY_BACKEND_CORE_H */
