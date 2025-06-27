#ifndef GEOMETRY_SIMD_OPS_H
#define GEOMETRY_SIMD_OPS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Adds two arrays of unsigned 8-bit integers using saturating arithmetic.
 *
 * This function computes `dst[i] = saturate(dst[i] + src[i])`. It will use
 * AVX2 SIMD instructions if available for high performance, otherwise it
 * falls back to a standard C implementation.
 */
void simd_add_u8_saturate(uint8_t* dst, const uint8_t* src, size_t count);

/**
 * @brief Adds two arrays of unsigned 16-bit integers using saturating arithmetic.
 */
void simd_add_u16_saturate(uint16_t* dst, const uint16_t* src, size_t count);

/**
 * @brief Adds two arrays of signed 8-bit integers using saturating arithmetic.
 */
void simd_add_s8_saturate(int8_t* dst, const int8_t* src, size_t count);

/**
 * @brief Adds two arrays of single-precision floating-point numbers.
 */
void simd_add_f32(float* dst, const float* src, size_t count);

/**
 * @brief Adds two arrays of double-precision floating-point numbers.
 */
void simd_add_f64(double* dst, const double* src, size_t count);

/**
 * @brief Performs a bitwise AND operation on two arrays of unsigned 8-bit integers.
 */
void simd_and_u8(uint8_t* dst, const uint8_t* src, size_t count);

/**
 * @brief Performs a bitwise OR operation on two arrays of unsigned 8-bit integers.
 */
void simd_or_u8(uint8_t* dst, const uint8_t* src, size_t count);

/**
 * @brief Performs a bitwise XOR operation on two arrays of unsigned 8-bit integers.
 */
void simd_xor_u8(uint8_t* dst, const uint8_t* src, size_t count);

/**
 * @brief Copies bytes from source to destination. Optimized with SIMD.
 */
void simd_memcpy(void* dst, const void* src, size_t num_bytes);

/**
 * @brief Fills a block of memory with a specified byte value. Optimized with SIMD.
 */
void simd_memset(void* dst, int value, size_t num_bytes);

// You can extend this pattern for other operations (sub, mul, div, min, max, etc.)
// and other integer/float types as needed.

#ifdef __cplusplus
}
#endif

#endif // GEOMETRY_SIMD_OPS_H