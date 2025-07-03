#ifndef MEMORY_INTRINSICS_H
#define MEMORY_INTRINSICS_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h> // for posix_memalign
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void* simd_aligned_alloc(size_t size);
void  simd_aligned_free(void* ptr);

size_t simd_byte_diff_count(const uint8_t* a, const uint8_t* b, size_t len);

// Fill dst with repeated copies of a byte value
// For single-byte patterns use simd_memset_impl directly
// For general pattern fills, use simd_fill_pattern

// Fill dst with repeated copies of an arbitrary pattern
// pattern: pointer to the template data of size pattern_size
// dst: destination buffer of total_bytes size
void simd_fill_pattern(const void* pattern, size_t pattern_size, void* dst, size_t total_bytes);

// Fill a contiguous array of 32-bit elements with a constant via SIMD lanes
void simd_fill_u32(uint32_t value, uint32_t* dst, size_t count);

#ifdef __cplusplus
}
#endif

#endif /* MEMORY_INTRINSICS_H */
