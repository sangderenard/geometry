#ifndef MEMORY_INTRINSICS_H
#define MEMORY_INTRINSICS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void* simd_aligned_alloc(size_t size);
void  simd_aligned_free(void* ptr);

size_t simd_byte_diff_count(const uint8_t* a, const uint8_t* b, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* MEMORY_INTRINSICS_H */
