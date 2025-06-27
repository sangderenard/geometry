#ifndef SIMD_DISPATCH_H
#define SIMD_DISPATCH_H

#include <stddef.h>
#include <stdint.h>

typedef void (*simd_add_u8_fn)(uint8_t* dst, const uint8_t* src, size_t count);
extern simd_add_u8_fn simd_add_u8;

void simd_init_dispatch();

#endif // SIMD_DISPATCH_H