#ifndef GEOMETRY_ENCODING_PRIMITIVES_H
#define GEOMETRY_ENCODING_PRIMITIVES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Strided bit-level stream */
typedef struct {
    uint8_t* data;
    size_t stride_bits;
    size_t length_bits;
} BitStrideStream;

/* Simple byte-addressable field */
typedef struct {
    uint8_t* bytes;
    size_t length_bytes;
} ByteField;

/* Arbitrary radix encoded value */
typedef struct {
    uint8_t* digits;
    size_t length;
    uint8_t base;
    uint8_t is_balanced;
} RadixEncodedValue;

/* Palette-based pattern stream */
typedef struct {
    void** pattern_slots;
    uint32_t* index_stream;
    size_t pattern_count;
} PatternPaletteStream;

#ifdef __cplusplus
}
#endif

#endif /* GEOMETRY_ENCODING_PRIMITIVES_H */
