#include "geometry/graph_ops_bytefield.h"
#include "geometry/encoding_primitives.h"
#include <stddef.h>

static void* bytefield_and(void* a, void* b, GraphOpRegion region) {
    (void)region;
    ByteField* A = (ByteField*)a;
    ByteField* B = (ByteField*)b;
    size_t len = A->length_bytes < B->length_bytes ? A->length_bytes : B->length_bytes;
    for (size_t i = 0; i < len; ++i) {
        A->bytes[i] &= B->bytes[i];
    }
    return a;
}

static void* bytefield_or(void* a, void* b, GraphOpRegion region) {
    (void)region;
    ByteField* A = (ByteField*)a;
    ByteField* B = (ByteField*)b;
    size_t len = A->length_bytes < B->length_bytes ? A->length_bytes : B->length_bytes;
    for (size_t i = 0; i < len; ++i) {
        A->bytes[i] |= B->bytes[i];
    }
    return a;
}

static void* bytefield_xor(void* a, void* b, GraphOpRegion region) {
    (void)region;
    ByteField* A = (ByteField*)a;
    ByteField* B = (ByteField*)b;
    size_t len = A->length_bytes < B->length_bytes ? A->length_bytes : B->length_bytes;
    for (size_t i = 0; i < len; ++i) {
        A->bytes[i] ^= B->bytes[i];
    }
    return a;
}

static void* bytefield_not(void* a, GraphOpRegion region) {
    (void)region;
    ByteField* A = (ByteField*)a;
    for (size_t i = 0; i < A->length_bytes; ++i) {
        A->bytes[i] = (uint8_t)(~A->bytes[i]);
    }
    return a;
}

static void* bytefield_shift_left(void* a, int shift, GraphOpRegion region) {
    (void)region;
    if (shift <= 0 || shift >= 8) return a;
    ByteField* A = (ByteField*)a;
    for (size_t i = 0; i < A->length_bytes; ++i) {
        uint8_t next = (i + 1 < A->length_bytes) ? A->bytes[i + 1] : 0;
        A->bytes[i] = (uint8_t)((A->bytes[i] << shift) | (next >> (8 - shift)));
    }
    return a;
}

static void* bytefield_shift_right(void* a, int shift, GraphOpRegion region) {
    (void)region;
    if (shift <= 0 || shift >= 8) return a;
    ByteField* A = (ByteField*)a;
    for (size_t i = A->length_bytes; i-- > 0;) {
        uint8_t prev = (i > 0) ? A->bytes[i - 1] : 0;
        A->bytes[i] = (uint8_t)((A->bytes[i] >> shift) | (prev << (8 - shift)));
    }
    return a;
}

const OperationSuite graph_ops_bytefield = {
    .bit_ops = {
        .bit_and   = bytefield_and,
        .bit_or    = bytefield_or,
        .bit_xor   = bytefield_xor,
        .bit_not   = bytefield_not,
        .shift_left  = bytefield_shift_left,
        .shift_right = bytefield_shift_right
    }
};
