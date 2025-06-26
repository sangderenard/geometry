#include "geometry/graph_ops_bitfield.h"
#include "geometry/encoding_primitives.h"
#include <stddef.h>

static void* bitfield_and(void* a, void* b, GraphOpRegion region) {
    (void)region;
    BitStrideStream* A = (BitStrideStream*)a;
    BitStrideStream* B = (BitStrideStream*)b;
    size_t len = A->length_bits / 8;
    if (B->length_bits / 8 < len) len = B->length_bits / 8;
    for (size_t i = 0; i < len; ++i) {
        A->data[i] &= B->data[i];
    }
    return a;
}

static void* bitfield_or(void* a, void* b, GraphOpRegion region) {
    (void)region;
    BitStrideStream* A = (BitStrideStream*)a;
    BitStrideStream* B = (BitStrideStream*)b;
    size_t len = A->length_bits / 8;
    if (B->length_bits / 8 < len) len = B->length_bits / 8;
    for (size_t i = 0; i < len; ++i) {
        A->data[i] |= B->data[i];
    }
    return a;
}

static void* bitfield_xor(void* a, void* b, GraphOpRegion region) {
    (void)region;
    BitStrideStream* A = (BitStrideStream*)a;
    BitStrideStream* B = (BitStrideStream*)b;
    size_t len = A->length_bits / 8;
    if (B->length_bits / 8 < len) len = B->length_bits / 8;
    for (size_t i = 0; i < len; ++i) {
        A->data[i] ^= B->data[i];
    }
    return a;
}

static void* bitfield_not(void* a, GraphOpRegion region) {
    (void)region;
    BitStrideStream* A = (BitStrideStream*)a;
    size_t len = A->length_bits / 8;
    for (size_t i = 0; i < len; ++i) {
        A->data[i] = (uint8_t)(~A->data[i]);
    }
    return a;
}

static void* bitfield_shift_left(void* a, int shift, GraphOpRegion region) {
    (void)region;
    if (shift <= 0 || shift >= 8) return a;
    BitStrideStream* A = (BitStrideStream*)a;
    size_t len = A->length_bits / 8;
    for (size_t i = 0; i < len; ++i) {
        uint8_t next = (i + 1 < len) ? A->data[i + 1] : 0;
        A->data[i] = (uint8_t)((A->data[i] << shift) | (next >> (8 - shift)));
    }
    return a;
}

static void* bitfield_shift_right(void* a, int shift, GraphOpRegion region) {
    (void)region;
    if (shift <= 0 || shift >= 8) return a;
    BitStrideStream* A = (BitStrideStream*)a;
    size_t len = A->length_bits / 8;
    for (size_t i = len; i-- > 0;) {
        uint8_t prev = (i > 0) ? A->data[i - 1] : 0;
        A->data[i] = (uint8_t)((A->data[i] >> shift) | (prev << (8 - shift)));
    }
    return a;
}

const OperationSuite graph_ops_bitfield = {
    .bit_ops = {
        .bit_and   = bitfield_and,
        .bit_or    = bitfield_or,
        .bit_xor   = bitfield_xor,
        .bit_not   = bitfield_not,
        .shift_left  = bitfield_shift_left,
        .shift_right = bitfield_shift_right
    }
};
