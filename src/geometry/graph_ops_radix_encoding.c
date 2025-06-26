#include "geometry/graph_ops_radix_encoding.h"
#include "geometry/encoding_primitives.h"
#include <stddef.h>

static void* radix_and(void* a, void* b, GraphOpRegion region) {
    (void)region;
    RadixEncodedValue* A = (RadixEncodedValue*)a;
    RadixEncodedValue* B = (RadixEncodedValue*)b;
    size_t len = A->length < B->length ? A->length : B->length;
    for (size_t i = 0; i < len; ++i) {
        A->digits[i] &= B->digits[i];
    }
    return a;
}

static void* radix_or(void* a, void* b, GraphOpRegion region) {
    (void)region;
    RadixEncodedValue* A = (RadixEncodedValue*)a;
    RadixEncodedValue* B = (RadixEncodedValue*)b;
    size_t len = A->length < B->length ? A->length : B->length;
    for (size_t i = 0; i < len; ++i) {
        A->digits[i] |= B->digits[i];
    }
    return a;
}

static void* radix_xor(void* a, void* b, GraphOpRegion region) {
    (void)region;
    RadixEncodedValue* A = (RadixEncodedValue*)a;
    RadixEncodedValue* B = (RadixEncodedValue*)b;
    size_t len = A->length < B->length ? A->length : B->length;
    for (size_t i = 0; i < len; ++i) {
        A->digits[i] ^= B->digits[i];
    }
    return a;
}

static void* radix_not(void* a, GraphOpRegion region) {
    (void)region;
    RadixEncodedValue* A = (RadixEncodedValue*)a;
    for (size_t i = 0; i < A->length; ++i) {
        A->digits[i] = (uint8_t)(~A->digits[i]);
    }
    return a;
}

static void* radix_shift_left(void* a, int shift, GraphOpRegion region) {
    (void)region;
    if (shift <= 0 || shift >= 8) return a;
    RadixEncodedValue* A = (RadixEncodedValue*)a;
    for (size_t i = 0; i < A->length; ++i) {
        uint8_t next = (i + 1 < A->length) ? A->digits[i + 1] : 0;
        A->digits[i] = (uint8_t)((A->digits[i] << shift) | (next >> (8 - shift)));
    }
    return a;
}

static void* radix_shift_right(void* a, int shift, GraphOpRegion region) {
    (void)region;
    if (shift <= 0 || shift >= 8) return a;
    RadixEncodedValue* A = (RadixEncodedValue*)a;
    for (size_t i = A->length; i-- > 0;) {
        uint8_t prev = (i > 0) ? A->digits[i - 1] : 0;
        A->digits[i] = (uint8_t)((A->digits[i] >> shift) | (prev << (8 - shift)));
    }
    return a;
}

const OperationSuite graph_ops_radix_encoding = {
    .bit_ops = {
        .bit_and   = radix_and,
        .bit_or    = radix_or,
        .bit_xor   = radix_xor,
        .bit_not   = radix_not,
        .shift_left  = radix_shift_left,
        .shift_right = radix_shift_right
    }
};
