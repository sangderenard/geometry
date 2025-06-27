#include "assembly_backend/memory_ops.h"

/*
 * Stub implementations for the differentiable memory graph backend.
 * These are placeholders and do not perform real memory management yet.
 */

void* mg_encode_block(const void* raw_data, size_t size_bytes, uint16_t type_id, uint8_t flags) {
    (void)raw_data;
    (void)size_bytes;
    (void)type_id;
    (void)flags;
    // Memory allocation is disallowed in this repository. Returning NULL for now.
    return NULL;
}

void* mg_decode_block(const void* encoded_block, size_t* out_payload_size) {
    if (out_payload_size) *out_payload_size = 0;
    // No actual decoding performed.
    return (void*)encoded_block;
}

void mg_tensor_compare_64x64(const void* block_a, const void* block_b, float* out_diff_tensor) {
    (void)block_a;
    (void)block_b;
    if (out_diff_tensor) {
        for (size_t i = 0; i < 64 * 64; ++i) {
            out_diff_tensor[i] = 0.0f;
        }
    }
}

const DiffBlockHeader* mg_peek_header(const void* block) {
    return (const DiffBlockHeader*)block;
}
