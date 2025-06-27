#include "assembly_backend/memory_ops.h"
#include <stddef.h>
#include <stdint.h>

// Temporary definition for debugging
typedef struct DiffBlockHeader {
    uint32_t magic;
    uint8_t  version;
    uint16_t type_id;
    uint8_t  flags;
    uint32_t payload_bytes;
    uint32_t pointer_index_offset;
    uint32_t metadata_offset;
    uint16_t stride;
    uint64_t block_id;
} DiffBlockHeader; // Removed `__attribute__((packed))`

// Helper function to determine system endianness
static int is_little_endian() {
    uint16_t test = 0x1;
    return *((uint8_t*)&test) == 0x1;
}

void* mg_encode_block(const void* raw_data, size_t size_bytes, uint16_t type_id, uint8_t flags) {
    // Directly call the assembly routine
    extern void* mg_encode_block(const void* raw_data, size_t size_bytes, uint16_t type_id, uint8_t flags);
    return mg_encode_block(raw_data, size_bytes, type_id, flags);
}

void* mg_decode_block(const void* encoded_block, size_t* out_payload_size) {
    // Directly call the assembly routine
    extern void* mg_decode_block(const void* encoded_block, size_t* out_payload_size);
    return mg_decode_block(encoded_block, out_payload_size);
}

void mg_tensor_compare_64x64(const void* block_a, const void* block_b, float* out_diff_tensor) {
    // Placeholder for tensor comparison logic
    if (out_diff_tensor) {
        for (size_t i = 0; i < 64 * 64; ++i) {
            out_diff_tensor[i] = 0.0f;
        }
    }
}

const DiffBlockHeader* mg_peek_header(const void* block) {
    return (const DiffBlockHeader*)block;
}
