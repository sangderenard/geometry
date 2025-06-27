#ifndef ASSEMBLY_BACKEND_MEMORY_OPS_H
#define ASSEMBLY_BACKEND_MEMORY_OPS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct __attribute__((packed)) DiffBlockHeader {
    uint32_t magic;
    uint8_t  version;
    uint16_t type_id;
    uint8_t  flags;
    uint32_t payload_bytes;
    uint32_t pointer_index_offset;
    uint32_t metadata_offset;
    uint16_t stride;
    uint64_t block_id;
} DiffBlockHeader;

void* mg_encode_block(const void* raw_data, size_t size_bytes, uint16_t type_id, uint8_t flags);
void* mg_decode_block(const void* encoded_block, size_t* out_payload_size);
void  mg_tensor_compare_64x64(const void* block_a, const void* block_b, float* out_diff_tensor);
const DiffBlockHeader* mg_peek_header(const void* block);

#ifdef __cplusplus
}
#endif

#endif /* ASSEMBLY_BACKEND_MEMORY_OPS_H */
