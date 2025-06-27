#include "assembly_backend/memory_ops.h"
#include "assembly_backend/core.h"
#include "assembly_backend/simd/simd_dispatch.h"
#include "assembly_backend/simd/memory_intrinsics.h"
#include "geometry/guardian_platform.h"  // for guardian_now()
#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <time.h>  // for nanosecond timestamps

// extern assembly routines renamed to avoid recursion
extern void* mg_encode_block_asm(const void* raw_data, size_t size_bytes, uint16_t type_id, uint8_t flags);
extern void* mg_decode_block_asm(const void* encoded_block, size_t* out_payload_size);

void* mg_encode_block(const void* raw_data, size_t size_bytes, uint16_t type_id, uint8_t flags) {
    return mg_encode_block_asm(raw_data, size_bytes, type_id, flags);
}

void* mg_decode_block(const void* encoded_block, size_t* out_payload_size) {
    return mg_decode_block_asm(encoded_block, out_payload_size);
}


void* mg_tensor_compare(const void* block_a, const void* block_b) {
    if (!block_a || !block_b) return NULL;

    const DiffBlockHeader* hdr_a = (const DiffBlockHeader*)block_a;
    const DiffBlockHeader* hdr_b = (const DiffBlockHeader*)block_b;

    size_t size_a = hdr_a->payload_bytes;
    size_t size_b = hdr_b->payload_bytes;
    size_t bytes = (size_a < size_b) ? size_a : size_b;

    const uint8_t* a = (const uint8_t*)(hdr_a + 1);
    const uint8_t* b = (const uint8_t*)(hdr_b + 1);

    size_t diff_count = simd_byte_diff_count(a, b, bytes);

    float* result = (float*)mg_alloc(sizeof(float));
    if (!result) return NULL;
    result[0] = (float)diff_count;
    return result;
}


// Allocate a dynamic span: size = floor(init_n) * 2^floor(init_m) * 64 bytes
void* memops_span_alloc(float init_n, float init_m, float decay_factor) {
    // Compute integer dims
    uint32_t n = (uint32_t)floorf(init_n);
    uint32_t m = (uint32_t)floorf(init_m);
    uint64_t total = (uint64_t)n << m;           // n * 2^m
    

    // Allocate header + data with alignment and zeroing
    size_t header_size = sizeof(MemSpanHeader);
    uint8_t* base = simd_memset_impl(mg_alloc(header_size + total), 0,
                                     header_size + total);
    if (!base) return NULL;

    // Fill header with defaults
    MemSpanHeader* h = (MemSpanHeader*)base;
    h->n = init_n;
    h->m = init_m;
    h->growth_n = init_n;
    h->growth_m = init_m;
    h->decay_n = decay_factor;
    h->decay_m = decay_factor;
    for (int i = 0; i < MEMORY_BLOCK_HISTORY; ++i) {
        h->timestamp_ns[i] = 0;
        h->payload_bytes[i] = 0;
    }
    guardian_time_t now = guardian_now();
    h->timestamp_ns[0] = now.nanoseconds;
    h->payload_bytes[0] = total;
    h->diff_tape = NULL;
    h->diff_tape_flags = DIFF_TAPE_FORMAT_FLAG_NONE;
    h->encoding_scheme = BIT_ENCODING_SCHEME_BINARY;
    h->history_type = MEM_BLOCK_HISTORY_TYPE_ROLLING;
    h->flags = MEM_BLOCK_FLAG_DYNAMIC;

    // Return pointer to data just past header
    return base + header_size;
}

// Free a span
void memops_span_free(void* span_ptr) {
    if (!span_ptr) return;
    // Compute base pointer
    uint8_t* base = (uint8_t*)span_ptr - sizeof(MemSpanHeader);
    mg_free(base);
}
