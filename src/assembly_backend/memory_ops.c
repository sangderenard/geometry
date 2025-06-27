#include "assembly_backend/memory_ops.h"
#include "simd/simd_dispatch.h"
#include "geometry/guardian_platform.h"  // for guardian_now()
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>  // for nanosecond timestamps

// Initialize SIMD dispatch on startup
void assembly_backend_init(void) {
    simd_init_dispatch();
}

// extern assembly routines renamed to avoid recursion
extern void* mg_encode_block_asm(const void* raw_data, size_t size_bytes, uint16_t type_id, uint8_t flags);
extern void* mg_decode_block_asm(const void* encoded_block, size_t* out_payload_size);

void* mg_encode_block(const void* raw_data, size_t size_bytes, uint16_t type_id, uint8_t flags) {
    return mg_encode_block_asm(raw_data, size_bytes, type_id, flags);
}

void* mg_decode_block(const void* encoded_block, size_t* out_payload_size) {
    return mg_decode_block_asm(encoded_block, out_payload_size);
}

void* mg_alloc(size_t size) {
    void* ptr = malloc(size);
    if (ptr) simd_memset_impl(ptr, 0, size);
    return ptr;
}

void mg_free(void* ptr) {
    free(ptr);
}

void mg_tensor_compare_64x64(const void* block_a, const void* block_b, float* out_diff_tensor) {
    if (out_diff_tensor) {
        for (size_t i = 0; i < 64 * 64; ++i) {
            out_diff_tensor[i] = 0.0f;
        }
    }
}

const DiffBlockHeader* mg_peek_header(const void* block) {
    return (const DiffBlockHeader*)block;
}

// Allocate a dynamic span: size = floor(init_n) * 2^floor(init_m) * 64 bytes
void* memops_span_alloc(float init_n, float init_m, float decay_factor) {
    // Compute integer dims
    uint32_t n = (uint32_t)floorf(init_n);
    uint32_t m = (uint32_t)floorf(init_m);
    uint64_t total = (uint64_t)n << m;           // n * 2^m
    

    // Allocate header + data
    size_t header_size = sizeof(MemSpanHeader);
    uint8_t* base = simd_memset_impl(mg_alloc(header_size + total), 0, header_size + total);
    if (!base) return NULL;

    // Fill header
    MemSpanHeader* h = (MemSpanHeader*)base;
    h->growth_n = init_n;
    h->growth_m = init_m;
    h->decay_factor = decay_factor;
    // get current time via guardian platform
    guardian_time_t now = guardian_now();
    h->last_resize_ns = now.nanoseconds;
    h->payload_bytes = total;

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
