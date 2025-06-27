#include "assembly_backend/core.h"
#include "assembly_backend/simd/simd_dispatch.h"
#include "assembly_backend/simd/memory_intrinsics.h"

void assembly_backend_core_init(void) {
    simd_init_dispatch();
}

void* mg_alloc(size_t size) {
    return simd_aligned_alloc(size);
}

void mg_free(void* ptr) {
    simd_aligned_free(ptr);
}
