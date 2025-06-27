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
typedef struct GuardianLinkNode GuardianLinkNode;

void* mg_link_factory(long long id, size_t size, int count, void* template, void* field_functions) {

    //algorithm to determine the size of all links data plus links plus projected space for dyanmica allotment according to the initial state of the memory block header properties defining such

    size_t total_size = sizeof(&((GuardianLinkNode*)(template))) * count + size * count;

    return NULL;
}