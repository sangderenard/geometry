#include "assembly_backend/memory_ops.h"
#include "assembly_backend/core.h"
#include "assembly_backend/simd/simd_dispatch.h"
#include "assembly_backend/simd/memory_intrinsics.h"
#include "geometry/guardian_platform.h"
#include "geometry/types.h"
#include "geometry/graph_ops_handler.h"
#include "geometry/graph_ops_dictionary.h"
#include "geometry/utils.h"
#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

const size_t INPUT_CACHE_SIZE = 1024 * 1024 * 1024; // 1 GB

// External assembly encode/decode entry points
extern void* mg_encode_block_asm(const void* raw_data, size_t size_bytes, uint16_t type_id, uint8_t flags);
extern void* mg_decode_block_asm(const void* encoded_block, size_t* out_payload_size);

// Encode/Decode passthroughs
void* mg_encode_block(const void* raw_data, size_t size_bytes, uint16_t type_id, uint8_t flags) {
    return mg_encode_block_asm(raw_data, size_bytes, type_id, flags);
}

void* mg_decode_block(const void* encoded_block, size_t* out_payload_size) {
    return mg_decode_block_asm(encoded_block, out_payload_size);
}

void* mg_tensor_compare(const void* block_a, const void* block_b) {
    return NULL;
}

void ** mg_bulk_initialize(void *ptr, NodeFeatureIndex type, size_t count) {
    if (!ptr || count == 0) return NULL;

    GuardianObjectSet** objects = (GuardianObjectSet**)ptr;
    // USE INTRINSICS TO INITIALIZE OBJECTS ASAP
    return (void**)objects;
}

void ** apply_mask(void **array, const int **mask, size_t count, int invert_mask) {
    if (!array || !mask || count == 0) return NULL;

    size_t j = 0;
    for (size_t i = 0; i < count; ++i) {
        if ((invert_mask && !(*mask)[i]) || (!invert_mask && (*mask)[i])) {
            array[j++] = array[i];
        } else {
            array[i] = NULL; // Clear the pointer if it doesn't match the mask
        }
    }
    return array;
}

// Full implementation from original
void* compose_structured_block(const StructLayout* layout, int instance_id, void* user_payload, const MemSpanHeader* default_header) {
    const size_t struct_size = layout->struct_size;
    const size_t payload_size = layout->payload_size;
    const size_t tail_bytes = layout->extra_space;
    const size_t header_size = sizeof(MemSpanHeader);
    const size_t block_total_size = header_size + struct_size + payload_size + tail_bytes;
    const size_t aligned_size = (block_total_size + 63) & ~((size_t)63);

    /*
     * All assembly backend allocations must use the 64-byte aligned
     * allocation pathway.  mg_alloc is routed through simd_aligned_alloc,
     * which guarantees this alignment.
     */
    void* raw_block = mg_alloc(aligned_size);
    if (!raw_block) return NULL;

    MemSpanHeader* header = (MemSpanHeader*)raw_block;
    if (default_header) {
        *header = *default_header;
    } else {
        /*
         * We avoid zeroing memory automatically.  Only explicitly
         * set fields that require initialization.
         */
        header->flags = MEM_BLOCK_FLAG_DYNAMIC;
    }

    header->payload_bytes[0] = payload_size;
    header->timestamp_ns[0] = guardian_now().nanoseconds;
    header->encoding_scheme = BIT_ENCODING_SCHEME_BINARY;
    header->history_type = MEM_BLOCK_HISTORY_TYPE_ROLLING;
    header->diff_tape = NULL;
    header->diff_tape_flags = DIFF_TAPE_FORMAT_FLAG_NONE;
    return (void*)((uint8_t*)raw_block + header_size);
}

void* memops_span_alloc(float init_n, float init_m, float decay_factor) {
    uint32_t n = (uint32_t)floorf(init_n);
    uint32_t m = (uint32_t)floorf(init_m);
    uint64_t total = (uint64_t)n << m;

    size_t header_size = sizeof(MemSpanHeader);
    uint8_t* base = mg_alloc(header_size + total);
    if (!base) return NULL;

    MemSpanHeader* h = (MemSpanHeader*)base;
    h->n = init_n;
    h->m = init_m;
    h->growth_n = init_n;
    h->growth_m = init_m;
    h->decay_n = decay_factor;
    h->decay_m = decay_factor;
    h->timestamp_ns[0] = guardian_now().nanoseconds;
    h->payload_bytes[0] = total;
    h->diff_tape = NULL;
    h->diff_tape_flags = DIFF_TAPE_FORMAT_FLAG_NONE;
    h->encoding_scheme = BIT_ENCODING_SCHEME_BINARY;
    h->history_type = MEM_BLOCK_HISTORY_TYPE_ROLLING;
    h->flags = MEM_BLOCK_FLAG_DYNAMIC;

    return base + header_size;
}

void memops_span_free(void* span_ptr) {
    if (!span_ptr) return;
    uint8_t* base = (uint8_t*)span_ptr - sizeof(MemSpanHeader);
    mg_free(base);
}

void* memops_span_get_data(const MemSpanHeader* header) {
    if (!header) return NULL;
    return (void*)((const uint8_t*)header + sizeof(MemSpanHeader));
}
GuardianDict* global_pointer_dict = NULL;
void * memops_get_pointer_from_token(GuardianPointerToken* token) {
    if (!token) return NULL;

    if (!global_pointer_dict) {
        global_pointer_dict = instantiate_on_input_cache(NODE_FEATURE_IDX_DICTIONARY);
        if (!global_pointer_dict) {
            // Handle allocation failure
            return NULL;
        }
        
        graph_ops_dictionary.set(global_pointer_dict, token->token, global_pointer_dict);
    } else {
        void* return_pointer = graph_ops_dictionary.get(global_pointer_dict, token->token);
        if (!return_pointer) {
            // Handle error: dictionary not found for the token
            return NULL;
        }
        return return_pointer; // Assuming values contain the pointers
    }
}   

boolean memops_set_prev_on_link(GuardianPointerToken* node, GuardianPointerToken* prev) {
    if (!node) return false;

    GuardianLinkNode* actual_node = memops_get_pointer_from_token(node);
    if (!actual_node) return false;
    actual_node->prev = prev;
    return true;
}
boolean memops_set_next_on_link(GuardianPointerToken* node, GuardianPointerToken* next) {
    if (!node) return false;

    GuardianLinkNode* actual_node = memops_get_pointer_from_token(node);
    if (!actual_node) return false;
    actual_node->next = next;
    return true;
}
struct GuardianLinkNode* memops_init_linked_nodes(size_t count) {
    if (count == 0) return NULL;
    size_t total_size = sizeof(GuardianLinkNode) * count;
    GuardianLinkNode* nodes = (GuardianLinkNode*)mg_alloc(total_size);
    if (!nodes) return NULL;

    for (size_t i = 0; i < count; ++i) {
        nodes[i].prev = (i > 0) ? &nodes[i - 1] : NULL;
        nodes[i].next = (i + 1 < count) ? &nodes[i + 1] : NULL;
    }
    return nodes;
}

int id = 0; // Global ID counter
int initialize_id(int* id) {
    if (!id) return -1;
    *id = 0; // Initialize to zero
    return 0; // Return success
}
boolean increment_id(int* id) {
    if (!id) return false;
    (*id)++;
    return true;
}
int id_dispenser(int* id) {
    if (!id) return -1;
    if (*id < 0) {
        initialize_id(id);
    }
    increment_id(id);
    return *id;
}   

GuardianHeap * input_cache = NULL;
void * initialize_input_cache() {
    input_cache = (GuardianHeap *)mg_raw_create(NODE_FEATURE_IDX_GUARDIAN);
    if (!input_cache) {
        // Handle allocation failure
        return NULL;
    }
    return input_cache;
}

void** bulk_initialize_objects(void* storage_location, NodeFeatureIndex type, size_t count) {
    // actual creation of objects should be done in bulk when possible
    if (!storage_location || count == 0) return NULL;
    void** objects = (void**)storage_location;
    objects = mg_bulk_initialize(objects, type, count); // Initialize memory to zero
    if (!objects) {
        // Handle allocation failure
        return NULL;
    }
    return objects;
}

void ** instantiate_on_input_cache_with_count_and_raw_create(NodeFeatureIndex type, size_t count, boolean raw_create) {
    if (raw_create) {
        return mg_alloc(guardian_sizeof(type) * count);
    }
    if (!input_cache) {
        input_cache = initialize_input_cache();
        if (!input_cache) {
            // Handle allocation failure
            return NULL;
        }
    }

    // Allocate memory for the requested type and count
    size_t size = sizeof(GuardianObjectSet) * count;
    GuardianObjectSet* objects = (GuardianObjectSet*)mg_alloc(size);
    if (!objects) {
        // Handle allocation failure
        return NULL;
    }

    objects = mg_bulk_initialize(objects, type, count); // Initialize memory to zero

    return objects;
}
GuardianObjectSet ** instantiate_on_input_cache_with_count(NodeFeatureIndex type, size_t count) {
    void ** return_objects = instantiate_on_input_cache_with_count_and_raw_create(type, count, false);
    for(size_t i = 0; i < count; ++i) {
        GuardianObjectSet * obj = (GuardianObjectSet *)return_objects[i];

    }

}
GuardianObjectSet * instantiate_on_input_cache(NodeFeatureIndex type) {
    return instantiate_on_input_cache_with_count(type, 1);
}

// Skeletons for expansion
boolean addObjectToGraph(SuballocationGraphData* graph, object* obj) { return false; }
boolean removeObjectFromGraph(SuballocationGraphData* graph, object* obj) { return false; }
boolean tokenValidate(object* obj, long long token) { return false; }
boolean initializeObject(object* obj, long long token, mutex_t* mutex, size_t size, void* data) { return false; }
SuballocationGraphData* suballocation_graph_create(void* objects, void* edges, size_t object_count, size_t edge_count) { return NULL; }
GuardianSimpleGraph* getSuballocationGraph(SuballocationGraphData* graph) { return NULL; }
SuballocationGraphData* guardianGraphToSuballocationGraph(GuardianSimpleGraph* graph) { return NULL; }

object* findObjectByToken(SuballocationGraphData* graph, long long token) { return NULL; }
BidirectEdgePointers* findEdgeBetween(SuballocationGraphData* graph, object* from, object* to) { return NULL; }
boolean linkObjects(SuballocationGraphData* graph, object* from, object* to, size_t weight) { return false; }
boolean unlinkObjects(SuballocationGraphData* graph, object* from, object* to) { return false; }
void memops_subgraph_free(SuballocationGraphData* graph) {}
void memops_object_clear(object* obj) {}

void memops_span_update_payload(MemSpanHeader* header, size_t new_size) {
    if (!header) return;
    header->payload_bytes[0] = new_size;
}
GuardianObjectSet * instantiate_on_input_cache(NodeFeatureIndex type){

}
void memops_span_record_resize(MemSpanHeader* header, size_t new_size) {
    if (!header) return;
    for (int i = MEMORY_BLOCK_HISTORY - 1; i > 0; --i) {
        header->payload_bytes[i] = header->payload_bytes[i - 1];
        header->timestamp_ns[i] = header->timestamp_ns[i - 1];
    }
    header->payload_bytes[0] = new_size;
    header->timestamp_ns[0] = guardian_now().nanoseconds;
}

size_t memops_span_get_payload_size(const MemSpanHeader* header) {
    if (!header) return 0;
    return header->payload_bytes[0];
}

float memops_span_growth_factor(const MemSpanHeader* header) {
    return header ? header->growth_n : 0.0f;
}

float memops_span_decay_factor(const MemSpanHeader* header) {
    return header ? header->decay_n : 0.0f;
}

void* memops_encode_dispatch(const void* raw, size_t size, BitEncodingScheme scheme) { return NULL; }
void* memops_decode_dispatch(const void* encoded, size_t* out_size, BitEncodingScheme scheme) { return NULL; }
const char* bit_encoding_scheme_name(BitEncodingScheme scheme) { return ""; }
size_t bit_encoding_estimate_output_size(BitEncodingScheme scheme, size_t input_bytes) { return 0; }

float memops_compare_blocks(const void* a, const void* b, MemoryCompareMode mode, MemoryCompareFlags flags) { return 0.0f; }
float memops_distance_metric(const void* a, const void* b, MemoryCompareFlags metric) { return 0.0f; }
void* memops_flatten_block(const void* block, MemoryCompareFlags flatten_mode) { return NULL; }

void memops_print_span_info(const MemSpanHeader* header) {}
void memops_print_object(const object* obj) {}
void memops_debug_graph(const SuballocationGraphData* graph) {}
void memops_diff_tape_clear(void* diff_tape) {}

// Retire an object/container by freeing its memory
void memory_ops_retire(void* obj, NodeFeatureIndex type) {
    // TODO: everything that needs to be done here
    // reference checks
    // Ensure the object is not NULL before proceeding
    // recursive cleanup if necessary
    // would any blocks already be useful recycled as is?
    // such as linked lists, graphs, etc of standard types and sizes
    if (!obj) return;
    mg_free(obj);
}
