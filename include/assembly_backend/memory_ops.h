#ifndef ASSEMBLY_BACKEND_MEMORY_OPS_H
#define ASSEMBLY_BACKEND_MEMORY_OPS_H

#include <stdint.h>
#include <stddef.h>
#include "assembly_backend/thread_ops.h"
#include "geometry/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------- */
/*  Core Intrinsics Quick Reference                                          */
/* ------------------------------------------------------------------------- */
/*  memory_ops relies on vectorized helpers provided by core.  After calling  */
/*  simd_init_dispatch(), the following function pointers and utilities are   */
/*  available:                                                                */
/*                                                                           */
/*    simd_add_u8_impl(dst, src, count)    - saturating add uint8 vectors     */
/*    simd_add_u16_impl(dst, src, count)   - saturating add uint16 vectors    */
/*    simd_add_s8_impl(dst, src, count)    - saturating add int8 vectors      */
/*    simd_add_f32_impl(dst, src, count)   - add float32 vectors              */
/*    simd_add_f64_impl(dst, src, count)   - add float64 vectors              */
/*    simd_and_u8_impl(dst, src, count)    - bitwise AND uint8 vectors        */
/*    simd_or_u8_impl(dst, src, count)     - bitwise OR uint8 vectors         */
/*    simd_xor_u8_impl(dst, src, count)    - bitwise XOR uint8 vectors        */
/*    simd_memcpy_impl(dst, src, bytes)    - accelerated memcpy              */
/*    simd_memset_impl(dst, val, bytes)    - accelerated memset              */
/*    simd_byte_diff_count(a, b, len)      - Hamming distance of byte arrays  */
/*    simd_aligned_alloc(size)             - 64-byte aligned allocation       */
/*    simd_aligned_free(ptr)               - release aligned memory           */
/* ------------------------------------------------------------------------- */

// Memory block history size for tracking last N resizes
#define MEMORY_BLOCK_HISTORY 8

typedef enum {
    MEM_BLOCK_FLAG_UNINITIALIZED          = 0x00,
    MEM_BLOCK_FLAG_DYNAMIC                = 0x01,
    MEM_BLOCK_FLAG_FIXED                  = 0x02,
    MEM_BLOCK_FLAG_ENCODED                = 0x04,
    MEM_BLOCK_FLAG_MANAGED                = 0x08,
} MemBlockFlags;

typedef enum {
    MEM_BLOCK_HISTORY_TYPE_ROLLING = 0,
    MEM_BLOCK_HISTORY_TYPE_UNIQUE_ROLLING = 1,
    MEM_BLOCK_HISTORY_TYPE_ORDERS_OF_AVERAGES = 2,
    MEM_BLOCK_HISTORY_TYPE_OUTLIERS = 3,
} MemBlockHistoryType;

typedef enum {
    BIT_ENCODING_SCHEME_BINARY = 0,
    BIT_ENCODING_SCHEME_GREY_CODE = 1,
    BIT_ENCODING_SCHEME_HUFFMAN = 2,
    BIT_ENCODING_SCHEME_ELIAS_GAMMA = 3,
    BIT_ENCODING_SCHEME_FIBONACCI = 4,
    BIT_ENCODING_SCHEME_ZIGZAG = 5,
    BIT_ENCODING_SCHEME_DELTA = 6,
    BIT_ENCODING_SCHEME_BURROWS_WHEELER_TRANSFORM = 7,
    BIT_ENCODING_SCHEME_RUN_LENGTH = 8,
    BIT_ENCODING_SCHEME_TERNARY_BALANCED = 9,
    BIT_ENCODING_SCHEME_PERMUTATION = 10,
    BIT_ENCODING_SCHEME_BITWISE_FOURIER = 11,
    BIT_ENCODING_SCHEME_NEURAL_FIELD = 12,
    BIT_ENCODING_SCHEME_CONTEXTUAL_BLOCK = 13,
    BIT_ENCODING_SCHEME_WAVELET_PACK = 14,
    BIT_ENCODING_SCHEME_CUSTOM_RESERVED = 15
} BitEncodingScheme;

typedef enum {
    DIFF_TAPE_FORMAT_FLAG_NONE = 0x00,
    DIFF_TAPE_FORMAT_FLAG_UNTIMED = 0x01,
    DIFF_TAPE_FORMAT_FLAG_TIMED_US = 0x02,
    DIFF_TAPE_FORMAT_FLAG_TIMED_NS = 0x04,
    DIFF_TAPE_FORMAT_FLAG_TIMED_MS = 0x08,
    DIFF_TAPE_FORMAT_FLAG_TIMED_S = 0x10,
    DIFF_TAPE_FORMAT_FLAG_GLOBAL_ITERATION = 0x20,
    DIFF_TAPE_FORMAT_FLAG_LOCAL_ITERATION = 0x40,
    DIFF_TAPE_FORMAT_FLAG_FULL = 0x80,
    DIFF_TAPE_FORMAT_FLAG_COMPRESSED = 0x100,
} DiffTapeFormat;

typedef enum {
    MEMORY_COMPARE_MODE_PAD = 0,
    MEMORY_COMPARE_MODE_TRUNCATE = 1,
    MEMORY_COMPARE_MODE_SCALE_UP = 2,
    MEMORY_COMPARE_MODE_SCALE_DOWN = 3,
} MemoryCompareMode;

typedef enum {
    MEMORY_COMPARE_FLAG_NONE = 0x00,
    MEMORY_COMPARE_FLAG_IGNORE_PADDING = 0x01,
    MEMORY_COMPARE_FLAG_IGNORE_ZEROES = 0x02,
    MEMORY_COMPARE_FLAG_SCALING_ANTIALIASING = 0x04,
    MEMORY_COMPARE_FLAG_PRESERVE_EDGES = 0x08,
    MEMORY_COMPARE_FLAG_ALLOW_TRUNCATION = 0x10,
    MEMORY_COMPARE_FLAG_ALLOW_PADDING = 0x20,
    MEMORY_COMPARE_FLAG_FLATTEN_OBJECTS_TO_MEAN = 0x40,
    MEMORY_COMPARE_FLAG_FLATTEN_OBJECTS_TO_MAX = 0x80,
    MEMORY_COMPARE_FLAG_FLATTEN_OBJECTS_TO_MIN = 0x100,
    MEMORY_COMPARE_FLAG_FLATTEN_OBJECTS_TO_SUM = 0x200,
    MEMORY_COMPARE_FLAG_FLATTEN_OBJECTS_TO_HAMMING = 0x400,
    MEMORY_COMPARE_FLAG_FLATTEN_OBJECTS_TO_JACCARD = 0x800,
    MEMORY_COMPARE_FLAG_FLATTEN_OBJECTS_TO_COSINE = 0x1000,
    MEMORY_COMPARE_FLAG_FLATTEN_OBJECTS_TO_EUCLIDEAN = 0x2000,
    MEMORY_COMPARE_FLAG_FLATTEN_OBJECTS_TO_MANHATTAN = 0x4000,
    MEMORY_COMPARE_FLAG_FLATTEN_OBJECTS_TO_CHEBYSHEV = 0x8000,
    MEMORY_COMPARE_FLAG_FLATTEN_OBJECTS_TO_MINKOWSKI = 0x10000,
} MemoryCompareFlags;

typedef struct {
    void* object;
    mutex_t mutex;
    long long token;
    size_t size;
    size_t index;
} object;

typedef struct {
    object* from;
    object* to;
    size_t weight;
} BidirectEdgePointers;

typedef struct {
    object* objects;
    BidirectEdgePointers* edges;
    size_t object_count;
    size_t edge_count;
    size_t total_size;
} SuballocationGraphData;

typedef struct {
    float n;
    float m;
    float growth_n;
    float growth_m;
    float decay_n;
    float decay_m;
    uint64_t timestamp_ns[MEMORY_BLOCK_HISTORY];
    size_t payload_bytes[MEMORY_BLOCK_HISTORY];
    void* diff_tape;
    SuballocationGraphData* suballocation_graph_data;
    DiffTapeFormat diff_tape_flags;
    BitEncodingScheme encoding_scheme;
    MemBlockHistoryType history_type;
    MemBlockFlags flags;
} MemSpanHeader;

typedef struct {
    const char* field_name;
    uint8_t offset;
    uint8_t size;
    uint8_t role;
    uint8_t align;
} FieldTemplate;

typedef struct {
    const char* struct_name;
    size_t struct_size;
    size_t payload_size;
    size_t extra_space;
    const FieldTemplate* fields;
    size_t field_count;
} StructLayout;

// -- CORE FUNCTIONS --

void ** apply_mask(void **array, const int **mask, size_t count, int invert_mask);
GuardianObjectSet* instantiate_on_input_cache(NodeFeatureIndex type);
GuardianObjectSet** instantiate_on_input_cache_with_count(NodeFeatureIndex type, size_t count);
GuardianObjectSet** instantiate_on_input_cache_with_count_and_raw_create(NodeFeatureIndex type, size_t count, boolean raw_create);
void ** mg_bulk_initialize(void *ptr, NodeFeatureIndex type, size_t count);
void* compose_structured_block(const StructLayout* layout, int instance_id, void* user_payload, const MemSpanHeader* default_header);
void* mg_encode_block(const void* raw_data, size_t size_bytes, uint16_t type_id, uint8_t flags);
void* mg_decode_block(const void* encoded_block, size_t* out_payload_size);
void* mg_tensor_compare(const void* block_a, const void* block_b);
void* memops_span_alloc(float init_n, float init_m, float decay_factor);
void  memops_span_free(void* span_ptr);
void* memops_span_get_data(const MemSpanHeader* header);
struct GuardianLinkNode* memops_init_linked_nodes(size_t count);
void memory_ops_retire(void* obj, NodeFeatureIndex type);

// -- GRAPH MANAGEMENT --

boolean addObjectToGraph(SuballocationGraphData* graph, object* obj);
boolean removeObjectFromGraph(SuballocationGraphData* graph, object* obj);
boolean tokenValidate(object* obj, long long token);
boolean initializeObject(object* obj, long long token, mutex_t* mutex, size_t size, void* data);
SuballocationGraphData* suballocation_graph_create(void* objects, void* edges, size_t object_count, size_t edge_count);
GuardianSimpleGraph* getSuballocationGraph(SuballocationGraphData* graph);
SuballocationGraphData* guardianGraphToSuballocationGraph(GuardianSimpleGraph* graph);

// -- EXTENDED GRAPH FUNCTIONS --

object* findObjectByToken(SuballocationGraphData* graph, long long token);
BidirectEdgePointers* findEdgeBetween(SuballocationGraphData* graph, object* from, object* to);
boolean linkObjects(SuballocationGraphData* graph, object* from, object* to, size_t weight);
boolean unlinkObjects(SuballocationGraphData* graph, object* from, object* to);
void memops_subgraph_free(SuballocationGraphData* graph);
void memops_object_clear(object* obj);

// -- SPAN METADATA ACCESSORS --

void memops_span_update_payload(MemSpanHeader* header, size_t new_size);
void memops_span_record_resize(MemSpanHeader* header, size_t new_size);
size_t memops_span_get_payload_size(const MemSpanHeader* header);
float memops_span_growth_factor(const MemSpanHeader* header);
float memops_span_decay_factor(const MemSpanHeader* header);

// -- ENCODING DISPATCH --

void* memops_encode_dispatch(const void* raw, size_t size, BitEncodingScheme scheme);
void* memops_decode_dispatch(const void* encoded, size_t* out_size, BitEncodingScheme scheme);
const char* bit_encoding_scheme_name(BitEncodingScheme scheme);
size_t bit_encoding_estimate_output_size(BitEncodingScheme scheme, size_t input_bytes);

// -- COMPARISON FRAMEWORK --

float memops_compare_blocks(const void* a, const void* b, MemoryCompareMode mode, MemoryCompareFlags flags);
float memops_distance_metric(const void* a, const void* b, MemoryCompareFlags metric);
void* memops_flatten_block(const void* block, MemoryCompareFlags flatten_mode);

// -- DEBUG & INTROSPECTION --

void memops_print_span_info(const MemSpanHeader* header);
void memops_print_object(const object* obj);
void memops_debug_graph(const SuballocationGraphData* graph);
void memops_diff_tape_clear(void* diff_tape);

#ifdef __cplusplus
}
#endif

#endif /* ASSEMBLY_BACKEND_MEMORY_OPS_H */
