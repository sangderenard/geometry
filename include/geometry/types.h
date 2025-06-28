#ifndef GEOMETRY_TYPES_H
#define GEOMETRY_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/* Basic boolean and byte types */
typedef unsigned char boolean;
#define true 1
#define false 0

typedef unsigned char byte;

/* Platform neutral time structure */
typedef struct {
    int64_t milliseconds;
    int64_t microseconds;
    int64_t nanoseconds;
} guardian_time_t;

/* ======================= */
/* Parametric Domain Types */
/* ======================= */
#define PD_MAX_DIM 16

typedef enum {
    PD_BOUNDARY_INCLUSIVE,
    PD_BOUNDARY_EXCLUSIVE,
    PD_BOUNDARY_POS_INFINITY,
    PD_BOUNDARY_NEG_INFINITY
} BoundaryType;

typedef enum {
    PD_PERIODIC_NONE,
    PD_PERIODIC_SIMPLE,
    PD_PERIODIC_MIRROR,
    PD_PERIODIC_CYCLIC,
    PD_PERIODIC_POLARIZED
} PeriodicityType;

typedef enum {
    PD_BC_DIRICHLET,
    PD_BC_NEUMANN,
    PD_BC_CUSTOM
} BoundaryConditionType;

typedef struct DiscontinuityNode {
    double position;
    BoundaryType type;
    struct DiscontinuityNode* next;
    struct DiscontinuityNode* prev;
} DiscontinuityNode;

typedef struct AxisDescriptor {
    double start;
    double end;
    bool periodic;
    PeriodicityType periodicity_type;
    BoundaryType start_boundary;
    BoundaryType end_boundary;

    BoundaryConditionType bc_type;
    void (*bc_function)(double*, size_t);
    DiscontinuityNode* discontinuities;

    double (*extrapolate_neg)(double);
    double (*extrapolate_pos)(double);
} AxisDescriptor;

typedef struct ParametricDomain {
    size_t dim;
    AxisDescriptor axes[PD_MAX_DIM];
} ParametricDomain;

/* ===== DAG / Neural Network Types ===== */

typedef struct DAGNode DAGNode;
typedef struct DAGEdge DAGEdge;

typedef void (*DAGForwardFn)(DAGNode* self);
typedef void (*DAGBackwardFn)(DAGNode* self);

typedef struct {
    void* data;
} DAGParams;

struct DAGNode {
    DAGForwardFn forward;
    DAGBackwardFn backward;
    DAGNode** inputs;
    size_t num_inputs;
    DAGParams* params;
    void* output;
    void* grad;
    DAGNode* next;
};

typedef struct {
    struct Node** inputs;
    size_t num_inputs;
    struct Node** outputs;
    size_t num_outputs;
} DagManifestMapping;

typedef struct {
    DagManifestMapping* mappings;
    size_t num_mappings;
    int level_index;
} DagManifestLevel;

typedef struct {
    DagManifestLevel* levels;
    size_t num_levels;
} DagManifest;

typedef struct Dag {
    DagManifest* manifests;
    size_t num_manifests, cap_manifests;
} Dag;

typedef struct NeuralNetwork NeuralNetwork;

typedef void (*NNForwardFn)(struct Node** inputs, size_t num_inputs, struct Node** outputs, size_t num_outputs, void* user);
typedef void (*NNBackwardFn)(struct Node** inputs, size_t num_inputs, struct Node** outputs, size_t num_outputs, void* user);

typedef struct {
    DagManifestMapping* mapping;
    NNForwardFn forward;
    NNBackwardFn backward;
    void* user_data;
} NeuralNetworkStep;

#define NN_MAX_FUNCTIONS 32

typedef struct {
    const char* name;
    NNForwardFn forward;
    NNBackwardFn backward;
} NeuralNetworkFunctionEntry;

typedef struct {
    NeuralNetworkFunctionEntry entries[NN_MAX_FUNCTIONS];
    size_t num_entries;
} NeuralNetworkFunctionRepo;

#define NN_MAX_DAGS 8
#define NN_MAX_STEPS 256

typedef struct NeuralNetwork {
    Dag* dags[NN_MAX_DAGS];
    size_t num_dags;
    NeuralNetworkStep* steps[NN_MAX_DAGS][NN_MAX_STEPS];
    size_t num_steps[NN_MAX_DAGS];
    NeuralNetworkFunctionRepo function_repo;
} NeuralNetwork;

/* ===== Execution Graph ===== */
#define EXEC_GRAPH_MAX_NODES 64

typedef struct {
    DAGNode* nodes[EXEC_GRAPH_MAX_NODES];
    size_t num_nodes;
} ExecutionGraph;

/* ===== Graph Ops Types ===== */

typedef enum {
    NODE_FEATURE_IDX_INT = 0,
    NODE_FEATURE_IDX_FLOAT,
    NODE_FEATURE_IDX_DOUBLE,
    NODE_FEATURE_IDX_STRING,
    NODE_FEATURE_IDX_BOOLEAN,
    NODE_FEATURE_IDX_POINTER,
    NODE_FEATURE_IDX_COMPLEX,
    NODE_FEATURE_IDX_COMPLEX_DOUBLE,
    NODE_FEATURE_IDX_VECTOR,
    NODE_FEATURE_IDX_TENSOR,
    NODE_FEATURE_IDX_NODE,
    NODE_FEATURE_IDX_EDGE,
    NODE_FEATURE_IDX_STENCIL,
    NODE_FEATURE_IDX_GENEALOGY,
    NODE_FEATURE_IDX_EMERGENCE,
    NODE_FEATURE_IDX_LINKED_LIST,
    NODE_FEATURE_IDX_DICTIONARY,
    NODE_FEATURE_IDX_SET,
    NODE_FEATURE_IDX_MAP,
    NODE_FEATURE_IDX_PARALLEL_LIST,
    NODE_FEATURE_IDX_LIST,
    NODE_FEATURE_IDX_POINTER_TOKEN,
    NODE_FEATURE_IDX_GUARDIAN,
    NODE_FEATURE_IDX_STACK,
    NODE_FEATURE_IDX_HEAP,
    NODE_FEATURE_IDX_TOKEN,
    NODE_FEATURE_IDX_OBJECT_SET,
    NODE_FEATURE_IDX_MEMORY_TOKEN,
    NODE_FEATURE_IDX_TOKEN_LOCK,
    NODE_FEATURE_IDX_MEMORY_MAP,
    NODE_FEATURE_IDX_MESSAGE,
    NODE_FEATURE_IDX_CUSTOM,
    NODE_FEATURE_IDX_BITFIELD,
    NODE_FEATURE_IDX_BYTEFIELD,
    NODE_FEATURE_IDX_RADIX_ENCODING,
    NODE_FEATURE_IDX_PATTERN_PALETTE_STREAM,
    NODE_FEATURE_IDX_ENCODING_ENGINE,
    NODE_FEATURE_IDX_COUNT,
    NODE_FEATURE_IDX_MUTEX_T,
    NODE_FEATURE_IDX_STENCIL_SET,
} NodeFeatureIndex;

typedef enum {
    NODE_CATEGORY_PRIMITIVE  = 1ULL << 16,
    NODE_CATEGORY_VECTOR     = 1ULL << 17,
    NODE_CATEGORY_TENSOR     = 1ULL << 18,
    NODE_CATEGORY_GRAPH      = 1ULL << 19,
    NODE_CATEGORY_COLLECTION = 1ULL << 20,
    NODE_CATEGORY_GUARDIAN   = 1ULL << 21,
    NODE_CATEGORY_ENCODING   = 1ULL << 22,
    NODE_CATEGORY_CUSTOM     = 1ULL << 23
} NodeFeatureCategory;

#define NODE_FEATURE_INDEX_MASK 0xFFFFULL

typedef unsigned long long NodeFeatureType;

#define NODE_FEATURE_TYPE_INT (NODE_CATEGORY_PRIMITIVE | NODE_FEATURE_IDX_INT)
#define NODE_FEATURE_TYPE_FLOAT (NODE_CATEGORY_PRIMITIVE | NODE_FEATURE_IDX_FLOAT)
#define NODE_FEATURE_TYPE_DOUBLE (NODE_CATEGORY_PRIMITIVE | NODE_FEATURE_IDX_DOUBLE)
#define NODE_FEATURE_TYPE_STRING (NODE_CATEGORY_PRIMITIVE | NODE_FEATURE_IDX_STRING)
#define NODE_FEATURE_TYPE_BOOLEAN (NODE_CATEGORY_PRIMITIVE | NODE_FEATURE_IDX_BOOLEAN)
#define NODE_FEATURE_TYPE_POINTER (NODE_CATEGORY_PRIMITIVE | NODE_FEATURE_IDX_POINTER)
#define NODE_FEATURE_TYPE_VECTOR (NODE_CATEGORY_COLLECTION | NODE_CATEGORY_VECTOR | NODE_FEATURE_IDX_VECTOR)
#define NODE_FEATURE_TYPE_TENSOR (NODE_CATEGORY_COLLECTION | NODE_CATEGORY_TENSOR | NODE_FEATURE_IDX_TENSOR)
#define NODE_FEATURE_TYPE_NODE (NODE_CATEGORY_GRAPH | NODE_FEATURE_IDX_NODE)
#define NODE_FEATURE_TYPE_EDGE (NODE_CATEGORY_GRAPH | NODE_FEATURE_IDX_EDGE)
#define NODE_FEATURE_TYPE_STENCIL (NODE_CATEGORY_GRAPH | NODE_FEATURE_IDX_STENCIL)
#define NODE_FEATURE_TYPE_GENEALOGY (NODE_CATEGORY_GRAPH | NODE_FEATURE_IDX_GENEALOGY)
#define NODE_FEATURE_TYPE_EMERGENCE (NODE_CATEGORY_GRAPH | NODE_FEATURE_IDX_EMERGENCE)
#define NODE_FEATURE_TYPE_LINKED_LIST (NODE_CATEGORY_COLLECTION | NODE_FEATURE_IDX_LINKED_LIST)
#define NODE_FEATURE_TYPE_DICTIONARY (NODE_CATEGORY_COLLECTION | NODE_FEATURE_IDX_DICTIONARY)
#define NODE_FEATURE_TYPE_SET (NODE_CATEGORY_COLLECTION | NODE_FEATURE_IDX_SET)
#define NODE_FEATURE_TYPE_MAP (NODE_CATEGORY_COLLECTION | NODE_FEATURE_IDX_MAP)
#define NODE_FEATURE_TYPE_PARALLEL_LIST (NODE_CATEGORY_COLLECTION | NODE_FEATURE_IDX_PARALLEL_LIST)
#define NODE_FEATURE_TYPE_LIST (NODE_CATEGORY_COLLECTION | NODE_FEATURE_IDX_LIST)
#define NODE_FEATURE_TYPE_POINTER_TOKEN (NODE_CATEGORY_GUARDIAN | NODE_FEATURE_IDX_POINTER_TOKEN)
#define NODE_FEATURE_TYPE_GUARDIAN (NODE_CATEGORY_GUARDIAN | NODE_FEATURE_IDX_GUARDIAN)
#define NODE_FEATURE_TYPE_TOKEN (NODE_CATEGORY_GUARDIAN | NODE_FEATURE_IDX_TOKEN)
#define NODE_FEATURE_TYPE_OBJECT_SET (NODE_CATEGORY_GUARDIAN | NODE_FEATURE_IDX_OBJECT_SET)
#define NODE_FEATURE_TYPE_MEMORY_TOKEN (NODE_CATEGORY_GUARDIAN | NODE_FEATURE_IDX_MEMORY_TOKEN)
#define NODE_FEATURE_TYPE_TOKEN_LOCK (NODE_CATEGORY_GUARDIAN | NODE_FEATURE_IDX_TOKEN_LOCK)
#define NODE_FEATURE_TYPE_MEMORY_MAP (NODE_CATEGORY_GUARDIAN | NODE_FEATURE_IDX_MEMORY_MAP)
#define NODE_FEATURE_TYPE_MESSAGE (NODE_CATEGORY_GUARDIAN | NODE_FEATURE_IDX_MESSAGE)
#define NODE_FEATURE_TYPE_CUSTOM (NODE_CATEGORY_CUSTOM | NODE_FEATURE_IDX_CUSTOM)
#define NODE_FEATURE_TYPE_COMPLEX (NODE_CATEGORY_PRIMITIVE | NODE_FEATURE_IDX_COMPLEX)
#define NODE_FEATURE_TYPE_COMPLEX_DOUBLE (NODE_CATEGORY_PRIMITIVE | NODE_FEATURE_IDX_COMPLEX_DOUBLE)
#define NODE_FEATURE_TYPE_BITFIELD (NODE_CATEGORY_ENCODING | NODE_FEATURE_IDX_BITFIELD)
#define NODE_FEATURE_TYPE_BYTEFIELD (NODE_CATEGORY_ENCODING | NODE_FEATURE_IDX_BYTEFIELD)
#define NODE_FEATURE_TYPE_RADIX_ENCODING (NODE_CATEGORY_ENCODING | NODE_FEATURE_IDX_RADIX_ENCODING)
#define NODE_FEATURE_TYPE_PATTERN_PALETTE_STREAM (NODE_CATEGORY_ENCODING | NODE_FEATURE_IDX_PATTERN_PALETTE_STREAM)
#define NODE_FEATURE_TYPE_ENCODING_ENGINE (NODE_CATEGORY_ENCODING | NODE_FEATURE_IDX_ENCODING_ENGINE)
#define NODE_FEATURE_TYPE_COUNT NODE_FEATURE_IDX_COUNT

typedef NodeFeatureType OperationSuiteType;

/* Function pointer typedefs for container operations */
typedef void*   (*OpCreateFn)(void);
typedef void    (*OpDestroyFn)(void* container);
typedef void    (*OpClearFn)(void* container);
typedef void    (*OpPushFn)(void* container, void* element);
typedef void*   (*OpPopFn)(void* container);
typedef void*   (*OpShiftFn)(void* container);
typedef void    (*OpUnshiftFn)(void* container, void* element);
typedef void    (*OpInsertFn)(void* container, size_t index, void* element);
typedef void*   (*OpRemoveFn)(void* container, size_t index);
typedef void*   (*OpGetFn)(void* container, size_t index);
typedef void    (*OpSetFn)(void* container, size_t index, void* element);
typedef size_t  (*OpSizeFn)(const void* container);
typedef void    (*OpSortFn)(void* container, int (*cmp)(const void*, const void*));
typedef void*   (*OpSearchFn)(void* container, int (*pred)(const void*, void*), void* user_data);
typedef void    (*OpSliceFn)(void* container, size_t start, size_t end, void** out_array);
typedef void    (*OpStencilFn)(void* container, const size_t* indices, size_t count, void** out_array);
typedef void    (*OpForEachFn)(void* container, void (*fn)(void* element, void* user_data), void* user_data);
typedef void*   (*OpMapFn)(void* container, void* (*map_fn)(void* element, void* user_data), void* user_data);
typedef void*   (*OpFilterFn)(void* container, int (*pred)(void* element, void* user_data), void* user_data);
typedef void*   (*OpReduceFn)(void* container, void* (*reduce_fn)(void* acc, void* element, void* user_data), void* user_data, void* initial);
typedef void*   (*OpCloneFn)(const void* container);
typedef void    (*OpMergeFn)(void* dest, const void* src);
typedef int     (*OpSerializeFn)(const void* container, void* out_buffer, size_t buffer_size);
typedef void*   (*OpDeserializeFn)(const void* in_buffer, size_t buffer_size);
typedef void    (*OpLockFn)(void* container);
typedef void    (*OpUnlockFn)(void* container);
typedef int     (*OpGetRelationshipFn)(void* container, void* a, void* b);
typedef void*   (*OpDiffFn)(void* container);
typedef void*   (*OpIntegrateFn)(void* container);
typedef void*   (*OpInterpolateFn)(void* container, double t);
typedef void*   (*OpProbabilisticSelectFn)(void* container, void* histogram);
typedef void*   (*OpDiscretizeFn)(void* container, double resolution);
typedef void*   (*OpLaplaceFn)(void* container);
typedef void*   (*OpStepFn)(void* container, double t);
typedef void*   (*OpQuantizeFn)(void* container);
typedef void*   (*OpNormalizeFn)(void* container);
typedef void*   (*OpHistogramFn)(void* container, size_t bins);
typedef void*   (*OpQuantileHistogramFn)(void* container, size_t quantiles);
typedef void*   (*OpRankOrderNormalizeFn)(void* container);

/* Graph Math Operations */
typedef enum {
    GRAPH_OP_REGION_DOMAIN = 0,
    GRAPH_OP_REGION_SUBDOMAIN,
    GRAPH_OP_REGION_SLICE,
    GRAPH_OP_REGION_KERNEL
} GraphOpRegion;

typedef void* (*graph_add_fn)(void* a, void* b, GraphOpRegion region);
typedef void* (*graph_subtract_fn)(void* a, void* b, GraphOpRegion region);
typedef void* (*graph_multiply_fn)(void* a, void* b, GraphOpRegion region);
typedef void* (*graph_divide_fn)(void* a, void* b, GraphOpRegion region);
typedef void* (*graph_negate_fn)(void* a, GraphOpRegion region);

typedef void* (*graph_transpose_fn)(void* a, GraphOpRegion region);
typedef void* (*graph_matmul_fn)(void* a, void* b, GraphOpRegion region);
typedef void* (*graph_inverse_fn)(void* a, GraphOpRegion region);
typedef void* (*graph_determinant_fn)(void* a, GraphOpRegion region);

typedef void* (*graph_matrix_pad_fn)(void* a, size_t top, size_t bottom, size_t left, size_t right, GraphOpRegion region);
typedef void* (*graph_make_symmetric_fn)(void* a, GraphOpRegion region);

typedef void* (*graph_union_fn)(void* a, void* b, GraphOpRegion region);
typedef void* (*graph_intersection_fn)(void* a, void* b, GraphOpRegion region);
typedef void* (*graph_difference_fn)(void* a, void* b, GraphOpRegion region);
typedef int   (*graph_subset_fn)(void* a, void* b, GraphOpRegion region);
typedef boolean (*graph_gap_inventory_fn)(void* container);

typedef boolean  (*graph_make_contiguous_fn)(void* container);
typedef boolean  (*graph_make_contiguous_no_wait_fn)(void* container);
typedef boolean  (*graph_make_contiguous_wait_fn)(void* container);
typedef boolean  (*graph_make_contiguous_wait_timeout_fn)(void* container);
typedef boolean  (*graph_make_contiguous_force_fn)(void* container);

typedef void  (*graph_sync_fn)(void* container);
typedef void* (*graph_factorize_fn)(void* a, GraphOpRegion region);
typedef void* (*graph_gcf_fn)(void* a, void* b, GraphOpRegion region);
typedef void* (*graph_lcm_fn)(void* a, void* b, GraphOpRegion region);

/* Bitwise Operations */
typedef void* (*graph_bit_and_fn)(void* a, void* b, GraphOpRegion region);
typedef void* (*graph_bit_or_fn)(void* a, void* b, GraphOpRegion region);
typedef void* (*graph_bit_xor_fn)(void* a, void* b, GraphOpRegion region);
typedef void* (*graph_bit_not_fn)(void* a, GraphOpRegion region);
typedef void* (*graph_bit_shift_left_fn)(void* a, int shift, GraphOpRegion region);
typedef void* (*graph_bit_shift_right_fn)(void* a, int shift, GraphOpRegion region);

typedef enum {
    DIFFUSION_MODEL_AUXIN = 0,
    DIFFUSION_MODEL_SLIME_MOLD,
    DIFFUSION_MODEL_DEPTH_PRESSURE,
    DIFFUSION_MODEL_BREADTH_PRESSURE
} DiffusionModel;

typedef enum {
    DIFFUSION_SOURCE_SINGLE = 0,
    DIFFUSION_SOURCE_UBIQUITOUS,
    DIFFUSION_SOURCE_MAP,
    DIFFUSION_SOURCE_DAG,
    DIFFUSION_SOURCE_BOUNDARY_Y
} DiffusionSourceMode;

typedef enum {
    DIFFUSION_SINK_SINGLE = 0,
    DIFFUSION_SINK_UBIQUITOUS,
    DIFFUSION_SINK_MAP,
    DIFFUSION_SINK_DAG,
    DIFFUSION_SINK_BOUNDARY_Y
} DiffusionSinkMode;

typedef void* (*graph_diffuse_fn)(void* graph, DiffusionModel model, DiffusionSourceMode source, DiffusionSinkMode sink, double rate, int iterations, GraphOpRegion region);

typedef struct {
    graph_add_fn         add;
    graph_subtract_fn    subtract;
    graph_multiply_fn    multiply;
    graph_divide_fn      divide;
    graph_negate_fn      negate;
    graph_transpose_fn   transpose;
    graph_matmul_fn      matmul;
    graph_inverse_fn     inverse;
    graph_determinant_fn determinant;
    graph_matrix_pad_fn  pad;
    graph_make_symmetric_fn make_symmetric;
    graph_union_fn       set_union;
    graph_intersection_fn set_intersection;
    graph_difference_fn  set_difference;
    graph_subset_fn      is_subset;
    graph_gap_inventory_fn gap_inventory;
    graph_make_contiguous_fn make_contiguous;
    graph_make_contiguous_no_wait_fn make_contiguous_no_wait;
    graph_make_contiguous_wait_fn make_contiguous_wait;
    graph_make_contiguous_wait_timeout_fn make_contiguous_wait_timeout;
    graph_make_contiguous_force_fn make_contiguous_force;
    graph_sync_fn         sync;
    graph_factorize_fn    factorize;
    graph_gcf_fn          gcf;
    graph_lcm_fn          lcm;
    graph_diffuse_fn     diffuse;
} GraphMathOps;

typedef struct {
    graph_bit_and_fn        bit_and;
    graph_bit_or_fn         bit_or;
    graph_bit_xor_fn        bit_xor;
    graph_bit_not_fn        bit_not;
    graph_bit_shift_left_fn shift_left;
    graph_bit_shift_right_fn shift_right;
} GraphBitOps;

typedef struct {
    void *         translate_ptr;

    OpCreateFn     create;
    OpDestroyFn    destroy;
    OpClearFn      clear;

    OpPushFn       push;
    OpPopFn        pop;
    OpShiftFn      shift;
    OpUnshiftFn    unshift;
    OpInsertFn     insert;
    OpRemoveFn     remove;

    OpGetFn        get;
    OpSetFn        set;
    OpSizeFn       size;

    OpSortFn       sort;
    OpSearchFn     search;
    OpSliceFn      slice;
    OpStencilFn    stencil;
    OpForEachFn    for_each;
    OpMapFn        map;
    OpFilterFn     filter;
    OpReduceFn     reduce;
    OpCloneFn      clone;
    OpMergeFn      merge;
    OpSerializeFn  serialize;
    OpDeserializeFn deserialize;
    OpLockFn       lock;
    OpUnlockFn     unlock;
    OpGetRelationshipFn      get_relationship;
    OpDiffFn                diff;
    OpIntegrateFn           integrate;
    OpInterpolateFn         interpolate;
    OpProbabilisticSelectFn probabilistic_select;
    OpDiscretizeFn          discretize;
    OpLaplaceFn             laplace;
    OpStepFn                step;
    OpQuantizeFn            quantize;
    OpNormalizeFn           normalize;
    OpHistogramFn           histogram;
    OpQuantileHistogramFn   quantile_histogram;
    OpRankOrderNormalizeFn  rank_order_normalize;

    GraphMathOps math_ops;
    GraphBitOps  bit_ops;
} OperationSuite;

#ifdef __cplusplus
}
#endif

#endif /* GEOMETRY_TYPES_H */
