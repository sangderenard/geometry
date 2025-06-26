#ifndef GEOMETRY_GRAPH_OPS_H
#define GEOMETRY_GRAPH_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

typedef enum {
	NODE_FEATURE_TYPE_INT = 0, // Integer feature
	NODE_FEATURE_TYPE_FLOAT = 1, // Float feature
	NODE_FEATURE_TYPE_DOUBLE = 2, // Double feature
	NODE_FEATURE_TYPE_STRING = 3, // String feature
	NODE_FEATURE_TYPE_BOOLEAN = 4, // Boolean feature
	NODE_FEATURE_TYPE_POINTER = 5, // Pointer feature
	NODE_FEATURE_TYPE_VECTOR_INT = 6, // Vector feature
	NODE_FEATURE_TYPE_VECTOR_FLOAT = 7, // Vector feature
	NODE_FEATURE_TYPE_VECTOR_DOUBLE = 8, // Vector feature
	NODE_FEATURE_TYPE_VECTOR_STRING = 9, // Vector feature
	NODE_FEATURE_TYPE_VECTOR_BOOLEAN = 10, // Vector feature
	NODE_FEATURE_TYPE_VECTOR_POINTER = 11, // Vector feature
	NODE_FEATURE_TYPE_TENSOR_INT = 12, // Tensor feature
	NODE_FEATURE_TYPE_TENSOR_FLOAT = 13, // Tensor feature
	NODE_FEATURE_TYPE_TENSOR_DOUBLE = 14, // Tensor feature
	NODE_FEATURE_TYPE_TENSOR_STRING = 15, // Tensor feature
	NODE_FEATURE_TYPE_TENSOR_BOOLEAN = 16, // Tensor feature
	NODE_FEATURE_TYPE_TENSOR_POINTER = 17, // Tensor feature
	NODE_FEATURE_TYPE_TENSOR_VECTOR_INT = 18, // Tensor of vectors of integers
	NODE_FEATURE_TYPE_TENSOR_VECTOR_FLOAT = 19, // Tensor of vectors of floats
	NODE_FEATURE_TYPE_TENSOR_VECTOR_DOUBLE = 20, // Tensor of vectors of doubles
	NODE_FEATURE_TYPE_TENSOR_VECTOR_STRING = 21, // Tensor of vectors of strings
	NODE_FEATURE_TYPE_TENSOR_VECTOR_BOOLEAN = 22, // Tensor of vectors of booleans
	NODE_FEATURE_TYPE_TENSOR_VECTOR_POINTER = 23, // Tensor of vectors of pointers
	NODE_FEATURE_TYPE_NODE = 24, // Node feature type
	NODE_FEATURE_TYPE_EDGE = 25, // Edge feature type
	NODE_FEATURE_TYPE_STENCIL = 26, // Stencil feature type
	NODE_FEATURE_TYPE_GENEALOGY = 27, // Genealogy feature type
	NODE_FEATURE_TYPE_EMERGENCE = 28, // Emergence feature type
	NODE_FEATURE_TYPE_LINKED_LIST = 29, // Linked list feature type
	NODE_FEATURE_TYPE_DICTIONARY = 30, // Dictionary feature type
	NODE_FEATURE_TYPE_SET = 31, // Set feature type
	NODE_FEATURE_TYPE_MAP = 32, // Map feature type
	NODE_FEATURE_TYPE_PARALLEL_LIST = 33, // Parallel list feature type
	NODE_FEATURE_TYPE_LIST = 34, // List feature type
	NODE_FEATURE_TYPE_POINTER_TOKEN = 35, // Pointer token feature type
	NODE_FEATURE_TYPE_GUARDIAN = 36, // Guardian feature type
	NODE_FEATURE_TYPE_TOKEN = 37, // Guardian token feature type
	NODE_FEATURE_TYPE_OBJECT_SET = 38, // Guardian object set feature type
	NODE_FEATURE_TYPE_MEMORY_TOKEN = 39, // Memory token feature type
	NODE_FEATURE_TYPE_TOKEN_LOCK = 40, // Token lock feature type
	NODE_FEATURE_TYPE_MEMORY_MAP = 41, // Memory map feature type
        NODE_FEATURE_TYPE_MESSAGE = 42, // Guardian message feature type
        NODE_FEATURE_TYPE_CUSTOM = 43, // Custom feature type for user-defined features
        NODE_FEATURE_TYPE_COMPLEX = 44, // Complex number (float precision)
        NODE_FEATURE_TYPE_COMPLEX_DOUBLE = 45, // Complex number (double precision)
        NODE_FEATURE_TYPE_VECTOR_COMPLEX = 46, // Vector of complex numbers (float)
        NODE_FEATURE_TYPE_VECTOR_COMPLEX_DOUBLE = 47, // Vector of complex numbers (double)
        NODE_FEATURE_TYPE_TENSOR_COMPLEX = 48, // Tensor of complex numbers (float)
        NODE_FEATURE_TYPE_TENSOR_COMPLEX_DOUBLE = 49, // Tensor of complex numbers (double)
        NODE_FEATURE_TYPE_TENSOR_VECTOR_COMPLEX = 50, // Tensor of vectors of complex numbers (float)
        NODE_FEATURE_TYPE_TENSOR_VECTOR_COMPLEX_DOUBLE = 51, // Tensor of vectors of complex numbers (double)
        NODE_FEATURE_TYPE_COUNT
} NodeFeatureType;

/* Alias for compatibility with the operation suite API */
typedef NodeFeatureType OperationSuiteType;


/* Generic function-pointer typedefs for container operations */
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

/* --- Graph Math Operations --- */
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

typedef void* (*graph_matrix_pad_fn)(void* a, size_t top, size_t bottom,
                                     size_t left, size_t right,
                                     GraphOpRegion region);
typedef void* (*graph_make_symmetric_fn)(void* a, GraphOpRegion region);

typedef void* (*graph_union_fn)(void* a, void* b, GraphOpRegion region);
typedef void* (*graph_intersection_fn)(void* a, void* b, GraphOpRegion region);
typedef void* (*graph_difference_fn)(void* a, void* b, GraphOpRegion region);
typedef int   (*graph_subset_fn)(void* a, void* b, GraphOpRegion region);

typedef void  (*graph_make_contiguous_fn)(void* container);
typedef void  (*graph_sync_fn)(void* container);
typedef void* (*graph_factorize_fn)(void* a, GraphOpRegion region);
typedef void* (*graph_gcf_fn)(void* a, void* b, GraphOpRegion region);
typedef void* (*graph_lcm_fn)(void* a, void* b, GraphOpRegion region);

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

typedef void* (*graph_diffuse_fn)(
    void* graph,
    DiffusionModel model,
    DiffusionSourceMode source,
    DiffusionSinkMode sink,
    double rate,
    int iterations,
    GraphOpRegion region
);

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
    graph_make_contiguous_fn make_contiguous;
    graph_sync_fn         sync;
    graph_factorize_fn    factorize;
    graph_gcf_fn          gcf;
    graph_lcm_fn          lcm;
    graph_diffuse_fn     diffuse;
} GraphMathOps;

/* Suite of operations for a specific data-type container */
typedef struct {
    OpTranslatePtrFn translate_ptr;

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

    /* Extended mathematical operations */
    GraphMathOps math_ops;
} OperationSuite;


/* Array of pointers to implementations for each suite type */
extern const OperationSuite* const OperationSuites[];

#ifdef __cplusplus
}
#endif

#endif /* GEOMETRY_GRAPH_OPS_H */
