#include "geometry/graph_ops_handler.h"

/* OperationSuite table mapping each NodeFeatureIndex to its implementation.
   All suites are currently stubs defined in the corresponding headers. */
const OperationSuite* const OperationSuites[NODE_FEATURE_IDX_COUNT] = {
    &graph_ops_int,                /* NODE_FEATURE_TYPE_INT */
    &graph_ops_float,              /* NODE_FEATURE_TYPE_FLOAT */
    &graph_ops_double,             /* NODE_FEATURE_TYPE_DOUBLE */
    &graph_ops_string,             /* NODE_FEATURE_TYPE_STRING */
    &graph_ops_boolean,            /* NODE_FEATURE_TYPE_BOOLEAN */
    &graph_ops_pointer,            /* NODE_FEATURE_TYPE_POINTER */
    &graph_ops_complex,            /* NODE_FEATURE_TYPE_COMPLEX */
    &graph_ops_complex_double,     /* NODE_FEATURE_TYPE_COMPLEX_DOUBLE */
    &graph_ops_vector,             /* NODE_FEATURE_TYPE_VECTOR */
    &graph_ops_tensor,             /* NODE_FEATURE_TYPE_TENSOR */
    &graph_ops_node,               /* NODE_FEATURE_TYPE_NODE */
    &graph_ops_edge,               /* NODE_FEATURE_TYPE_EDGE */
    &graph_ops_stencil,            /* NODE_FEATURE_TYPE_STENCIL */
    &graph_ops_genealogy,          /* NODE_FEATURE_TYPE_GENEALOGY */
    &graph_ops_emergence,          /* NODE_FEATURE_TYPE_EMERGENCE */
    &graph_ops_linked_list,        /* NODE_FEATURE_TYPE_LINKED_LIST */
    &graph_ops_dictionary,         /* NODE_FEATURE_TYPE_DICTIONARY */
    &graph_ops_set,                /* NODE_FEATURE_TYPE_SET */
    &graph_ops_map,                /* NODE_FEATURE_TYPE_MAP */
    &graph_ops_parallel_list,      /* NODE_FEATURE_TYPE_PARALLEL_LIST */
    &graph_ops_list,               /* NODE_FEATURE_TYPE_LIST */
    &graph_ops_pointer_token,      /* NODE_FEATURE_TYPE_POINTER_TOKEN */
    &graph_ops_guardian,           /* NODE_FEATURE_TYPE_GUARDIAN */
    &graph_ops_token,              /* NODE_FEATURE_TYPE_TOKEN */
    &graph_ops_object_set,         /* NODE_FEATURE_TYPE_OBJECT_SET */
    &graph_ops_memory_token,       /* NODE_FEATURE_TYPE_MEMORY_TOKEN */
    &graph_ops_token_lock,         /* NODE_FEATURE_TYPE_TOKEN_LOCK */
    &graph_ops_memory_map,         /* NODE_FEATURE_TYPE_MEMORY_MAP */
    &graph_ops_message,            /* NODE_FEATURE_TYPE_MESSAGE */
    &graph_ops_custom,             /* NODE_FEATURE_TYPE_CUSTOM */
    &graph_ops_bitfield,            /* NODE_FEATURE_TYPE_BITFIELD */
    &graph_ops_bytefield,           /* NODE_FEATURE_TYPE_BYTEFIELD */
    &graph_ops_radix_encoding,      /* NODE_FEATURE_TYPE_RADIX_ENCODING */
    &graph_ops_pattern_palette_stream, /* NODE_FEATURE_TYPE_PATTERN_PALETTE_STREAM */
    &graph_ops_encoding_engine      /* NODE_FEATURE_TYPE_ENCODING_ENGINE */
};

const OperationSuite* get_operation_suite(NodeFeatureType type) {
    NodeFeatureIndex idx = (NodeFeatureIndex)(type & NODE_FEATURE_INDEX_MASK);
    if (idx >= NODE_FEATURE_IDX_COUNT) {
        return NULL;
    }
    return OperationSuites[idx];
}
