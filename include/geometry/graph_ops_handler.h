#ifndef GEOMETRY_GRAPH_OPS_HANDLER_H
#define GEOMETRY_GRAPH_OPS_HANDLER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "geometry/graph_ops.h"

/* Auto-detect and include suite implementation headers */
#if defined(__has_include)
  #if __has_include("geometry/graph_ops_int.h")
    #include "geometry/graph_ops_int.h"
    #define HAS_OP_SUITE_INT
  #endif
  #if __has_include("geometry/graph_ops_uint.h")
    #include "geometry/graph_ops_uint.h"
    #define HAS_OP_SUITE_UINT
  #endif
  /* Repeat for other primitive suite headers */
  #if __has_include("geometry/graph_ops_node.h")
    #include "geometry/graph_ops_node.h"
    #define HAS_OP_SUITE_NODE
  #endif
  #if __has_include("geometry/graph_ops_dag.h")
    #include "geometry/graph_ops_dag.h"
    #define HAS_OP_SUITE_DAG
  #endif
  #if __has_include("geometry/graph_ops_neuralnetwork.h")
    #include "geometry/graph_ops_neuralnetwork.h"
    #define HAS_OP_SUITE_NEURALNETWORK
  #endif
  #if __has_include("geometry/graph_ops_linkedlist.h")
    #include "geometry/graph_ops_linkedlist.h"
    #define HAS_OP_SUITE_LINKEDLIST
  #endif
  #if __has_include("geometry/graph_ops_list.h")
    #include "geometry/graph_ops_list.h"
    #define HAS_OP_SUITE_LIST
  #endif
  #if __has_include("geometry/graph_ops_parallel_list.h")
    #include "geometry/graph_ops_parallel_list.h"
    #define HAS_OP_SUITE_PARALLEL_LIST
  #endif
  #if __has_include("geometry/graph_ops_dict.h")
    #include "geometry/graph_ops_dict.h"
    #define HAS_OP_SUITE_DICT
  #endif
  #if __has_include("geometry/graph_ops_set.h")
    #include "geometry/graph_ops_set.h"
    #define HAS_OP_SUITE_SET
  #endif
  #if __has_include("geometry/graph_ops_map.h")
    #include "geometry/graph_ops_map.h"
    #define HAS_OP_SUITE_MAP
  #endif
  #if __has_include("geometry/graph_ops_heap.h")
    #include "geometry/graph_ops_heap.h"
    #define HAS_OP_SUITE_HEAP
  #endif
  #if __has_include("geometry/graph_ops_mailbox.h")
    #include "geometry/graph_ops_mailbox.h"
    #define HAS_OP_SUITE_MAILBOX
  #endif
  #if __has_include("geometry/graph_ops_token.h")
    #include "geometry/graph_ops_token.h"
    #define HAS_OP_SUITE_TOKEN
  #endif
  #if __has_include("geometry/graph_ops_object_set.h")
    #include "geometry/graph_ops_object_set.h"
    #define HAS_OP_SUITE_OBJECT_SET
  #endif
  #if __has_include("geometry/graph_ops_stack.h")
    #include "geometry/graph_ops_stack.h"
    #define HAS_OP_SUITE_STACK
  #endif
  #if __has_include("geometry/graph_ops_stencil_set.h")
    #include "geometry/graph_ops_stencil_set.h"
    #define HAS_OP_SUITE_STENCIL_SET
  #endif
  #if __has_include("geometry/graph_ops_emergence.h")
    #include "geometry/graph_ops_emergence.h"
    #define HAS_OP_SUITE_EMERGENCE
  #endif
  /* Feature-type suite headers */
  #if __has_include("geometry/graph_ops_vector_int.h")
    #include "geometry/graph_ops_vector_int.h"
    #define HAS_OP_SUITE_VECTOR_INT
  #endif
  #if __has_include("geometry/graph_ops_vector_float.h")
    #include "geometry/graph_ops_vector_float.h"
    #define HAS_OP_SUITE_VECTOR_FLOAT
  #endif
  #if __has_include("geometry/graph_ops_vector_double.h")
    #include "geometry/graph_ops_vector_double.h"
    #define HAS_OP_SUITE_VECTOR_DOUBLE
  #endif
  #if __has_include("geometry/graph_ops_vector_string.h")
    #include "geometry/graph_ops_vector_string.h"
    #define HAS_OP_SUITE_VECTOR_STRING
  #endif
  #if __has_include("geometry/graph_ops_vector_boolean.h")
    #include "geometry/graph_ops_vector_boolean.h"
    #define HAS_OP_SUITE_VECTOR_BOOLEAN
  #endif
  #if __has_include("geometry/graph_ops_vector_pointer.h")
    #include "geometry/graph_ops_vector_pointer.h"
    #define HAS_OP_SUITE_VECTOR_POINTER
  #endif
  #if __has_include("geometry/graph_ops_tensor_int.h")
    #include "geometry/graph_ops_tensor_int.h"
    #define HAS_OP_SUITE_TENSOR_INT
  #endif
  #if __has_include("geometry/graph_ops_tensor_float.h")
    #include "geometry/graph_ops_tensor_float.h"
    #define HAS_OP_SUITE_TENSOR_FLOAT
  #endif
  #if __has_include("geometry/graph_ops_tensor_double.h")
    #include "geometry/graph_ops_tensor_double.h"
    #define HAS_OP_SUITE_TENSOR_DOUBLE
  #endif
  #if __has_include("geometry/graph_ops_tensor_string.h")
    #include "geometry/graph_ops_tensor_string.h"
    #define HAS_OP_SUITE_TENSOR_STRING
  #endif
  #if __has_include("geometry/graph_ops_tensor_boolean.h")
    #include "geometry/graph_ops_tensor_boolean.h"
    #define HAS_OP_SUITE_TENSOR_BOOLEAN
  #endif
  #if __has_include("geometry/graph_ops_tensor_pointer.h")
    #include "geometry/graph_ops_tensor_pointer.h"
    #define HAS_OP_SUITE_TENSOR_POINTER
  #endif
  #if __has_include("geometry/graph_ops_tensor_vector_int.h")
    #include "geometry/graph_ops_tensor_vector_int.h"
    #define HAS_OP_SUITE_TENSOR_VECTOR_INT
  #endif
  #if __has_include("geometry/graph_ops_tensor_vector_float.h")
    #include "geometry/graph_ops_tensor_vector_float.h"
    #define HAS_OP_SUITE_TENSOR_VECTOR_FLOAT
  #endif
  #if __has_include("geometry/graph_ops_tensor_vector_double.h")
    #include "geometry/graph_ops_tensor_vector_double.h"
    #define HAS_OP_SUITE_TENSOR_VECTOR_DOUBLE
  #endif
  #if __has_include("geometry/graph_ops_tensor_vector_string.h")
    #include "geometry/graph_ops_tensor_vector_string.h"
    #define HAS_OP_SUITE_TENSOR_VECTOR_STRING
  #endif
  #if __has_include("geometry/graph_ops_tensor_vector_boolean.h")
    #include "geometry/graph_ops_tensor_vector_boolean.h"
    #define HAS_OP_SUITE_TENSOR_VECTOR_BOOLEAN
  #endif
  #if __has_include("geometry/graph_ops_node_feature.h")
    #include "geometry/graph_ops_node_feature.h"
    #define HAS_OP_SUITE_NODE_FEATURE
  #endif
  #if __has_include("geometry/graph_ops_edge.h")
    #include "geometry/graph_ops_edge.h"
    #define HAS_OP_SUITE_EDGE
  #endif
  #if __has_include("geometry/graph_ops_stencil.h")
    #include "geometry/graph_ops_stencil.h"
    #define HAS_OP_SUITE_STENCIL
  #endif
  #if __has_include("geometry/graph_ops_geneology.h")
    #include "geometry/graph_ops_geneology.h"
    #define HAS_OP_SUITE_GENEALOGY_OP
  #endif
  #if __has_include("geometry/graph_ops_emergence.h")
    /* already included above */
  #endif
  /* Additional detections for utils primitives e.g. linkedlist, dict, set, etc. */
#endif /* __has_include */

/* Array of pointers to each suite; defined in graph_ops_handler.c */
extern const OperationSuite* const OperationSuites[];

/**
 * Retrieve the OperationSuite for the given type.
 * Returns NULL if the suite is not available or type is invalid.
 */
const OperationSuite* get_operation_suite(OperationSuiteType type);

#ifdef __cplusplus
}
#endif

#endif /* GEOMETRY_GRAPH_OPS_HANDLER_H */
