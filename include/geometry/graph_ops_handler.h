#ifndef GEOMETRY_GRAPH_OPS_HANDLER_H
#define GEOMETRY_GRAPH_OPS_HANDLER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "geometry/graph_ops.h"

/* Include operation suites for all known node feature types. */
#include "geometry/graph_ops_int.h"
#include "geometry/graph_ops_float.h"
#include "geometry/graph_ops_double.h"
#include "geometry/graph_ops_string.h"
#include "geometry/graph_ops_boolean.h"
#include "geometry/graph_ops_pointer.h"
#include "geometry/graph_ops_complex.h"
#include "geometry/graph_ops_complex_double.h"
#include "geometry/graph_ops_vector.h"
#include "geometry/graph_ops_tensor.h"
#include "geometry/graph_ops_node.h"
#include "geometry/graph_ops_edge.h"
#include "geometry/graph_ops_stencil.h"
#include "geometry/graph_ops_genealogy.h"
#include "geometry/graph_ops_emergence.h"
#include "geometry/graph_ops_linked_list.h"
#include "geometry/graph_ops_dictionary.h"
#include "geometry/graph_ops_set.h"
#include "geometry/graph_ops_map.h"
#include "geometry/graph_ops_parallel_list.h"
#include "geometry/graph_ops_list.h"
#include "geometry/graph_ops_pointer_token.h"
#include "geometry/graph_ops_guardian.h"
#include "geometry/graph_ops_token.h"
#include "geometry/graph_ops_object_set.h"
#include "geometry/graph_ops_memory_token.h"
#include "geometry/graph_ops_token_lock.h"
#include "geometry/graph_ops_memory_map.h"
#include "geometry/graph_ops_message.h"
#include "geometry/graph_ops_custom.h"
#include "geometry/graph_ops_bitfield.h"
#include "geometry/graph_ops_bytefield.h"
#include "geometry/graph_ops_radix_encoding.h"
#include "geometry/graph_ops_pattern_palette_stream.h"
#include "geometry/graph_ops_encoding_engine.h"

/* Array of pointers to each suite; defined in graph_ops_handler.c */
extern const OperationSuite* const OperationSuites[NODE_FEATURE_IDX_COUNT];

/**
 * Retrieve the OperationSuite for the given type.
 * Returns NULL if the suite is not available or type is invalid.
 */
const OperationSuite* get_operation_suite(NodeFeatureType type);

#ifdef __cplusplus
}
#endif

#endif /* GEOMETRY_GRAPH_OPS_HANDLER_H */
