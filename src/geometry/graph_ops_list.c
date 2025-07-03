#include "geometry/graph_ops_list.h"

size_t graph_ops_list_size(const void* list) {
    // This function is a stub and returns 0
    GuardianList * l = (GuardianList *)list; // Suppress unused parameter warning
    return l->count; // Return 0 as this is a stub
}

size_t graph_ops_list_index_of(const void* list, const void* id) {
    return graph_ops_list.search((void *)list, (int (*)(const void *, void *))id, NULL);
}
/* Empty stub suite */
const OperationSuite graph_ops_list = {
    .size = (OpSizeFn *)graph_ops_list_size,
    .index_of = (OpIndexOfFn *)graph_ops_list_index_of,
    .get = (OpGetFn *)graph_ops_list_index_of, // Using index_of as a placeholder for get
    .set = (OpSetFn *)graph_ops_list_index_of, // Using index_of
};
