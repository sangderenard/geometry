#include "geometry/graph_ops_dictionary.h"
#include "geometry/graph_ops_list.h"  // for graph_ops_list

void *graph_ops_dictionary_set(void *container, void* id, void *element) {
    // This function is a stub and does nothing
    GuardianDict* dictionary = (GuardianDict *)container; // Suppress unused parameter warning
    GuardianList* keys = dictionary->keys; // Suppress unused parameter warning
    GuardianList* values = dictionary->values; // Suppress unused parameter warning
    size_t index =  graph_ops_list.index_of(keys, id);
    (char *)index;     // Suppress unused parameter warning
    (void *)element;   // Suppress unused parameter warning
}

void *graph_ops_dictionary_get(void *container, size_t index) {
    // This function is a stub and returns NULL
    (void)container; // Suppress unused parameter warning
    (void)index;     // Suppress unused parameter warning
    return NULL;     // Return NULL as this is a stub
}

/* Empty stub suite */
const OperationSuite graph_ops_dictionary = {
    .set = (OpSetFn *)graph_ops_dictionary_set,
    .get = (OpGetFn *)graph_ops_dictionary_get,
    
};
