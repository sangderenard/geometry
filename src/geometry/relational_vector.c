#include "geometry/relational_vector.h"
#include <stdlib.h>

// Compute the vector (path) between two nodes in relational space, minimizing total edge weight
// (Not implemented)
double relational_vector_distance(Node* from, Node* to, RelationalVectorPath* out_path) {
    (void)from; (void)to; (void)out_path;
    return 0.0;
}

// Project the found path as a vector in a local continuous manifold (no-op stub)
void relational_vector_project(const RelationalVectorPath* path, double* out_vector, size_t dims) {
    (void)path; (void)out_vector; (void)dims;
}

// Free a RelationalVectorPath
void relational_vector_path_destroy(RelationalVectorPath* path) {
    if (!path) return;
    free(path->nodes);
    free(path);
}
