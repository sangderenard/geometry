#ifndef GEOMETRY_RELATIONAL_VECTOR_H
#define GEOMETRY_RELATIONAL_VECTOR_H

#include <stddef.h>
#include "geometry/utils.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Relational Vector API: Overview and Practical Use
 *
 * This API allows you to represent and compute paths ("vectors") through an n-dimensional, edge-weighted relational graph (manifold).
 *
 * Key Concepts:
 * - Paths are sequences of nodes, with the "distance" defined by the sum of edge weights along the path, not by any fixed geometric embedding.
 * - You can compute a minimal-weight path (vector) between any two nodes, using only the relationships (edges) relevant to the evaluation.
 * - You can project this path into a local continuous manifold (in any number of dimensions you choose) for analysis, optimization, or visualization.
 * - This approach enables local or regional analysis in highly non-Euclidean, discontinuous, or even paradoxically folded relational spaces, using only the vocabulary (edges, weights, and nodes) present in the evaluation context.
 *
 * Practical Examples:
 * 1. Neural Graphs: Find the most efficient signal path between two neurons, considering only synaptic weights, and project this path into a local coordinate system for further analysis or learning.
 * 2. Knowledge Graphs: Compute the "semantic distance" between two concepts, using only the relevant relationships, and embed this path in a local vector space for analogy or clustering tasks.
 * 3. Meshes/Manifolds: In a mesh with complex topology (e.g., with folds or non-manifold edges), compute shortest paths and locally flatten or embed them for simulation or visualization.
 * 4. Social/Network Analysis: Find the most influential or efficient connection path between two agents, and analyze the local structure without being affected by the global network geometry.
 *
 * This API primes the system for immense practical computation by:
 * - Allowing local, context-sensitive analysis and optimization in arbitrary relational spaces.
 * - Supporting projection and embedding of relational paths into continuous manifolds for downstream tasks (e.g., ML, physics, visualization).
 * - Enabling scalable, modular computation in highly complex, non-Euclidean, or discontinuous domains.
 */

typedef struct {
    Node** nodes;      // Path of nodes traversed
    size_t length;     // Number of nodes in the path
    double total_weight; // Sum of edge weights (distance)
} RelationalVectorPath;

// Compute the vector (path) between two nodes in relational space, minimizing total edge weight
double relational_vector_distance(Node* from, Node* to, RelationalVectorPath* out_path);

// Project the found path as a vector in a local continuous manifold (no-op stub)
void relational_vector_project(const RelationalVectorPath* path, double* out_vector, size_t dims);

// Free a RelationalVectorPath
void relational_vector_path_destroy(RelationalVectorPath* path);

#ifdef __cplusplus
}
#endif

#endif // GEOMETRY_RELATIONAL_VECTOR_H
