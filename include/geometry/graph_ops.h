#ifndef GEOMETRY_GRAPH_OPS_H
#define GEOMETRY_GRAPH_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include "geometry/utils.h"

struct Node;

/* Callback signatures used by the operations table */
typedef struct Node* (*NeighborFn)(struct Node* node);
typedef void (*NodeVoidFn)(struct Node* node);
typedef void (*NodeNodeFn)(struct Node* node, struct Node* other);
typedef struct Node* (*NodeReturnNodeFn)(struct Node* node);
typedef struct Node* (*NodeIndexFn)(struct Node* node, size_t index);
typedef size_t (*NodeSizeFn)(struct Node* node);

typedef void (*NodeSortFn)(struct Node* node,
                           int (*cmp)(const struct Node*, const struct Node*));
typedef struct Node* (*NodeSearchFn)(struct Node* node,
                                     int (*pred)(const struct Node*, void*),
                                     void* user);

typedef void (*NodeSliceFn)(struct Node* node, size_t start, size_t end,
                            struct Node** out);
typedef void (*NodeStencilFn)(struct Node* node, const size_t* indices,
                              size_t count, struct Node** out);

/* Generic set of operations that any container like Node, Geneology or
   SimpleGraph can provide.  The intent is that a single node can expose the
   entire structure through these functions. */
typedef struct {
    NodeNodeFn     push;      /* append child */
    NodeReturnNodeFn pop;     /* remove last child and return it */
    NodeReturnNodeFn shift;   /* remove first child and return it */
    NodeNodeFn     unshift;   /* prepend child */

    NodeIndexFn    get;       /* access by index */
    NodeSizeFn     size;      /* number of children */

    NodeSortFn     sort;      /* sort children using comparator */
    NodeSearchFn   search;    /* search using predicate */

    NeighborFn     left;      /* previous sibling */
    NeighborFn     right;     /* next sibling */
    NeighborFn     up;        /* parent */
    NeighborFn     down;      /* first child */

    NodeSliceFn    slice;     /* copy range of children */
    NodeStencilFn  stencil;   /* copy arbitrary subset */
} GraphOps;

// =====================
// Node Graph Operations
// =====================

/**
 * @brief Add a directed edge from src to dst with a given relation.
 */
void node_add_edge(Node* src, Node* dst, int relation);

/**
 * @brief Remove a directed edge from src to dst with a given relation.
 */
void node_remove_edge(Node* src, Node* dst, int relation);

/**
 * @brief Check if src is connected to dst by a given relation.
 * @return 1 if connected, 0 otherwise.
 */
int node_are_connected(const Node* src, const Node* dst, int relation);

/**
 * @brief Get the number of children of a node.
 */
size_t node_num_children(const Node* node);

/**
 * @brief Get the i-th child of a node.
 */
Node* node_get_child(const Node* node, size_t idx);

/**
 * @brief Get the parent of a node (if any, else NULL).
 */
Node* node_get_parent(const Node* node);

/**
 * @brief Get the number of siblings of a node (including itself).
 */
size_t node_num_siblings(const Node* node);

/**
 * @brief Get the i-th sibling of a node.
 */
Node* node_get_sibling(const Node* node, size_t idx);

// =====================
// Runtime Relationship Query
// =====================

/**
 * @brief Compute the relationship type between two nodes at runtime.
 *        This function traverses the graph and determines the relationship
 *        (e.g., parent, child, sibling, cousin, arbitrary, etc.)
 *        according to the current graph structure and relation types.
 *        Returns an enum or code representing the relationship, or -1 if unrelated.
 *        Extend this as needed for your taxonomy.
 */
int node_query_relationship(const Node* a, const Node* b);

// ========================
// Geneology Graph Operations
// ========================

/**
 * @brief Merge all nodes and edges from src into dest.
 */
void geneology_merge(Geneology* dest, const Geneology* src);

/**
 * @brief Find all ancestors of a node in a geneology.
 * @param out Array to fill with ancestor nodes (user allocates).
 * @param out_count Pointer to number of ancestors found.
 */
void geneology_find_ancestors(const Geneology* g, const Node* node, Node** out, size_t* out_count);

/**
 * @brief Find all descendants of a node in a geneology.
 * @param out Array to fill with descendant nodes (user allocates).
 * @param out_count Pointer to number of descendants found.
 */
void geneology_find_descendants(const Geneology* g, const Node* node, Node** out, size_t* out_count);

/**
 * @brief Extract a lineage (ancestral path) from a node up to the root.
 * @param out Array to fill with lineage nodes (user allocates, root last).
 * @param out_count Pointer to number of nodes in lineage.
 */
void geneology_extract_lineage(const Geneology* g, const Node* node, Node** out, size_t* out_count);

/**
 * @brief Extract a 2D slice: generations [gen_start,gen_end), siblings [sib_start,sib_end) at each generation.
 *        Fills out as a flat array, row-major (generation major, then sibling).
 *        Returns number of nodes found in out_count.
 */
void geneology_extract_slice_2d(const Geneology* g, const Node* root, size_t gen_start, size_t gen_end, size_t sib_start, size_t sib_end, Node** out, size_t* out_count);

/**
 * @brief Clone a subtree rooted at node (deep copy, new nodes, same structure).
 *        Returns pointer to new root node.
 */
Node* geneology_clone_subtree(const Geneology* g, const Node* node);

// =========================
// SimpleGraph Graph Operations
// =========================

/**
 * @brief Add a node and its features to the SimpleGraph.
 */
void simplegraph_add_node(SimpleGraph* graph, Node* node);

/**
 * @brief Remove a node and its features from the SimpleGraph.
 */
void simplegraph_remove_node(SimpleGraph* graph, Node* node);

/**
 * @brief Find all nodes connected by a specific edge type.
 * @param out Array to fill with nodes (user allocates).
 * @param out_count Pointer to number of nodes found.
 */
void simplegraph_find_by_edge_type(const SimpleGraph* graph, SimpleGraphEdgeType type, Node** out, size_t* out_count);

/**
 * @brief Extract a 2D slice from the SimpleGraph (see geneology_extract_slice_2d for semantics).
 */
void simplegraph_extract_slice_2d(const SimpleGraph* graph, const Node* root, size_t gen_start, size_t gen_end, size_t sib_start, size_t sib_end, Node** out, size_t* out_count);

// =====================
// DAG Graph Operations
// =====================

/**
 * @brief Create a new DAG container.
 */
Dag* dag_create(void);

/**
 * @brief Destroy a DAG container.
 */
void dag_destroy(Dag* dag);

/**
 * @brief Add a manifest to the DAG.
 */
void dag_add_manifest(Dag* dag, DagManifest* manifest);

/**
 * @brief Get the number of manifests in the DAG.
 */
size_t dag_num_manifests(const Dag* dag);

/**
 * @brief Get a manifest by index.
 */
DagManifest* dag_get_manifest(const Dag* dag, size_t idx);

/**
 * @brief Get the number of levels in a manifest.
 */
size_t dag_manifest_num_levels(const DagManifest* manifest);

/**
 * @brief Get a level by index from a manifest.
 */
DagManifestLevel* dag_manifest_get_level(const DagManifest* manifest, size_t level_idx);

// DAG GraphOps (by manifest index, depth, and node index)
void dag_push(Dag* dag, size_t manifest_idx, size_t depth, Node* node);
Node* dag_pop(Dag* dag, size_t manifest_idx, size_t depth);
Node* dag_shift(Dag* dag, size_t manifest_idx, size_t depth);
void dag_unshift(Dag* dag, size_t manifest_idx, size_t depth, Node* node);
Node* dag_get(Dag* dag, size_t manifest_idx, size_t depth, size_t idx);
size_t dag_size(Dag* dag, size_t manifest_idx, size_t depth);
void dag_sort(Dag* dag, size_t manifest_idx, size_t depth, int (*cmp)(const Node*, const Node*));
Node* dag_search(Dag* dag, size_t manifest_idx, size_t depth, int (*pred)(const Node*, void*), void* user);
Node* dag_left(Dag* dag, size_t manifest_idx, size_t depth, size_t idx);
Node* dag_right(Dag* dag, size_t manifest_idx, size_t depth, size_t idx);
Node* dag_up(Dag* dag, size_t manifest_idx, size_t depth, size_t idx);
Node* dag_down(Dag* dag, size_t manifest_idx, size_t depth, size_t idx);
void dag_slice(Dag* dag, size_t manifest_idx, size_t depth, size_t start, size_t end, Node** out);
void dag_stencil(Dag* dag, size_t manifest_idx, size_t depth, const size_t* indices, size_t count, Node** out);
void dag_contiguous(Dag* dag, size_t manifest_idx, size_t depth);

// =====================
// NeuralNetwork Graph Operations
// =====================

void neuralnetwork_add_dag(NeuralNetwork* nn, Dag* dag);
void neuralnetwork_remove_dag(NeuralNetwork* nn, Dag* dag);
Dag* neuralnetwork_get_dag(const NeuralNetwork* nn, size_t idx);
size_t neuralnetwork_num_dags(const NeuralNetwork* nn);

size_t dag_level_num_mappings(const DagManifestLevel* level);
DagManifestMapping* dag_level_get_mapping(const DagManifestLevel* level);
void dag_gather(const DagManifestMapping* mapping, void* out);
void dag_scatter(const DagManifestMapping* mapping, void* data);

extern const GraphOps NodeGraphOps;
extern const GraphOps GeneologyGraphOps;
extern const GraphOps SimpleGraphGraphOps;
extern const GraphOps DagGraphOps;
extern const GraphOps NeuralNetworkGraphOps;

#ifdef __cplusplus
}
#endif

#endif /* GEOMETRY_GRAPH_OPS_H */
