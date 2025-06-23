#ifndef GEOMETRY_UTILS_H
#define GEOMETRY_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#ifdef _WIN32
#include <windows.h>
typedef CRITICAL_SECTION node_mutex_t;
#else
#include <pthread.h>
typedef pthread_mutex_t node_mutex_t;
#endif

struct Node;

typedef void (*NodeForwardFn)(struct Node* self, void* out);
typedef void (*NodeBackwardFn)(struct Node* self, void* grad);

typedef struct {
    int type;
    NodeForwardFn forward;
    NodeBackwardFn backward;
    char* name;
    void* context;
} NodeRelation;

typedef void (*NodeProduceFn)(struct Node* self, void* product);
typedef void (*NodeReverseFn)(struct Node* self, void* product);

typedef struct {
    NodeProduceFn produce;
    NodeReverseFn reverse;
} NodeExposure;

typedef struct {
    struct Node* node;
    int relation;
} NodeLink;

typedef struct Node {
    char* id;
    unsigned long long uid;

    NodeLink* forward_links;
    NodeLink* backward_links;
    size_t num_forward_links, cap_forward_links;
    size_t num_backward_links, cap_backward_links;

    NodeRelation* relations;
    size_t num_relations, cap_relations;

    char** features;
    size_t num_features, cap_features;

    NodeExposure* exposures;
    size_t num_exposures, cap_exposures;

    node_mutex_t mutex;
} Node;

// --- Geneology structure ---
typedef struct Geneology Geneology;

Geneology* geneology_create(void);
void geneology_destroy(Geneology* g);
void geneology_add_node(Geneology* g, Node* node);
void geneology_remove_node(Geneology* g, Node* node);
size_t geneology_num_nodes(const Geneology* g);
Node* geneology_get_node(const Geneology* g, size_t idx);

// Traversal (DFS, BFS)
typedef void (*GeneologyVisitFn)(Node* node, void* user);
void geneology_traverse_dfs(Geneology* g, Node* root, GeneologyVisitFn visit, void* user);
void geneology_traverse_bfs(Geneology* g, Node* root, GeneologyVisitFn visit, void* user);

// Stubs for search/sort
void geneology_sort(Geneology* g, int (*cmp)(const Node*, const Node*));
Node* geneology_search(Geneology* g, int (*pred)(const Node*, void*), void* user);

Node* node_create(void);
void node_destroy(Node* node);

size_t node_add_relation(Node* node, int type, NodeForwardFn forward, NodeBackwardFn backward);
NodeRelation* node_get_relation(const Node* node, size_t index);

size_t node_add_feature(Node* node, const char* feature);
const char* node_get_feature(const Node* node, size_t index);

size_t node_add_exposure(Node* node, NodeProduceFn produce, NodeReverseFn reverse);
NodeExposure* node_get_exposure(const Node* node, size_t index);

size_t node_add_forward_link(Node* node, Node* link, int relation);
size_t node_add_backward_link(Node* node, Node* link, int relation);
size_t node_add_bidirectional_link(Node* a, Node* b, int relation);
const NodeLink* node_get_forward_link(const Node* node, size_t index);
const NodeLink* node_get_backward_link(const Node* node, size_t index);

typedef void (*NodeVisitFn)(Node* node, int relation, void* user);
void node_for_each_forward(Node* node, NodeVisitFn visit, void* user);
void node_for_each_backward(Node* node, NodeVisitFn visit, void* user);

void node_scatter_to_siblings(Node* node, void* data);
void node_gather_from_siblings(Node* node, void* out);
void node_scatter_to_descendants(Node* node, void* data);
void node_gather_from_ancestors(Node* node, void* out);

// --- SimpleGraph edge types ---
typedef enum {
    EDGE_PARENT_CHILD_CONTIGUOUS,
    EDGE_LINEAGE_NONCONTIGUOUS,
    EDGE_SIBLING_SIBLING_CONTIGUOUS,
    EDGE_SIBLING_SIBLING_NONCONTIGUOUS,
    EDGE_ARBITRARY
} SimpleGraphEdgeType;

typedef struct {
    Node* src;
    Node* dst;
    SimpleGraphEdgeType type;
    int relation; // index or type for the relationship
} SimpleGraphEdge;

// --- Feature hash map (simple open addressing) ---
typedef struct {
    char* key;
    void* value; // pointer to tensor (ONNX/Eigen)
} SimpleGraphFeatureEntry;

typedef struct {
    SimpleGraphFeatureEntry* entries;
    size_t num_entries, cap_entries;
} SimpleGraphFeatureMap;

// --- SimpleGraph structure ---
typedef struct {
    Geneology* geneology;
    SimpleGraphEdge* edges;
    size_t num_edges, cap_edges;

    // Contiguous feature storage (array of pointers to tensors)
    void** feature_block;
    size_t num_features, cap_features;

    // Per-node feature hash maps (indexed by node index in geneology)
    SimpleGraphFeatureMap* node_feature_maps;
    size_t num_nodes, cap_nodes;
} SimpleGraph;

SimpleGraph* simplegraph_create(Geneology* g);
void simplegraph_destroy(SimpleGraph* graph);
void simplegraph_add_edge(SimpleGraph* graph, Node* src, Node* dst, SimpleGraphEdgeType type, int relation);
void simplegraph_add_feature(SimpleGraph* graph, Node* node, const char* feature_name, void* tensor_ptr);
void* simplegraph_get_feature(SimpleGraph* graph, Node* node, const char* feature_name);

void simplegraph_forward(SimpleGraph* graph);
void simplegraph_backward(SimpleGraph* graph);

// --- DAG Manifest Structures ---

// A mapping from a set of input nodes to a set of output nodes at a given level
typedef struct {
    Node** inputs;
    size_t num_inputs;
    Node** outputs;
    size_t num_outputs;
} DagManifestMapping;

// A level in the manifest: an array of mappings (convergences/divergences)
typedef struct {
    DagManifestMapping* mappings;
    size_t num_mappings;
    int level_index; // can be negative or positive
} DagManifestLevel;

// A manifest: an array of levels (ordered by causal index)
typedef struct {
    DagManifestLevel* levels;
    size_t num_levels;
} DagManifest;

// The DAG structure: an array of manifests (multiple circuits/views)
typedef struct Dag Dag;

struct Dag {
    DagManifest* manifests;
    size_t num_manifests, cap_manifests;
};

// Manifest management
Dag* dag_create(void);
void dag_destroy(Dag* dag);
void dag_add_manifest(Dag* dag, DagManifest* manifest);
size_t dag_num_manifests(const Dag* dag);
DagManifest* dag_get_manifest(const Dag* dag, size_t idx);
size_t dag_manifest_num_levels(const DagManifest* manifest);
DagManifestLevel* dag_manifest_get_level(const DagManifest* manifest, size_t level_idx);
size_t dag_level_num_mappings(const DagManifestLevel* level);
DagManifestMapping* dag_level_get_mapping(const DagManifestLevel* level);
void dag_gather(const DagManifestMapping* mapping, void* out);
void dag_scatter(const DagManifestMapping* mapping, void* data);

// --- NeuralNetwork structure ---

// Forward/backward function signatures for a mapping step
typedef void (*NNForwardFn)(Node** inputs, size_t num_inputs, Node** outputs, size_t num_outputs, void* user);
typedef void (*NNBackwardFn)(Node** inputs, size_t num_inputs, Node** outputs, size_t num_outputs, void* user);

// A mapping step in the network (corresponds to a DagManifestMapping, but with hooks)
typedef struct {
    DagManifestMapping* mapping;
    NNForwardFn forward;
    NNBackwardFn backward;
    void* user_data; // e.g. activation/normalization params
} NeuralNetworkStep;

// A repository of available functions for mapping steps
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

// The neural network object: an array of DAGs, and steps for each DAG
#define NN_MAX_DAGS 8
#define NN_MAX_STEPS 256

typedef struct {
    Dag* dags[NN_MAX_DAGS];
    size_t num_dags;
    NeuralNetworkStep* steps[NN_MAX_DAGS][NN_MAX_STEPS];
    size_t num_steps[NN_MAX_DAGS];
    NeuralNetworkFunctionRepo function_repo;
} NeuralNetwork;

// Neural network management
NeuralNetwork* neuralnetwork_create(void);
void neuralnetwork_destroy(NeuralNetwork* nn);

// Register a function in the repo
void neuralnetwork_register_function(NeuralNetwork* nn, const char* name, NNForwardFn forward, NNBackwardFn backward);

// Attach a function to a step
void neuralnetwork_set_step_function(NeuralNetwork* nn, size_t dag_idx, size_t step_idx, const char* function_name, void* user_data);

// Forward/backward wrappers
void neuralnetwork_forward(NeuralNetwork* nn);
void neuralnetwork_backward(NeuralNetwork* nn);

// Step-level forward/backward
void neuralnetwork_forwardstep(NeuralNetwork* nn, size_t dag_idx, size_t step_idx);
void neuralnetwork_backwardstep(NeuralNetwork* nn, size_t dag_idx, size_t step_idx);

/*
Diagram:

NeuralNetwork
  |-- dags[]: Dag
  |-- steps[dag][step]: NeuralNetworkStep
  |-- function_repo: {name, forward, backward}

Each NeuralNetworkStep:
  - mapping: DagManifestMapping (inputs/outputs)
  - forward/backward: function pointers (activation, normalization, etc)
  - user_data: params for the function

Execution:
  - neuralnetwork_forward: for each dag, for each step, call step->forward(inputs, outputs, user_data)
  - neuralnetwork_backward: for each dag, for each step (reverse), call step->backward(...)

Data:
  - Node features (Eigen/ONNX tensors) are used for all data movement and computation
*/

#ifdef __cplusplus
}
#endif

#endif // GEOMETRY_UTILS_H
