#ifndef GEOMETRY_DAG_H
#define GEOMETRY_DAG_H

#include <stddef.h>

// Forward declaration to avoid circular dependency
struct Node;

typedef struct DAGNode DAGNode;
typedef struct DAGEdge DAGEdge;

typedef void (*DAGForwardFn)(DAGNode* self);
typedef void (*DAGBackwardFn)(DAGNode* self);

typedef struct {
    void* data;
} DAGParams;

struct DAGNode {
    DAGForwardFn forward;
    DAGBackwardFn backward;
    DAGNode** inputs;
    size_t num_inputs;
    DAGParams* params;
    void* output;
    void* grad;
    DAGNode* next;
};

// --- DAG Manifest Structures ---
typedef struct {
    struct Node** inputs;
    size_t num_inputs;
    struct Node** outputs;
    size_t num_outputs;
} DagManifestMapping;

typedef struct {
    DagManifestMapping* mappings;
    size_t num_mappings;
    int level_index;
} DagManifestLevel;

typedef struct {
    DagManifestLevel* levels;
    size_t num_levels;
} DagManifest;

typedef struct Dag {
    DagManifest* manifests;
    size_t num_manifests, cap_manifests;
} Dag;

// --- NeuralNetwork structure ---
typedef struct NeuralNetwork NeuralNetwork;

typedef void (*NNForwardFn)(struct Node** inputs, size_t num_inputs, struct Node** outputs, size_t num_outputs, void* user);
typedef void (*NNBackwardFn)(struct Node** inputs, size_t num_inputs, struct Node** outputs, size_t num_outputs, void* user);

typedef struct {
    DagManifestMapping* mapping;
    NNForwardFn forward;
    NNBackwardFn backward;
    void* user_data;
} NeuralNetworkStep;

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

#define NN_MAX_DAGS 8
#define NN_MAX_STEPS 256

typedef struct NeuralNetwork {
    Dag* dags[NN_MAX_DAGS];
    size_t num_dags;
    NeuralNetworkStep* steps[NN_MAX_DAGS][NN_MAX_STEPS];
    size_t num_steps[NN_MAX_DAGS];
    NeuralNetworkFunctionRepo function_repo;
} NeuralNetwork;

DAGNode* create_dag_node(DAGForwardFn f, DAGBackwardFn b, size_t num_inputs);
void connect_dag_nodes(DAGNode* from, DAGNode* to);
void destroy_dag_node(DAGNode* node);

#endif
