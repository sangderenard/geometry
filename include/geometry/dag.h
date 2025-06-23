
#ifndef GEOMETRY_DAG_H
#define GEOMETRY_DAG_H

#include <stddef.h>

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

DAGNode* create_dag_node(DAGForwardFn f, DAGBackwardFn b, size_t num_inputs);
void connect_dag_nodes(DAGNode* from, DAGNode* to);
void destroy_dag_node(DAGNode* node);

#endif
