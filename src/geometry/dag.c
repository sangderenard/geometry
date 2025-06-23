
#include "geometry/dag.h"
#include <stdlib.h>

DAGNode* create_dag_node(DAGForwardFn f, DAGBackwardFn b, size_t num_inputs) {
    DAGNode* node = malloc(sizeof(DAGNode));
    node->forward = f;
    node->backward = b;
    node->num_inputs = num_inputs;
    node->inputs = calloc(num_inputs, sizeof(DAGNode*));
    node->params = NULL;
    node->output = NULL;
    node->grad = NULL;
    node->next = NULL;
    return node;
}

void connect_dag_nodes(DAGNode* from, DAGNode* to) {
    for (size_t i = 0; i < to->num_inputs; ++i) {
        if (to->inputs[i] == NULL) {
            to->inputs[i] = from;
            break;
        }
    }
}

void destroy_dag_node(DAGNode* node) {
    if (!node) return;
    free(node->inputs);
    free(node->params);
    free(node);
}
