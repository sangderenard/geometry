#include "geometry/execution_graph.h"

void execution_graph_init(ExecutionGraph* g) {
    if (!g) return;
    g->num_nodes = 0;
}

int execution_graph_add(ExecutionGraph* g, DAGNode* node) {
    if (!g || !node) return 0;
    if (g->num_nodes >= EXEC_GRAPH_MAX_NODES) return 0;
    g->nodes[g->num_nodes++] = node;
    return 1;
}

void execution_graph_run(ExecutionGraph* g) {
    if (!g) return;
    for (size_t i = 0; i < g->num_nodes; ++i) {
        if (g->nodes[i] && g->nodes[i]->forward)
            g->nodes[i]->forward(g->nodes[i]);
    }
}
