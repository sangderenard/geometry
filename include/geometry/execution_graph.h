#ifndef GEOMETRY_EXECUTION_GRAPH_H
#define GEOMETRY_EXECUTION_GRAPH_H

#include <stddef.h>
#include "geometry/types.h"

#ifdef __cplusplus
extern "C" {
#endif



void execution_graph_init(ExecutionGraph* g);
int execution_graph_add(ExecutionGraph* g, DAGNode* node);
void execution_graph_run(ExecutionGraph* g);

#ifdef __cplusplus
}
#endif

#endif // GEOMETRY_EXECUTION_GRAPH_H
