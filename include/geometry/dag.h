#ifndef GEOMETRY_DAG_H
#define GEOMETRY_DAG_H

#include <stddef.h>
#include "geometry/types.h"

DAGNode* create_dag_node(DAGForwardFn f, DAGBackwardFn b, size_t num_inputs);
void connect_dag_nodes(DAGNode* from, DAGNode* to);
void destroy_dag_node(DAGNode* node);

#endif
