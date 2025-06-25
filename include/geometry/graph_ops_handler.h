#ifndef GEOMETRY_GRAPH_OPS_HANDLER_H
#define GEOMETRY_GRAPH_OPS_HANDLER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "geometry/graph_ops.h"

/* Error codes for GraphOpsHandler */
#define GRAPH_OPS_OK 0
#define GRAPH_OPS_ERR -1

/*
 * GraphOpsHandler provides a thin indirection layer over a GraphOps table.
 * It allows containers to register their operation implementations and
 * clients to call the standard GraphOps interface with built-in null
 * checks. If a particular function pointer is not provided, the wrapper
 * returns GRAPH_OPS_ERR (or NULL for pointer-returning operations).
 */
typedef struct {
    const GraphOps* ops;
} GraphOpsHandler;

static inline int graph_ops_handler_init(GraphOpsHandler* h, const GraphOps* ops) {
    if (!h) return GRAPH_OPS_ERR;
    h->ops = ops;
    return GRAPH_OPS_OK;
}

static inline int graph_ops_push(const GraphOpsHandler* h, Node* n, Node* child) {
    if (!h || !h->ops || !h->ops->push) return GRAPH_OPS_ERR;
    h->ops->push(n, child);
    return GRAPH_OPS_OK;
}

static inline Node* graph_ops_pop(const GraphOpsHandler* h, Node* n) {
    return (!h || !h->ops || !h->ops->pop) ? NULL : h->ops->pop(n);
}

static inline Node* graph_ops_shift(const GraphOpsHandler* h, Node* n) {
    return (!h || !h->ops || !h->ops->shift) ? NULL : h->ops->shift(n);
}

static inline int graph_ops_unshift(const GraphOpsHandler* h, Node* n, Node* child) {
    if (!h || !h->ops || !h->ops->unshift) return GRAPH_OPS_ERR;
    h->ops->unshift(n, child);
    return GRAPH_OPS_OK;
}

static inline Node* graph_ops_get(const GraphOpsHandler* h, Node* n, size_t idx) {
    return (!h || !h->ops || !h->ops->get) ? NULL : h->ops->get(n, idx);
}

static inline size_t graph_ops_size(const GraphOpsHandler* h, Node* n) {
    return (!h || !h->ops || !h->ops->size) ? 0 : h->ops->size(n);
}

static inline int graph_ops_sort(const GraphOpsHandler* h, Node* n,
                                 int (*cmp)(const Node*, const Node*)) {
    if (!h || !h->ops || !h->ops->sort) return GRAPH_OPS_ERR;
    h->ops->sort(n, cmp);
    return GRAPH_OPS_OK;
}

static inline Node* graph_ops_search(const GraphOpsHandler* h, Node* n,
                                     int (*pred)(const Node*, void*), void* user) {
    return (!h || !h->ops || !h->ops->search) ? NULL : h->ops->search(n, pred, user);
}

static inline Node* graph_ops_left(const GraphOpsHandler* h, Node* n) {
    return (!h || !h->ops || !h->ops->left) ? NULL : h->ops->left(n);
}

static inline Node* graph_ops_right(const GraphOpsHandler* h, Node* n) {
    return (!h || !h->ops || !h->ops->right) ? NULL : h->ops->right(n);
}

static inline Node* graph_ops_up(const GraphOpsHandler* h, Node* n) {
    return (!h || !h->ops || !h->ops->up) ? NULL : h->ops->up(n);
}

static inline Node* graph_ops_down(const GraphOpsHandler* h, Node* n) {
    return (!h || !h->ops || !h->ops->down) ? NULL : h->ops->down(n);
}

static inline int graph_ops_slice(const GraphOpsHandler* h, Node* n,
                                  size_t start, size_t end, Node** out) {
    if (!h || !h->ops || !h->ops->slice) return GRAPH_OPS_ERR;
    h->ops->slice(n, start, end, out);
    return GRAPH_OPS_OK;
}

static inline int graph_ops_stencil(const GraphOpsHandler* h, Node* n,
                                    const size_t* indices, size_t count, Node** out) {
    if (!h || !h->ops || !h->ops->stencil) return GRAPH_OPS_ERR;
    h->ops->stencil(n, indices, count, out);
    return GRAPH_OPS_OK;
}

#ifdef __cplusplus
}
#endif

#endif /* GEOMETRY_GRAPH_OPS_HANDLER_H */
