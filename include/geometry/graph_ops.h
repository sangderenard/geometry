#ifndef GEOMETRY_GRAPH_OPS_H
#define GEOMETRY_GRAPH_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

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

#ifdef __cplusplus
}
#endif

#endif /* GEOMETRY_GRAPH_OPS_H */
