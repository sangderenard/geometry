#ifndef GEOMETRY_UTILS_H
#define GEOMETRY_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

struct Node;

typedef void (*NodeForwardFn)(struct Node* self, void* out);
typedef void (*NodeBackwardFn)(struct Node* self, const void* grad);

typedef struct {
    int type;
    NodeForwardFn forward;
    NodeBackwardFn backward;
} NodeRelation;

typedef void (*NodeProduceFn)(struct Node* self, void* product);
typedef void (*NodeReverseFn)(struct Node* self, const void* product);

typedef struct {
    NodeProduceFn produce;
    NodeReverseFn reverse;
} NodeExposure;

typedef struct {
    struct Node* node;
    int relation;
} NodeLink;

typedef struct Node {
    NodeRelation* relations;
    size_t num_relations;
    size_t cap_relations;

    char** features;
    size_t num_features;
    size_t cap_features;

    NodeExposure* exposures;
    size_t num_exposures;
    size_t cap_exposures;

    NodeLink* forward_links;
    size_t num_forward_links;
    size_t cap_forward_links;
    NodeLink* backward_links;
    size_t num_backward_links;
    size_t cap_backward_links;
} Node;

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

#ifdef __cplusplus
}
#endif

#endif // GEOMETRY_UTILS_H
