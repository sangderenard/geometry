#include "geometry/utils.h"
#include <stdlib.h>
#include <string.h>

static void* grow_array(void* array, size_t elem_size, size_t* cap) {
    size_t new_cap = (*cap == 0) ? 4 : (*cap * 2);
    void* new_arr = realloc(array, new_cap * elem_size);
    if (new_arr) *cap = new_cap;
    return new_arr;
}

Node* node_create(void) {
    Node* n = (Node*)calloc(1, sizeof(Node));
    return n;
}

void node_destroy(Node* node) {
    if (!node) return;
    free(node->relations);
    if (node->features) {
        for (size_t i = 0; i < node->num_features; ++i) {
            free(node->features[i]);
        }
    }
    free(node->features);
    free(node->exposures);
    free(node->forward_links);
    free(node->backward_links);
    free(node);
}

size_t node_add_relation(Node* node, int type, NodeForwardFn forward, NodeBackwardFn backward) {
    if (node->num_relations == node->cap_relations) {
        void* tmp = grow_array(node->relations, sizeof(NodeRelation), &node->cap_relations);
        if (!tmp) return (size_t)-1;
        node->relations = (NodeRelation*)tmp;
    }
    NodeRelation r = {type, forward, backward};
    node->relations[node->num_relations] = r;
    return node->num_relations++;
}

NodeRelation* node_get_relation(const Node* node, size_t index) {
    if (!node || index >= node->num_relations) return NULL;
    return &node->relations[index];
}

size_t node_add_feature(Node* node, const char* feature) {
    if (node->num_features == node->cap_features) {
        void* tmp = grow_array(node->features, sizeof(char*), &node->cap_features);
        if (!tmp) return (size_t)-1;
        node->features = (char**)tmp;
    }
    node->features[node->num_features] = strdup(feature);
    return node->num_features++;
}

const char* node_get_feature(const Node* node, size_t index) {
    if (!node || index >= node->num_features) return NULL;
    return node->features[index];
}

size_t node_add_exposure(Node* node, NodeProduceFn produce, NodeReverseFn reverse) {
    if (node->num_exposures == node->cap_exposures) {
        void* tmp = grow_array(node->exposures, sizeof(NodeExposure), &node->cap_exposures);
        if (!tmp) return (size_t)-1;
        node->exposures = (NodeExposure*)tmp;
    }
    NodeExposure e = {produce, reverse};
    node->exposures[node->num_exposures] = e;
    return node->num_exposures++;
}

NodeExposure* node_get_exposure(const Node* node, size_t index) {
    if (!node || index >= node->num_exposures) return NULL;
    return &node->exposures[index];
}

size_t node_add_forward_link(Node* node, Node* link, int relation) {
    if (node->num_forward_links == node->cap_forward_links) {
        void* tmp = grow_array(node->forward_links, sizeof(NodeLink), &node->cap_forward_links);
        if (!tmp) return (size_t)-1;
        node->forward_links = (NodeLink*)tmp;
    }
    NodeLink l = {link, relation};
    node->forward_links[node->num_forward_links] = l;
    return node->num_forward_links++;
}

size_t node_add_backward_link(Node* node, Node* link, int relation) {
    if (node->num_backward_links == node->cap_backward_links) {
        void* tmp = grow_array(node->backward_links, sizeof(NodeLink), &node->cap_backward_links);
        if (!tmp) return (size_t)-1;
        node->backward_links = (NodeLink*)tmp;
    }
    NodeLink l = {link, relation};
    node->backward_links[node->num_backward_links] = l;
    return node->num_backward_links++;
}

const NodeLink* node_get_forward_link(const Node* node, size_t index) {
    if (!node || index >= node->num_forward_links) return NULL;
    return &node->forward_links[index];
}

const NodeLink* node_get_backward_link(const Node* node, size_t index) {
    if (!node || index >= node->num_backward_links) return NULL;
    return &node->backward_links[index];
}

size_t node_add_bidirectional_link(Node* a, Node* b, int relation) {
    if (!a || !b) return (size_t)-1;
    size_t idx1 = node_add_forward_link(a, b, relation);
    size_t idx2 = node_add_backward_link(b, a, relation);
    if (idx1 == (size_t)-1 || idx2 == (size_t)-1) return (size_t)-1;
    return idx1;
}

void node_for_each_forward(Node* node, NodeVisitFn visit, void* user) {
    if (!node || !visit) return;
    for (size_t i = 0; i < node->num_forward_links; ++i) {
        visit(node->forward_links[i].node, node->forward_links[i].relation, user);
    }
}

void node_for_each_backward(Node* node, NodeVisitFn visit, void* user) {
    if (!node || !visit) return;
    for (size_t i = 0; i < node->num_backward_links; ++i) {
        visit(node->backward_links[i].node, node->backward_links[i].relation, user);
    }
}

void node_scatter_to_siblings(Node* node, void* data) {
    if (!node) return;
    for (size_t i = 0; i < node->num_forward_links; ++i) {
        NodeLink* link = &node->forward_links[i];
        if (link->relation >= 0 && (size_t)link->relation < node->num_relations) {
            NodeRelation* rel = &node->relations[link->relation];
            if (rel->forward)
                rel->forward(link->node, data);
        }
    }
}

void node_gather_from_siblings(Node* node, void* out) {
    if (!node) return;
    for (size_t i = 0; i < node->num_backward_links; ++i) {
        NodeLink* link = &node->backward_links[i];
        if (link->relation >= 0 && (size_t)link->relation < node->num_relations) {
            NodeRelation* rel = &node->relations[link->relation];
            if (rel->backward)
                rel->backward(link->node, out);
        }
    }
}

void node_scatter_to_descendants(Node* node, void* data) {
    if (!node) return;
    node_scatter_to_siblings(node, data);
    for (size_t i = 0; i < node->num_forward_links; ++i) {
        node_scatter_to_descendants(node->forward_links[i].node, data);
    }
}

void node_gather_from_ancestors(Node* node, void* out) {
    if (!node) return;
    node_gather_from_siblings(node, out);
    for (size_t i = 0; i < node->num_backward_links; ++i) {
        node_gather_from_ancestors(node->backward_links[i].node, out);
    }
}

