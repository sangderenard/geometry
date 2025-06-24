// =====================
// CONCURRENCY AND DEREFERENCING POLICY
// =====================
/*
All access to dynamic primitives (e.g., arrays, neighbors, links) must be performed via concurrency-checked guardian APIs.
Direct pointer dereferencing for dynamic primitives is not allowed. Traversal, neighbor access, and dereferencing must use guardian APIs that return tokens or concurrency-checked access, not raw pointers.
*/

#include "geometry/utils.h"
#include "geometry/dag.h"
#include "geometry/guardian.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif
#include <stdio.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// === Utility ===
static void* grow_array(TokenGuardian* guardian, void* array, size_t elem_size, size_t* cap) {
    size_t new_cap = (*cap == 0) ? 4 : (*cap * 2);
    void* new_arr = guardian_realloc(guardian, array, new_cap * elem_size);
    if (new_arr) *cap = new_cap;
    return new_arr;
}

// Helper to add a node to a multidimensional array (e.g., parents, children, siblings)
static void node_add_to_stencil(TokenGuardian* guardian, unsigned long* arr_token, unsigned long* num_arr_token, size_t* num_dims, size_t dim, Node* target) {
    // Grow dimensions if needed
    if (dim >= *num_dims) {
        size_t old_dims = *num_dims;
        size_t new_dims = dim + 1;
        unsigned long tmp_arr_token, tmp_num_token;
        guardian_list_resize(guardian, *arr_token, new_dims, &tmp_arr_token);
        guardian_list_resize(guardian, *num_arr_token, new_dims, &tmp_num_token);
        if (!tmp_arr_token || !tmp_num_token) {
            guardian_list_destroy(guardian, tmp_arr_token);
            guardian_list_destroy(guardian, tmp_num_token);
            return;
        }
        *arr_token = tmp_arr_token;
        *num_arr_token = tmp_num_token;
        for (size_t d = old_dims; d < new_dims; ++d) {
            guardian_list_set(guardian, *arr_token, d, NULL);
            guardian_list_set(guardian, *num_arr_token, d, 0);
        }
        *num_dims = new_dims;
    }
    // Grow array in this dimension
    size_t idx;
    guardian_list_get(guardian, *num_arr_token, dim, &idx);
    unsigned long tmp_dim_token;
    guardian_list_resize(guardian, *arr_token, idx + 1, &tmp_dim_token);
    if (!tmp_dim_token) return;
    guardian_list_set(guardian, *arr_token, dim, tmp_dim_token);
    guardian_list_set(guardian, tmp_dim_token, idx, target);
    guardian_list_set(guardian, *num_arr_token, dim, idx + 1);
}

// === Geneology Basic Operations ===

Geneology* geneology_create(TokenGuardian* guardian) {
    if (!guardian) return NULL;
    unsigned long geneology_token, nodes_token, dict_token;
    Geneology* g = (Geneology*)guardian_alloc(guardian, sizeof(Geneology), &geneology_token);
    if (!g) return NULL;
    // Create an empty list for nodes
    g->nodes_token = guardian_list_create(guardian, &nodes_token);
    g->num_nodes = 0;
    // Create an empty hash map for path hash dict
    g->path_hash_dict_token = guardian_dict_create(guardian, &dict_token);
    return g;
}

void geneology_destroy(TokenGuardian* guardian, Geneology* g) {
    if (!guardian || !g) return;
    guardian_list_destroy(guardian, g->nodes_token);
    guardian_dict_destroy(guardian, g->path_hash_dict_token);
    guardian_free(guardian, g);
}

void geneology_add_node(TokenGuardian* guardian, Geneology* g, Node* node) {
    if (!guardian || !g || !node) return;
    unsigned long node_token;
    guardian_list_append(guardian, g->nodes_token, node, &node_token);
    g->num_nodes++;
}

void geneology_remove_node(TokenGuardian* guardian, Geneology* g, Node* node) {
    if (!guardian || !g || !node) return;
    guardian_list_remove(guardian, g->nodes_token, node);
    g->num_nodes--;
}

size_t geneology_num_nodes(const Geneology* g) {
    return g ? g->num_nodes : 0;
}

Node* geneology_get_node(const Geneology* g, size_t idx) {
    if (!g) return NULL;
    unsigned long node_token;
    if (!guardian_list_get(NULL, g->nodes_token, idx, &node_token)) return NULL;
    return guardian_deref_node(NULL, node_token);
}

// === Node Data Structures ===

// All type definitions are provided in the public header.

// Traversal (DFS, BFS)
#include <stdbool.h>
void geneology_traverse_dfs(TokenGuardian* guardian, Geneology* g, Node* root, GeneologyVisitFn visit, void* user) {
    if (!guardian || !g || !root || !visit) return;
    bool* visited = guardian_calloc(guardian, g->num_nodes, sizeof(bool));
    size_t stack_cap = 16, stack_size = 0;
    Node** stack = guardian_malloc(guardian, stack_cap * sizeof(Node*));
    stack[stack_size++] = root;
    while (stack_size) {
        Node* n = stack[--stack_size];
        size_t idx = 0;
        for (; idx < g->num_nodes; ++idx) if (geneology_get_node(g, idx) == n) break;
        if (idx == g->num_nodes || visited[idx]) continue;
        visited[idx] = true;
        visit(n, user);
        // Use Guardian API to get number of forward links
        size_t num_links = guardian_list_size(guardian, n->forward_links_token);
        for (size_t i = 0; i < num_links; ++i) {
            unsigned long link_token;
            if (!guardian_list_get(guardian, n->forward_links_token, i, &link_token)) continue;
            NodeLink* link = guardian_deref_link(guardian, link_token);
            if (stack_size == stack_cap) {
                size_t new_cap = stack_cap * 2;
                Node** tmp = guardian_realloc(guardian, stack, new_cap * sizeof(Node*));
                if (!tmp) break;
                stack = tmp;
                stack_cap = new_cap;
            }
            stack[stack_size++] = link->node;
        }
    }
    guardian_free(guardian, stack);
    guardian_free(guardian, visited);
}

void geneology_traverse_bfs(TokenGuardian* guardian, Geneology* g, Node* root, GeneologyVisitFn visit, void* user) {
    if (!guardian || !g || !root || !visit) return;
    bool* visited = guardian_calloc(guardian, g->num_nodes, sizeof(bool));
    size_t queue_cap = 16, queue_size = 0, queue_head = 0;
    Node** queue = guardian_malloc(guardian, queue_cap * sizeof(Node*));
    queue[queue_size++] = root;
    while (queue_head < queue_size) {
        Node* n = queue[queue_head++];
        size_t idx = 0;
        for (; idx < g->num_nodes; ++idx) if (geneology_get_node(g, idx) == n) break;
        if (idx == g->num_nodes || visited[idx]) continue;
        visited[idx] = true;
        visit(n, user);
        size_t num_links = guardian_list_size(guardian, n->forward_links_token);
        for (size_t i = 0; i < num_links; ++i) {
            unsigned long link_token;
            if (!guardian_list_get(guardian, n->forward_links_token, i, &link_token)) continue;
            NodeLink* link = guardian_deref_link(guardian, link_token);
            if (queue_size == queue_cap) {
                size_t new_cap = queue_cap * 2;
                Node** tmp = guardian_realloc(guardian, queue, new_cap * sizeof(Node*));
                if (!tmp) break;
                queue = tmp;
                queue_cap = new_cap;
            }
            queue[queue_size++] = link->node;
        }
    }
    guardian_free(guardian, queue);
    guardian_free(guardian, visited);
}

// Stubs for search/sort
typedef int (*GeneologyNodeCmp)(const Node*, const Node*);
void geneology_sort(TokenGuardian* guardian, Geneology* g, GeneologyNodeCmp cmp) {
    if (!guardian || !g || !cmp || g->num_nodes < 2) return;
    for (size_t i = 0; i < g->num_nodes - 1; ++i) {
        for (size_t j = i + 1; j < g->num_nodes; ++j) {
            Node* node_i = geneology_get_node(g, i);
            Node* node_j = geneology_get_node(g, j);
            if (cmp(node_i, node_j) > 0) {
                unsigned long token_i, token_j;
                guardian_list_get(guardian, g->nodes_token, i, &token_i);
                guardian_list_get(guardian, g->nodes_token, j, &token_j);
                guardian_list_set(guardian, g->nodes_token, i, guardian_deref_node(guardian, token_j));
                guardian_list_set(guardian, g->nodes_token, j, guardian_deref_node(guardian, token_i));
            }
        }
    }
}

Node* geneology_search(TokenGuardian* guardian, Geneology* g, int (*pred)(const Node*, void*), void* user) {
    if (!guardian || !g || !pred) return NULL;
    for (size_t i = 0; i < g->num_nodes; ++i) {
        Node* node = geneology_get_node(g, i);
        if (pred(node, user)) return node;
    }
    return NULL;
}

// === Node Lifecycle ===

Node* node_create(TokenGuardian* guardian) {
    if (!guardian) return NULL;
    unsigned long node_token;
    Node* n = (Node*)guardian_alloc(guardian, sizeof(Node), &node_token);
    if (!n) return NULL;
    static uint64_t counter = 1;
    n->uid = counter++;
#ifdef _WIN32
    InitializeCriticalSection(&n->mutex);
#else
    pthread_mutex_init(&n->mutex, NULL);
#endif
    // Initialize multidimensional stencil arrays to NULL/0
    n->parents_token = guardian_list_create(guardian, NULL);
    n->num_parents_token = guardian_list_create(guardian, NULL);
    n->num_dims_parents = 0;
    n->children_token = guardian_list_create(guardian, NULL);
    n->num_children_token = guardian_list_create(guardian, NULL);
    n->num_dims_children = 0;
    n->left_siblings_token = guardian_list_create(guardian, NULL);
    n->num_left_siblings_token = guardian_list_create(guardian, NULL);
    n->num_dims_left_siblings = 0;
    n->right_siblings_token = guardian_list_create(guardian, NULL);
    n->num_right_siblings_token = guardian_list_create(guardian, NULL);
    n->num_dims_right_siblings = 0;
    n->forward_links_token = guardian_list_create(guardian, NULL);
    n->backward_links_token = guardian_list_create(guardian, NULL);
    return n;
}

void node_destroy(TokenGuardian* guardian, Node* node) {
    if (!guardian || !node) return;
#ifdef _WIN32
    DeleteCriticalSection(&node->mutex);
#else
    pthread_mutex_destroy(&node->mutex);
#endif
    guardian_free(guardian, node->id);
    guardian_free(guardian, node->relations);
    if (node->features) {
        for (size_t i = 0; i < node->num_features; ++i) {
            guardian_free(guardian, node->features[i]);
        }
    }
    guardian_free(guardian, node->features);
    guardian_free(guardian, node->exposures);
    guardian_list_destroy(guardian, node->forward_links_token);
    guardian_list_destroy(guardian, node->backward_links_token);
    // Free multidimensional stencil arrays
    guardian_list_destroy(guardian, node->parents_token);
    guardian_list_destroy(guardian, node->num_parents_token);
    guardian_list_destroy(guardian, node->children_token);
    guardian_list_destroy(guardian, node->num_children_token);
    guardian_list_destroy(guardian, node->left_siblings_token);
    guardian_list_destroy(guardian, node->num_left_siblings_token);
    guardian_list_destroy(guardian, node->right_siblings_token);
    guardian_list_destroy(guardian, node->num_right_siblings_token);
    guardian_free(guardian, node);
}

// === Node API ===

static size_t node_add_relation_full(TokenGuardian* guardian, Node* node, int type, NodeForwardFn forward, NodeBackwardFn backward, const char* name, void* context) {
    if (!guardian || guardian_list_size(guardian, node->relations_token) == node->cap_relations) {
        void* tmp = grow_array(guardian, node->relations, sizeof(NodeRelation), &node->cap_relations);
        if (!tmp) return (size_t)-1;
        node->relations = (NodeRelation*)tmp;
    }
    NodeRelation r = {type, forward, backward, strdup(name), context};
    size_t idx = guardian_list_size(guardian, node->relations_token);
    guardian_list_append(guardian, node->relations_token, &r, NULL);
    node->num_relations++;

    // Default to 0th dimension for now
    size_t dim = 0;
    Node* partner = context ? (Node*)context : NULL;
    switch (type) {
        case EDGE_PARENT_CHILD_CONTIGUOUS:
            if (partner) node_add_to_stencil(guardian, &node->children_token, &node->num_children_token, &node->num_dims_children, dim, partner);
            break;
        case EDGE_CHILD_PARENT_CONTIGUOUS:
            if (partner) node_add_to_stencil(guardian, &node->parents_token, &node->num_parents_token, &node->num_dims_parents, dim, partner);
            break;
        case EDGE_SIBLING_LEFT_TO_RIGHT_CONTIGUOUS:
            if (partner) node_add_to_stencil(guardian, &node->right_siblings_token, &node->num_right_siblings_token, &node->num_dims_right_siblings, dim, partner);
            break;
        case EDGE_SIBLING_RIGHT_TO_LEFT_CONTIGUOUS:
            if (partner) node_add_to_stencil(guardian, &node->left_siblings_token, &node->num_left_siblings_token, &node->num_dims_left_siblings, dim, partner);
            break;
        case EDGE_SIBLING_SIBLING_NONCONTIGUOUS:
            if (partner) {
                node_add_to_stencil(guardian, &node->left_siblings_token, &node->num_left_siblings_token, &node->num_dims_left_siblings, dim, partner);
                node_add_to_stencil(guardian, &node->right_siblings_token, &node->num_right_siblings_token, &node->num_dims_right_siblings, dim, partner);
            }
            break;
        case EDGE_LINEAGE_NONCONTIGUOUS:
            if (partner) {
                node_add_to_stencil(guardian, &node->parents_token, &node->num_parents_token, &node->num_dims_parents, dim, partner);
                node_add_to_stencil(guardian, &node->children_token, &node->num_children_token, &node->num_dims_children, dim, partner);
            }
            break;
        case EDGE_ARBITRARY:
        default:
            break;
    }
    return idx;
}

size_t node_add_relation(TokenGuardian* guardian, Node* node, int type, NodeForwardFn forward, NodeBackwardFn backward) {
    return node_add_relation_full(guardian, node, type, forward, backward, "", NULL);
}

size_t node_add_feature(TokenGuardian* guardian, Node* node, const char* feature) {
    if (!guardian || guardian_list_size(guardian, node->features_token) == node->cap_features) {
        void* tmp = grow_array(guardian, node->features, sizeof(char*), &node->cap_features);
        if (!tmp) return (size_t)-1;
        node->features = (char**)tmp;
    }
    guardian_list_append(guardian, node->features_token, strdup(feature), NULL);
    return guardian_list_size(guardian, node->features_token) - 1;
}

size_t node_add_exposure(TokenGuardian* guardian, Node* node, NodeProduceFn produce, NodeReverseFn reverse) {
    if (!guardian || guardian_list_size(guardian, node->exposures_token) == node->cap_exposures) {
        void* tmp = grow_array(guardian, node->exposures, sizeof(NodeExposure), &node->cap_exposures);
        if (!tmp) return (size_t)-1;
        node->exposures = (NodeExposure*)tmp;
    }
    NodeExposure e = {produce, reverse};
    guardian_list_append(guardian, node->exposures_token, &e, NULL);
    return guardian_list_size(guardian, node->exposures_token) - 1;
}

// === DAG Linkage ===

size_t node_add_forward_link(TokenGuardian* guardian, Node* node, Node* link, int relation) {
    if (!guardian || guardian_list_size(guardian, node->forward_links_token) == node->cap_forward_links) {
        void* tmp = grow_array(guardian, node->forward_links, sizeof(NodeLink), &node->cap_forward_links);
        if (!tmp) return (size_t)-1;
        node->forward_links = (NodeLink*)tmp;
    }
    NodeLink l = {link, relation};
    guardian_list_append(guardian, node->forward_links_token, &l, NULL);
    return guardian_list_size(guardian, node->forward_links_token) - 1;
}

size_t node_add_backward_link(TokenGuardian* guardian, Node* node, Node* link, int relation) {
    if (!guardian || guardian_list_size(guardian, node->backward_links_token) == node->cap_backward_links) {
        void* tmp = grow_array(guardian, node->backward_links, sizeof(NodeLink), &node->cap_backward_links);
        if (!tmp) return (size_t)-1;
        node->backward_links = (NodeLink*)tmp;
    }
    NodeLink l = {link, relation};
    guardian_list_append(guardian, node->backward_links_token, &l, NULL);
    return guardian_list_size(guardian, node->backward_links_token) - 1;
}

size_t node_add_bidirectional_link(TokenGuardian* guardian, Node* a, Node* b, int relation) {
    if (!guardian || !a || !b) return (size_t)-1;
    // Add forward link for a -> b
    size_t idx1 = node_add_forward_link(guardian, a, b, relation);
    // Add to a's stencil (children, right_siblings, etc.)
    switch (relation) {
        case EDGE_PARENT_CHILD_CONTIGUOUS:
            node_add_to_stencil(guardian, &a->children_token, &a->num_children_token, &a->num_dims_children, 0, b);
            break;
        case EDGE_SIBLING_LEFT_TO_RIGHT_CONTIGUOUS:
            node_add_to_stencil(guardian, &a->right_siblings_token, &a->num_right_siblings_token, &a->num_dims_right_siblings, 0, b);
            break;
        default:
            break;
    }
    // Add backward link for b -> a (inverse relation)
    int inverse_relation = geneology_invert_relation(relation);
    if (inverse_relation >= (int)b->num_relations) inverse_relation = 0;
    size_t idx2 = node_add_backward_link(guardian, b, a, inverse_relation);
    // Add to b's stencil (parents, left_siblings, etc.)
    switch (inverse_relation) {
        case EDGE_CHILD_PARENT_CONTIGUOUS:
            node_add_to_stencil(guardian, &b->parents_token, &b->num_parents_token, &b->num_dims_parents, 0, a);
            break;
        case EDGE_SIBLING_RIGHT_TO_LEFT_CONTIGUOUS:
            node_add_to_stencil(guardian, &b->left_siblings_token, &b->num_left_siblings_token, &b->num_dims_left_siblings, 0, a);
            break;
        default:
            break;
    }
    if (idx1 == (size_t)-1 || idx2 == (size_t)-1) return (size_t)-1;
    return idx1;
}

int geneology_invert_relation(int relation_type) {
    switch (relation_type) {
        case EDGE_PARENT_CHILD_CONTIGUOUS:
            return EDGE_CHILD_PARENT_CONTIGUOUS;
        case EDGE_CHILD_PARENT_CONTIGUOUS:
            return EDGE_PARENT_CHILD_CONTIGUOUS;
        case EDGE_SIBLING_LEFT_TO_RIGHT_CONTIGUOUS:
            return EDGE_SIBLING_RIGHT_TO_LEFT_CONTIGUOUS;
        case EDGE_SIBLING_RIGHT_TO_LEFT_CONTIGUOUS:
            return EDGE_SIBLING_LEFT_TO_RIGHT_CONTIGUOUS;
        case EDGE_SIBLING_SIBLING_NONCONTIGUOUS:
            return EDGE_SIBLING_SIBLING_NONCONTIGUOUS; // self-inverse
        case EDGE_LINEAGE_NONCONTIGUOUS:
            return EDGE_LINEAGE_NONCONTIGUOUS; // self-inverse or define more if needed
        case EDGE_ARBITRARY:
        default:
            return EDGE_ARBITRARY;
    }
}

// --- Adaptive Node Operations ---

Node* node_split(TokenGuardian* guardian, const Node* src) {
    if (!guardian || !src) return NULL;
    Node* n = node_create(guardian);
    size_t num_features = guardian_list_size(guardian, src->features_token);
    for (size_t i = 0; i < num_features; ++i) {
        unsigned long feature_token;
        if (guardian_list_get(guardian, src->features_token, i, &feature_token)) {
            char* feature = guardian_deref_string(guardian, feature_token);
            node_add_feature(guardian, n, feature);
        }
    }
    size_t num_exposures = guardian_list_size(guardian, src->exposures_token);
    for (size_t i = 0; i < num_exposures; ++i) {
        unsigned long exposure_token;
        if (guardian_list_get(guardian, src->exposures_token, i, &exposure_token)) {
            NodeExposure* exposure = guardian_deref_exposure(guardian, exposure_token);
            node_add_exposure(guardian, n, exposure->produce, exposure->reverse);
        }
    }
    size_t num_relations = guardian_list_size(guardian, src->relations_token);
    for (size_t i = 0; i < num_relations; ++i) {
        unsigned long relation_token;
        if (guardian_list_get(guardian, src->relations_token, i, &relation_token)) {
            NodeRelation* relation = guardian_deref_relation(guardian, relation_token);
            node_add_relation_full(guardian, n,
                                   relation->type,
                                   relation->forward,
                                   relation->backward,
                                   relation->name,
                                   relation->context);
        }
    }
    return n;
}

int node_should_split(Node* n) {
    if (!n || n->activation_count < 10) return 0;
    double mean = n->activation_sum / n->activation_count;
    double variance = n->activation_sq_sum / n->activation_count - mean * mean;
    return variance > 1.0;
}

void node_record_activation(Node* n, double act) {
    if (!n) return;
    n->activation_sum += act;
    n->activation_sq_sum += act * act;
    n->activation_count++;
}

// === DAG Traversal Utilities ===

void node_for_each_forward(Node* node, NodeVisitFn visit, void* user) {
    if (!node || !visit) return;
    size_t num_links = guardian_list_size(NULL, node->forward_links_token);
    for (size_t i = 0; i < num_links; ++i) {
        unsigned long link_token;
        if (guardian_list_get(NULL, node->forward_links_token, i, &link_token)) {
            NodeLink* link = guardian_deref_link(NULL, link_token);
            visit(link->node, link->relation, user);
        }
    }
}

void node_for_each_backward(Node* node, NodeVisitFn visit, void* user) {
    if (!node || !visit) return;
    size_t num_links = guardian_list_size(NULL, node->backward_links_token);
    for (size_t i = 0; i < num_links; ++i) {
        unsigned long link_token;
        if (guardian_list_get(NULL, node->backward_links_token, i, &link_token)) {
            NodeLink* link = guardian_deref_link(NULL, link_token);
            visit(link->node, link->relation, user);
        }
    }
}

void node_scatter_to_siblings(Node* node, void* data) {
    if (!node) return;
    size_t num_links = guardian_list_size(NULL, node->forward_links_token);
    for (size_t i = 0; i < num_links; ++i) {
        unsigned long link_token;
        if (guardian_list_get(NULL, node->forward_links_token, i, &link_token)) {
            NodeLink* link = guardian_deref_link(NULL, link_token);
            if (link->relation >= 0 && (size_t)link->relation < guardian_list_size(NULL, node->relations_token)) {
                unsigned long relation_token;
                if (guardian_list_get(NULL, node->relations_token, link->relation, &relation_token)) {
                    NodeRelation* rel = guardian_deref_relation(NULL, relation_token);
                    if (rel->forward)
                        rel->forward(link->node, data);
                }
            }
        }
    }
}

void node_gather_from_siblings(Node* node, void* out) {
    if (!node) return;
    size_t num_links = guardian_list_size(NULL, node->backward_links_token);
    for (size_t i = 0; i < num_links; ++i) {
        unsigned long link_token;
        if (guardian_list_get(NULL, node->backward_links_token, i, &link_token)) {
            NodeLink* link = guardian_deref_link(NULL, link_token);
            if (link->relation >= 0 && (size_t)link->relation < guardian_list_size(NULL, node->relations_token)) {
                unsigned long relation_token;
                if (guardian_list_get(NULL, node->relations_token, link->relation, &relation_token)) {
                    NodeRelation* rel = guardian_deref_relation(NULL, relation_token);
                    if (rel->backward)
                        rel->backward(link->node, out);
                }
            }
        }
    }
}

// === Additional Accessors ===

NodeRelation* node_get_relation(const Node* node, size_t index) {
    if (!node || index >= guardian_list_size(NULL, node->relations_token)) return NULL;
    unsigned long relation_token;
    if (!guardian_list_get(NULL, node->relations_token, index, &relation_token)) return NULL;
    return guardian_deref_relation(NULL, relation_token);
}

const char* node_get_feature(const Node* node, size_t index) {
    if (!node || index >= guardian_list_size(NULL, node->features_token)) return NULL;
    unsigned long feature_token;
    if (!guardian_list_get(NULL, node->features_token, index, &feature_token)) return NULL;
    return guardian_deref_string(NULL, feature_token);
}

NodeExposure* node_get_exposure(const Node* node, size_t index) {
    if (!node || index >= guardian_list_size(NULL, node->exposures_token)) return NULL;
    unsigned long exposure_token;
    if (!guardian_list_get(NULL, node->exposures_token, index, &exposure_token)) return NULL;
    return guardian_deref_exposure(NULL, exposure_token);
}

const NodeLink* node_get_forward_link(const Node* node, size_t index) {
    if (!node || index >= guardian_list_size(NULL, node->forward_links_token)) return NULL;
    unsigned long link_token;
    if (!guardian_list_get(NULL, node->forward_links_token, index, &link_token)) return NULL;
    return guardian_deref_link(NULL, link_token);
}

const NodeLink* node_get_backward_link(const Node* node, size_t index) {
    if (!node || index >= guardian_list_size(NULL, node->backward_links_token)) return NULL;
    unsigned long link_token;
    if (!guardian_list_get(NULL, node->backward_links_token, index, &link_token)) return NULL;
    return guardian_deref_link(NULL, link_token);
}

// === Recursive Traversal ===

void node_scatter_to_descendants(Node* node, void* data) {
    if (!node) return;
    node_scatter_to_siblings(node, data);
    size_t num_links = guardian_list_size(NULL, node->forward_links_token);
    for (size_t i = 0; i < num_links; ++i) {
        unsigned long link_token;
        if (guardian_list_get(NULL, node->forward_links_token, i, &link_token)) {
            NodeLink* link = guardian_deref_link(NULL, link_token);
            node_scatter_to_descendants(link->node, data);
        }
    }
}

void node_gather_from_ancestors(Node* node, void* out) {
    if (!node) return;
    node_gather_from_siblings(node, out);
    size_t num_links = guardian_list_size(NULL, node->backward_links_token);
    for (size_t i = 0; i < num_links; ++i) {
        unsigned long link_token;
        if (guardian_list_get(NULL, node->backward_links_token, i, &link_token)) {
            NodeLink* link = guardian_deref_link(NULL, link_token);
            node_gather_from_ancestors(link->node, out);
        }
    }
}

// --- SimpleGraph implementation ---
SimpleGraph* simplegraph_create(TokenGuardian* guardian, Geneology* g) {
    if (!guardian) return NULL;
    unsigned long graph_token;
    SimpleGraph* graph = (SimpleGraph*)guardian_alloc(guardian, sizeof(SimpleGraph), &graph_token);
    graph->geneology = g;
    return graph;
}

void simplegraph_destroy(TokenGuardian* guardian, SimpleGraph* graph) {
    if (!guardian || !graph) return;
    guardian_free(guardian, graph->edges);
    if (graph->feature_block) {
        // User is responsible for freeing tensor objects
        guardian_free(guardian, graph->feature_block);
    }
    if (graph->node_feature_maps) {
        for (size_t i = 0; i < graph->num_nodes; ++i) {
            SimpleGraphFeatureMap* fmap = &graph->node_feature_maps[i];
            for (size_t j = 0; j < fmap->cap_entries; ++j) {
                guardian_free(guardian, fmap->entries[j].key);
            }
            guardian_free(guardian, fmap->entries);
        }
        guardian_free(guardian, graph->node_feature_maps);
    }
    guardian_free(guardian, graph);
}

static void simplegraph_grow_edges(TokenGuardian* guardian, SimpleGraph* graph) {
    if (graph->num_edges == graph->cap_edges) {
        size_t new_cap = graph->cap_edges ? graph->cap_edges * 2 : 4;
        SimpleGraphEdge* tmp = guardian_realloc(guardian, graph->edges, new_cap * sizeof(SimpleGraphEdge));
        if (!tmp) return;
        graph->edges = tmp;
        graph->cap_edges = new_cap;
    }
}

void simplegraph_add_edge(TokenGuardian* guardian, SimpleGraph* graph, Node* src, Node* dst, SimpleGraphEdgeType type, int relation) {
    if (!guardian || !graph || !src || !dst) return;
    simplegraph_grow_edges(guardian, graph);
    SimpleGraphEdge e = {src, dst, type, relation};
    graph->edges[graph->num_edges++] = e;
    // Optionally, also add to geneology if needed
}

static void simplegraph_grow_features(TokenGuardian* guardian, SimpleGraph* graph) {
    if (graph->num_features == graph->cap_features) {
        size_t new_cap = graph->cap_features ? graph->cap_features * 2 : 8;
        void** tmp = guardian_realloc(guardian, graph->feature_block, new_cap * sizeof(void*));
        if (!tmp) return;
        graph->feature_block = tmp;
        graph->cap_features = new_cap;
    }
}

static void simplegraph_grow_node_maps(TokenGuardian* guardian, SimpleGraph* graph) {
    size_t g_nodes = geneology_num_nodes(graph->geneology);
    if (g_nodes > graph->cap_nodes) {
        size_t new_cap = graph->cap_nodes ? graph->cap_nodes * 2 : 8;
        if (new_cap < g_nodes) new_cap = g_nodes;
        SimpleGraphFeatureMap* tmp = guardian_realloc(guardian, graph->node_feature_maps, new_cap * sizeof(SimpleGraphFeatureMap));
        if (!tmp) return;
        graph->node_feature_maps = tmp;
        for (size_t i = graph->cap_nodes; i < new_cap; ++i) {
            memset(&graph->node_feature_maps[i], 0, sizeof(SimpleGraphFeatureMap));
        }
        graph->cap_nodes = new_cap;
    }
    graph->num_nodes = g_nodes;
}

// Simple open addressing hash for features
static size_t simplegraph_hash(const char* s, size_t cap) {
    size_t h = 5381;
    while (*s) h = ((h << 5) + h) + (unsigned char)(*s++);
    return h % cap;
}

void simplegraph_add_feature(TokenGuardian* guardian, SimpleGraph* graph, Node* node, const char* feature_name, void* tensor_ptr) {
    if (!guardian || !graph || !node || !feature_name) return;
    simplegraph_grow_features(guardian, graph);
    simplegraph_grow_node_maps(guardian, graph);
    // Find node index in geneology
    size_t idx = 0;
    for (; idx < graph->num_nodes; ++idx) if (geneology_get_node(graph->geneology, idx) == node) break;
    if (idx == graph->num_nodes) return;
    SimpleGraphFeatureMap* fmap = &graph->node_feature_maps[idx];
    // Grow feature map if needed
    if (fmap->num_entries * 2 >= fmap->cap_entries) {
        size_t new_cap = fmap->cap_entries ? fmap->cap_entries * 2 : 8;
        SimpleGraphFeatureEntry* new_entries = guardian_calloc(guardian, new_cap, sizeof(SimpleGraphFeatureEntry));
        for (size_t i = 0; i < fmap->cap_entries; ++i) {
            if (fmap->entries[i].key) {
                size_t h = simplegraph_hash(fmap->entries[i].key, new_cap);
                while (new_entries[h].key) h = (h + 1) % new_cap;
                new_entries[h] = fmap->entries[i];
            }
        }
        guardian_free(guardian, fmap->entries);
        fmap->entries = new_entries;
        fmap->cap_entries = new_cap;
    }
    // Insert
    size_t h = simplegraph_hash(feature_name, fmap->cap_entries);
    while (fmap->entries && fmap->entries[h].key) {
        if (strcmp(fmap->entries[h].key, feature_name) == 0) {
            fmap->entries[h].value = tensor_ptr;
            return;
        }
        h = (h + 1) % fmap->cap_entries;
    }
    if (!fmap->entries) {
        fmap->cap_entries = 8;
        fmap->entries = guardian_calloc(guardian, fmap->cap_entries, sizeof(SimpleGraphFeatureEntry));
    }
    fmap->entries[h].key = strdup(feature_name);
    fmap->entries[h].value = tensor_ptr;
    fmap->num_entries++;
    graph->feature_block[graph->num_features++] = tensor_ptr;
}

void* simplegraph_get_feature(TokenGuardian* guardian, SimpleGraph* graph, Node* node, const char* feature_name) {
    if (!guardian || !graph || !node || !feature_name) return NULL;
    simplegraph_grow_node_maps(guardian, graph);
    size_t idx = 0;
    for (; idx < graph->num_nodes; ++idx) if (geneology_get_node(graph->geneology, idx) == node) break;
    if (idx == graph->num_nodes) return NULL;
    SimpleGraphFeatureMap* fmap = &graph->node_feature_maps[idx];
    if (!fmap->entries) return NULL;
    size_t h = simplegraph_hash(feature_name, fmap->cap_entries);
    for (size_t i = 0; i < fmap->cap_entries; ++i) {
        size_t j = (h + i) % fmap->cap_entries;
        if (!fmap->entries[j].key) return NULL;
        if (strcmp(fmap->entries[j].key, feature_name) == 0) return fmap->entries[j].value;
    }
    return NULL;
}

void simplegraph_forward(TokenGuardian* guardian, SimpleGraph* graph) {
    if (!guardian || !graph) return;
    for (size_t i = 0; i < graph->num_edges; ++i) {
        Node* src = graph->edges[i].src;
        Node* dst = graph->edges[i].dst;
        int rel = graph->edges[i].relation;
        if (src && rel >= 0 && (size_t)rel < guardian_list_size(NULL, src->relations_token)) {
            unsigned long relation_token;
            if (guardian_list_get(NULL, src->relations_token, rel, &relation_token)) {
                NodeRelation* r = guardian_deref_relation(NULL, relation_token);
                if (r->forward) r->forward(dst, NULL); // or pass data as needed
            }
        }
    }
}

void simplegraph_backward(TokenGuardian* guardian, SimpleGraph* graph) {
    if (!guardian || !graph) return;
    for (size_t i = graph->num_edges; i-- > 0;) {
        Node* src = graph->edges[i].src;
        Node* dst = graph->edges[i].dst;
        int rel = graph->edges[i].relation;
        if (src && rel >= 0 && (size_t)rel < guardian_list_size(NULL, src->relations_token)) {
            unsigned long relation_token;
            if (guardian_list_get(NULL, src->relations_token, rel, &relation_token)) {
                NodeRelation* r = guardian_deref_relation(NULL, relation_token);
                if (r->backward) r->backward(dst, NULL); // or pass data as needed
            }
        }
    }
}

// --- DAG Manifest Structures Implementation ---
#include <stdlib.h>
#include <string.h>

Dag* dag_create(TokenGuardian* guardian) {
    if (!guardian) return NULL;
    unsigned long dag_token;
    Dag* dag = (Dag*)guardian_alloc(guardian, sizeof(Dag), &dag_token);
    return dag;
}

void dag_destroy(TokenGuardian* guardian, Dag* dag) {
    if (!guardian || !dag) return;
    for (size_t i = 0; i < dag->num_manifests; ++i) {
        DagManifest* manifest = &dag->manifests[i];
        for (size_t l = 0; l < manifest->num_levels; ++l) {
            DagManifestLevel* level = &manifest->levels[l];
            for (size_t m = 0; m < level->num_mappings; ++m) {
                DagManifestMapping* mapping = &level->mappings[m];
                guardian_free(guardian, mapping->inputs);
                guardian_free(guardian, mapping->outputs);
            }
            guardian_free(guardian, level->mappings);
        }
        guardian_free(guardian, manifest->levels);
    }
    guardian_free(guardian, dag->manifests);
    guardian_free(guardian, dag);
}
void dag_add_manifest(TokenGuardian* guardian, Dag* dag, DagManifest* manifest) {
    if (!guardian || !dag || !manifest) return;
    if (dag->num_manifests == dag->cap_manifests) {
        size_t new_cap = dag->cap_manifests ? dag->cap_manifests * 2 : 4;
        DagManifest* tmp = guardian_realloc(guardian, dag->manifests, new_cap * sizeof(DagManifest));
        if (!tmp) return;
        dag->manifests = tmp;
        dag->cap_manifests = new_cap;
    }
    dag->manifests[dag->num_manifests++] = *manifest;
}


// --- Emergence Implementation ---
Emergence* emergence_create(void) {
    unsigned long emergence_token;
    Emergence* e = (Emergence*)guardian_alloc(NULL, sizeof(Emergence), &emergence_token);
#ifdef _WIN32
    InitializeCriticalSection(&e->thread_lock);
#else
    pthread_mutex_init(&e->thread_lock, NULL);
#endif
    return e;
}

void emergence_destroy(Emergence* e) {
    if (!e) return;
#ifdef _WIN32
    DeleteCriticalSection(&e->thread_lock);
#else
    pthread_mutex_destroy(&e->thread_lock);
#endif
    guardian_free(NULL, e);
}

void emergence_lock(Emergence* e) {
    if (!e) return;
#ifdef _WIN32
    EnterCriticalSection(&e->thread_lock);
#else
    pthread_mutex_lock(&e->thread_lock);
#endif
    e->is_locked = 1;
}

void emergence_release(Emergence* e) {
    if (!e) return;
    e->is_locked = 0;
#ifdef _WIN32
    LeaveCriticalSection(&e->thread_lock);
#else
    pthread_mutex_unlock(&e->thread_lock);
#endif
}

void emergence_resolve(Emergence* e) {
    // Placeholder for lock resolution logic
    if (!e) return;
    // Could implement deadlock detection, etc.
}

void emergence_update(Emergence* e, Node* node, double activation, uint64_t global_step, uint64_t timestamp) {
    if (!e || !node) return;
    node_record_activation(node, activation);
    e->last_global_step = global_step;
    e->last_timestamp = timestamp;

    if (e->should_split && e->should_split(e, node)) {
        Node* new_node = node_split(node);
        if (e->split) e->split(e, new_node);
    }
    if (e->should_apoptose && e->should_apoptose(e, node)) {
        if (e->apoptose) e->apoptose(e, node);
    }
    if (e->should_metastasize && e->should_metastasize(e, node)) {
        if (e->metastasize) e->metastasize(e, node);
    }
}

size_t dag_num_manifests(const Dag* dag) {
    return dag ? dag->num_manifests : 0;
}

DagManifest* dag_get_manifest(const Dag* dag, size_t idx) {
    if (!dag || idx >= dag->num_manifests) return NULL;
    return &dag->manifests[idx];
}

size_t dag_manifest_num_levels(const DagManifest* manifest) {
    return manifest ? manifest->num_levels : 0;
}

DagManifestLevel* dag_manifest_get_level(const DagManifest* manifest, size_t level_idx) {
    if (!manifest || level_idx >= manifest->num_levels) return NULL;
    return &manifest->levels[level_idx];
}

size_t dag_level_num_mappings(const DagManifestLevel* level) {
    return level ? level->num_mappings : 0;
}

DagManifestMapping* dag_level_get_mapping(const DagManifestLevel* level) {
    return level ? level->mappings : NULL;
}

void dag_gather(const DagManifestMapping* mapping, void* out) {
    (void)mapping;
    (void)out;
    /* TODO: gather data from inputs */
}

void dag_scatter(const DagManifestMapping* mapping, void* data) {
    (void)mapping;
    (void)data;
    /* TODO: scatter data to outputs */
}

// --- NeuralNetwork implementation ---
#include <string.h>

NeuralNetwork* neuralnetwork_create(void) {
    NeuralNetwork* nn = (NeuralNetwork*)guardian_alloc(NULL, sizeof(NeuralNetwork), NULL);
    return nn;
}

void neuralnetwork_destroy(NeuralNetwork* nn) {
    if (!nn) return;
    for (size_t d = 0; d < nn->num_dags; ++d) {
        dag_destroy(nn->dags[d]);
        for (size_t s = 0; s < nn->num_steps[d]; ++s) {
            guardian_free(NULL, nn->steps[d][s]);
        }
    }
    guardian_free(NULL, nn);
}

void neuralnetwork_register_function(NeuralNetwork* nn, const char* name, NNForwardFn forward, NNBackwardFn backward) {
    if (!nn || !name) return;
    if (nn->function_repo.num_entries >= NN_MAX_FUNCTIONS) return;
    NeuralNetworkFunctionEntry* e = &nn->function_repo.entries[nn->function_repo.num_entries++];
    e->name = name;
    e->forward = forward;
    e->backward = backward;
}

void neuralnetwork_set_step_function(NeuralNetwork* nn, size_t dag_idx, size_t step_idx, const char* function_name, void* user_data) {
    if (!nn || dag_idx >= nn->num_dags || step_idx >= NN_MAX_STEPS) return;
    NeuralNetworkStep* step = nn->steps[dag_idx][step_idx];
    if (!step) {
        step = (NeuralNetworkStep*)guardian_alloc(NULL, sizeof(NeuralNetworkStep), NULL);
        nn->steps[dag_idx][step_idx] = step;
        if (step_idx >= nn->num_steps[dag_idx]) nn->num_steps[dag_idx] = step_idx + 1;
    }
    for (size_t i = 0; i < nn->function_repo.num_entries; ++i) {
        NeuralNetworkFunctionEntry* e = &nn->function_repo.entries[i];
        if (strcmp(e->name, function_name) == 0) {
            step->forward = e->forward;
            step->backward = e->backward;
            step->user_data = user_data;
            break;
        }
    }
}

void neuralnetwork_forward(NeuralNetwork* nn) {
    (void)nn; // TODO
}

void neuralnetwork_backward(NeuralNetwork* nn) {
    (void)nn; // TODO
}

void neuralnetwork_forwardstep(NeuralNetwork* nn, size_t dag_idx, size_t step_idx) {
    (void)nn; (void)dag_idx; (void)step_idx; // TODO
}

void neuralnetwork_backwardstep(NeuralNetwork* nn, size_t dag_idx, size_t step_idx) {
    (void)nn; (void)dag_idx; (void)step_idx; // TODO
}

// --- Stencil-Informed Neighbor Mapping Implementation ---
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void neighbor_map_grow(NeighborMap* map) {
    if (map->count == map->cap) {
        size_t new_cap = map->cap ? map->cap * 2 : 8;
        NeighborEntry* tmp = guardian_realloc(NULL, map->entries, new_cap * sizeof(NeighborEntry));
        if (!tmp) return;
        map->entries = tmp;
        map->cap = new_cap;
    }
}

int node_attach_neighbor(Node* node, Node* neighbor, size_t pole_index, const char* label) {
    if (!node || !neighbor) return 0;
    neighbor_map_grow(&node->neighbor_map);
    NeighborEntry* entry = &node->neighbor_map.entries[node->neighbor_map.count++];
    entry->neighbor = neighbor;
    entry->label.label = label ? strdup(label) : NULL;
    entry->label.pole_index = pole_index;
    return 1;
}

int node_detach_neighbor(Node* node, size_t pole_index) {
    if (!node) return 0;
    for (size_t i = 0; i < node->neighbor_map.count; ++i) {
        if (node->neighbor_map.entries[i].label.pole_index == pole_index) {
            guardian_free(NULL, node->neighbor_map.entries[i].label.label);
            node->neighbor_map.entries[i] = node->neighbor_map.entries[node->neighbor_map.count - 1];
            node->neighbor_map.count--;
            return 1;
        }
    }
    return 0;
}

int node_detach_neighbor_by_label(Node* node, const char* label) {
    if (!node || !label) return 0;
    for (size_t i = 0; i < node->neighbor_map.count; ++i) {
        if (node->neighbor_map.entries[i].label.label && strcmp(node->neighbor_map.entries[i].label.label, label) == 0) {
            guardian_free(NULL, node->neighbor_map.entries[i].label.label);
            node->neighbor_map.entries[i] = node->neighbor_map.entries[node->neighbor_map.count - 1];
            node->neighbor_map.count--;
            return 1;
        }
    }
    return 0;
}

Node* node_get_neighbor(const Node* node, size_t pole_index) {
    if (!node) return NULL;
    for (size_t i = 0; i < node->neighbor_map.count; ++i) {
        if (node->neighbor_map.entries[i].label.pole_index == pole_index)
            return node->neighbor_map.entries[i].neighbor;
    }
    return NULL;
}

Node* node_get_neighbor_by_label(const Node* node, const char* label) {
    if (!node || !label) return NULL;
    for (size_t i = 0; i < node->neighbor_map.count; ++i) {
        if (node->neighbor_map.entries[i].label.label && strcmp(node->neighbor_map.entries[i].label.label, label) == 0)
            return node->neighbor_map.entries[i].neighbor;
    }
    return NULL;
}

int node_ensure_bidirectional_neighbor(Node* node, Node* neighbor, size_t pole_index, const char* label, int require_bidirectional) {
    if (!node_attach_neighbor(node, neighbor, pole_index, label)) return 0;
    if (require_bidirectional) {
        // Attach reciprocal link if not already present
        int found = 0;
        for (size_t i = 0; i < neighbor->neighbor_map.count; ++i) {
            if (neighbor->neighbor_map.entries[i].neighbor == node) { found = 1; break; }
        }
        if (!found) {
            // Auto-label reciprocal if not provided
            char auto_label[64];
            snprintf(auto_label, sizeof(auto_label), "recip_%zu", pole_index);
            node_attach_neighbor(neighbor, node, pole_index, label ? auto_label : NULL);
        }
    }
    return 1;
}

// Procedural relationship naming for stencil-based relationships
// Returns a malloc'd string describing the relationship (caller must free)
char* node_generate_stencil_relation_name(const GeneralStencil* stencil, size_t pole_index, const char* base_type) {
    if (!stencil || !base_type) return NULL;
    // Compose a name like: "rect3d_r1_pole2" or "polar2d_r2_pole0"
    const char* type_str = base_type;
    size_t dims = stencil->dims;
    size_t radius = 0;
    // Try to infer radius if possible (for rectangular stencils)
    if (stencil->count > 0 && stencil->poles) {
        int max_offset = 0;
        for (size_t i = 0; i < stencil->count; ++i) {
            for (size_t d = 0; d < dims; ++d) {
                int off = stencil->poles[i].offsets[d];
                if (off < 0) off = -off;
                if (off > max_offset) max_offset = off;
            }
        }
        radius = max_offset;
    }
    char* name = (char*)guardian_alloc(NULL, 128, NULL);
    snprintf(name, 128, "%s_%zud_r%zu_pole%zu", type_str, dims, radius, pole_index);
    return name;
}

// --- StencilSet helpers ---
StencilSet* stencilset_wrap_single(GeneralStencil* stencil) {
    if (!stencil) return NULL;
    unsigned long set_token, stencils_token, relation_token;
    StencilSet* set = (StencilSet*)guardian_alloc(NULL, sizeof(StencilSet), &set_token);
    set->stencils = (GeneralStencil**)guardian_alloc(NULL, sizeof(GeneralStencil*), &stencils_token);
    set->stencils[0] = stencil;
    set->count = 1;
    set->cap = 1;
    set->relation = (StencilRelation**)guardian_alloc(NULL, sizeof(StencilRelation*), &relation_token);
    set->relation[0] = (StencilRelation*)guardian_alloc(NULL, sizeof(StencilRelation), NULL);
    set->relation[0][0].type = STENCIL_RELATION_SAME_COORDINATE_SET;
    set->relation[0][0].context = NULL;
    set->relation[0][0].similarity_fn = NULL;
    return set;
}

void stencilset_init_fully_connected(StencilSet* set, int default_relation_type) {
    if (!set || set->count == 0) return;
    unsigned long relation_token;
    set->relation = (StencilRelation**)guardian_alloc(NULL, set->count * sizeof(StencilRelation*), &relation_token);
    for (size_t i = 0; i < set->count; ++i) {
        set->relation[i] = (StencilRelation*)guardian_alloc(NULL, set->count * sizeof(StencilRelation), NULL);
        for (size_t j = 0; j < set->count; ++j) {
            set->relation[i][j].type = (i == j) ? STENCIL_RELATION_SAME_COORDINATE_SET : default_relation_type;
            set->relation[i][j].context = NULL;
            set->relation[i][j].similarity_fn = NULL;
        }
    }
}

// Negotiation logic for node bonds
int stencilset_negotiate_bond(const StencilSet* a, size_t pole_a, const StencilSet* b, size_t pole_b, StencilRelation* out_relation) {
    if (!a || !b || pole_a >= a->count || pole_b >= b->count) return STENCIL_RELATION_ORTHOGONAL;
    // Default: if stencils are the same pointer, same coordinate set
    if (a->stencils[pole_a] == b->stencils[pole_b]) {
        if (out_relation) {
            out_relation->type = STENCIL_RELATION_SAME_COORDINATE_SET;
            out_relation->context = NULL;
            out_relation->similarity_fn = NULL;
        }
        return STENCIL_RELATION_SAME_COORDINATE_SET;
    }
    // Example: check for rectangular or polar by dims/structure (simple heuristic)
    if (a->stencils[pole_a]->dims == b->stencils[pole_b]->dims) {
        // Could add more checks for axis-aligned, polar, etc.
        if (out_relation) {
            out_relation->type = STENCIL_RELATION_DIFFERENT_COORDINATE_SET;
            out_relation->context = NULL;
            out_relation->similarity_fn = NULL;
        }
        return STENCIL_RELATION_DIFFERENT_COORDINATE_SET;
    }
    // Otherwise, treat as orthogonal
    if (out_relation) {
        out_relation->type = STENCIL_RELATION_ORTHOGONAL;
        out_relation->context = NULL;
        out_relation->similarity_fn = NULL;
    }
    return STENCIL_RELATION_ORTHOGONAL;
}

// --- Quaternion and QuaternionHistory implementation ---
#include <math.h>
QuaternionHistory* quaternion_history_create(size_t window_size, QuaternionDecayFn decay_fn, void* user_data) {
    unsigned long hist_token, quats_token;
    QuaternionHistory* hist = (QuaternionHistory*)guardian_alloc(NULL, sizeof(QuaternionHistory), &hist_token);
    hist->window_size = window_size;
    hist->cap = window_size ? window_size : 16;
    hist->quats = (Quaternion*)guardian_alloc(NULL, hist->cap * sizeof(Quaternion), &quats_token);
    hist->decay_fn = decay_fn;
    hist->decay_user_data = user_data;
    return hist;
}

void quaternion_history_destroy(QuaternionHistory* hist) {
    if (!hist) return;
    guardian_free(NULL, hist->quats);
    guardian_free(NULL, hist);
}

void quaternion_history_add(QuaternionHistory* hist, Quaternion q) {
    if (!hist) return;
    if (hist->window_size && hist->count == hist->window_size) {
        // Move window left
        memmove(hist->quats, hist->quats + 1, (hist->count - 1) * sizeof(Quaternion));
        hist->quats[hist->count - 1] = q;
    }
    else {
        if (hist->count == hist->cap) {
            size_t new_cap = hist->cap * 2;
            Quaternion* tmp = guardian_realloc(NULL, hist->quats, new_cap * sizeof(Quaternion));
            if (!tmp) return;
            hist->quats = tmp;
            hist->cap = new_cap;
        }
        hist->quats[hist->count++] = q;
    }
}

// Simple weighted sum aggregate (normalize at end)
Quaternion quaternion_history_aggregate(const QuaternionHistory* hist) {
    Quaternion sum = { 0, 0, 0, 0 };
    if (!hist || !hist->count) return sum;
    double total_weight = 0.0;
    for (size_t i = 0; i < hist->count; ++i) {
        double w = 1.0;
        if (hist->decay_fn) w = hist->decay_fn(i, hist->count, hist->decay_user_data);
        sum.w += hist->quats[i].w * w;
        sum.x += hist->quats[i].x * w;
        sum.y += hist->quats[i].y * w;
        sum.z += hist->quats[i].z * w;
        total_weight += w;
    }
    if (total_weight > 0) {
        sum.w /= total_weight;
        sum.x /= total_weight;
        sum.y /= total_weight;
        sum.z /= total_weight;
    }
    // Normalize quaternion
    double norm = sqrt(sum.w * sum.w + sum.x * sum.x + sum.y * sum.y + sum.z * sum.z);
    if (norm > 0) {
        sum.w /= norm;
        sum.x /= norm;
        sum.y /= norm;
        sum.z /= norm;
    }
    return sum;
}

// --- Quaternion Operators and Orientation Validators ---
#include <math.h>

Quaternion quaternion_add(Quaternion a, Quaternion b) {
    return (Quaternion) { a.w + b.w, a.x + b.x, a.y + b.y, a.z + b.z };
}

Quaternion quaternion_sub(Quaternion a, Quaternion b) {
    return (Quaternion) { a.w - b.w, a.x - b.x, a.y - b.y, a.z - b.z };
}

Quaternion quaternion_scale(Quaternion q, double s) {
    return (Quaternion) { q.w* s, q.x* s, q.y* s, q.z* s };
}

Quaternion quaternion_div(Quaternion q, double s) {
    return (Quaternion) { q.w / s, q.x / s, q.y / s, q.z / s };
}

Quaternion quaternion_conjugate(Quaternion q) {
    return (Quaternion) { q.w, -q.x, -q.y, -q.z };
}

Quaternion quaternion_mul(Quaternion a, Quaternion b) {
    return (Quaternion) {
        a.w* b.w - a.x * b.x - a.y * b.y - a.z * b.z,
            a.w* b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w* b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w* b.z + a.x * b.y - a.y * b.x + a.z * b.w
    };
}

double quaternion_dot(Quaternion a, Quaternion b) {
    return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
}

double quaternion_norm(Quaternion q) {
    return sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
}

Quaternion quaternion_normalize(Quaternion q) {
    double n = quaternion_norm(q);
    if (n > 0) return quaternion_div(q, n);
    return q;
}

Quaternion quaternion_inverse(Quaternion q) {
    double n2 = quaternion_dot(q, q);
    if (n2 > 0) return quaternion_div(quaternion_conjugate(q), n2);
    return q;
}

// Slerp (spherical linear interpolation)
Quaternion quaternion_slerp(Quaternion a, Quaternion b, double t) {
    double dot = quaternion_dot(a, b);
    if (dot < 0) {
        b = quaternion_scale(b, -1);
        dot = -dot;
    }
    if (dot > 0.9995) {
        Quaternion result = quaternion_add(quaternion_scale(a, 1 - t), quaternion_scale(b, t));
        return quaternion_normalize(result);
    }
    double theta_0 = acos(dot);
    double theta = theta_0 * t;
    double sin_theta = sin(theta);
    double sin_theta_0 = sin(theta_0);
    double s0 = cos(theta) - dot * sin_theta / sin_theta_0;
    double s1 = sin_theta / sin_theta_0;
    return quaternion_add(quaternion_scale(a, s0), quaternion_scale(b, s1));
}

// Axis-angle conversion
void quaternion_to_axis_angle(Quaternion q, double* axis_out, double* angle_out) {
    if (!axis_out || !angle_out) return;
    if (q.w > 1) q = quaternion_normalize(q);
    *angle_out = 2 * acos(q.w);
    double s = sqrt(1 - q.w * q.w);
    if (s < 1e-8) {
        axis_out[0] = 1; axis_out[1] = 0; axis_out[2] = 0;
    }
    else {
        axis_out[0] = q.x / s;
        axis_out[1] = q.y / s;
        axis_out[2] = q.z / s;
    }
}

Quaternion quaternion_from_axis_angle(const double* axis, double angle) {
    double s = sin(angle / 2);
    return (Quaternion) { cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s };
}

// Euler conversion (ZYX order)
void quaternion_to_euler(Quaternion q, double* roll, double* pitch, double* yaw) {
    // roll (x), pitch (y), yaw (z)
    double ysqr = q.y * q.y;
    // roll (x-axis rotation)
    double t0 = +2.0 * (q.w * q.x + q.y * q.z);
    double t1 = +1.0 - 2.0 * (q.x * q.x + ysqr);
    if (roll) *roll = atan2(t0, t1);
    // pitch (y-axis rotation)
    double t2 = +2.0 * (q.w * q.y - q.z * q.x);
    t2 = t2 > 1.0 ? 1.0 : t2;
    t2 = t2 < -1.0 ? -1.0 : t2;
    if (pitch) *pitch = asin(t2);
    // yaw (z-axis rotation)
    double t3 = +2.0 * (q.w * q.z + q.x * q.y);
    double t4 = +1.0 - 2.0 * (ysqr + q.z * q.z);
    if (yaw) *yaw = atan2(t3, t4);
}

Quaternion quaternion_from_euler(double roll, double pitch, double yaw) {
    double cy = cos(yaw * 0.5);
    double sy = sin(yaw * 0.5);
    double cp = cos(pitch * 0.5);
    double sp = sin(pitch * 0.5);
    double cr = cos(roll * 0.5);
    double sr = sin(roll * 0.5);
    Quaternion q;
    q.w = cr * cp * cy + sr * sp * sy;
    q.x = sr * cp * cy - cr * sp * sy;
    q.y = cr * sp * cy + sr * cp * sy;
    q.z = cr * cp * sy - sr * sp * cy;
    return q;
}

void quaternion_to_matrix(Quaternion q, double m[3][3]) {
    q = quaternion_normalize(q);
    double xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
    double xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    double wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
    m[0][0] = 1 - 2 * (yy + zz);
    m[0][1] = 2 * (xy - wz);
    m[0][2] = 2 * (xz + wy);
    m[1][0] = 2 * (xy + wz);
    m[1][1] = 1 - 2 * (xx + zz);
    m[1][2] = 2 * (yz - wx);
    m[2][0] = 2 * (xz - wy);
    m[2][1] = 2 * (yz + wx);
    m[2][2] = 1 - 2 * (xx + yy);
}

Quaternion quaternion_from_matrix(const double m[3][3]) {
    Quaternion q;
    double trace = m[0][0] + m[1][1] + m[2][2];
    if (trace > 0) {
        double s = 0.5 / sqrt(trace + 1.0);
        q.w = 0.25 / s;
        q.x = (m[2][1] - m[1][2]) * s;
        q.y = (m[0][2] - m[2][0]) * s;
        q.z = (m[1][0] - m[0][1]) * s;
    }
    else {
        if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
            double s = 2.0 * sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]);
            q.w = (m[2][1] - m[1][2]) / s;
            q.x = 0.25 * s;
            q.y = (m[0][1] + m[1][0]) / s;
            q.z = (m[0][2] + m[2][0]) / s;
        }
        else if (m[1][1] > m[2][2]) {
            double s = 2.0 * sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]);
            q.w = (m[0][2] - m[2][0]) / s;
            q.x = (m[0][1] + m[1][0]) / s;
            q.y = 0.25 * s;
            q.z = (m[1][2] + m[2][1]) / s;
        }
        else {
            double s = 2.0 * sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]);
            q.w = (m[1][0] - m[0][1]) / s;
            q.x = (m[0][2] + m[2][0]) / s;
            q.y = (m[1][2] + m[2][1]) / s;
            q.z = 0.25 * s;
        }
    }
    return quaternion_normalize(q);
}

int quaternion_is_normalized(Quaternion q, double tol) {
    double n = quaternion_norm(q);
    return fabs(n - 1.0) < tol;
}

double quaternion_angle_between(Quaternion a, Quaternion b) {
    double dot = quaternion_dot(a, b);
    if (dot > 1.0) dot = 1.0;
    if (dot < -1.0) dot = -1.0;
    return acos(fabs(dot)) * 2.0;
}

int quaternion_is_continuous(const Quaternion* seq, size_t count, double tol) {
    if (!seq || count < 2) return 1;
    for (size_t i = 1; i < count; ++i) {
        if (quaternion_angle_between(seq[i - 1], seq[i]) > tol) return 0;
    }
    return 1;
}

int quaternion_is_gimbal_lock(Quaternion q, double tol) {
    // Gimbal lock if pitch is near +/-90 degrees
    double roll, pitch, yaw;
    quaternion_to_euler(q, &roll, &pitch, &yaw);
    return fabs(fabs(pitch) - M_PI / 2) < tol;
}

int quaternion_history_is_smooth(const QuaternionHistory* hist, double tol) {
    if (!hist || hist->count < 2) return 1;
    return quaternion_is_continuous(hist->quats, hist->count, tol);
}

int quaternion_history_has_flips(const QuaternionHistory* hist, double tol) {
    if (!hist || hist->count < 2) return 0;
    for (size_t i = 1; i < hist->count; ++i) {
        if (quaternion_angle_between(hist->quats[i - 1], hist->quats[i]) > tol) return 1;
    }
    return 0;
}

Quaternion quaternion_history_average(const QuaternionHistory* hist) {
    // Use the aggregate function (weighted sum, normalized)
    return quaternion_history_aggregate(hist);
}

// --- MISSING FUNCTION STUBS ---
#include "geometry/utils.h"

// Node locking API
void node_lock(Node* node) {
    if (!node) return;
#ifdef _WIN32
    EnterCriticalSection(&node->mutex);
#else
    pthread_mutex_lock(&node->mutex);
#endif
}

void node_unlock(Node* node) {
    if (!node) return;
#ifdef _WIN32
    LeaveCriticalSection(&node->mutex);
#else
    pthread_mutex_unlock(&node->mutex);
#endif
}

int node_trylock(Node* node) {
    if (!node) return 0;
#ifdef _WIN32
    return TryEnterCriticalSection(&node->mutex);
#else
    return pthread_mutex_trylock(&node->mutex) == 0;
#endif
}

int node_is_locked(const Node* node) {
    if (!node) return 0;
#ifdef _WIN32
    int locked = !TryEnterCriticalSection((LPCRITICAL_SECTION)&node->mutex);
    if (!locked) LeaveCriticalSection((LPCRITICAL_SECTION)&node->mutex);
    return locked;
#else
    if (pthread_mutex_trylock((pthread_mutex_t*)&node->mutex) == 0) {
        pthread_mutex_unlock((pthread_mutex_t*)&node->mutex);
        return 0;
    }
    return 1;
#endif
}

// Geneology Lock Bank
GeneologyLockBank* geneology_lockbank_create(void) {
    unsigned long bank_token;
    GeneologyLockBank* bank = guardian_alloc(NULL, sizeof(GeneologyLockBank), &bank_token);
    if (!bank) return NULL;
#ifdef _WIN32
    InitializeCriticalSection(&bank->bank_mutex);
#else
    pthread_mutex_init(&bank->bank_mutex, NULL);
#endif
    return bank;
}

void geneology_lockbank_destroy(GeneologyLockBank* bank) {
    if (!bank) return;
#ifdef _WIN32
    DeleteCriticalSection(&bank->bank_mutex);
#else
    pthread_mutex_destroy(&bank->bank_mutex);
#endif
    guardian_free(NULL, bank->locked_nodes);
    guardian_free(NULL, bank);
}

static int lockbank_has_node(GeneologyLockBank* bank, Node* node) {
    for (size_t i = 0; i < bank->num_locked; ++i)
        if (bank->locked_nodes[i] == node) return 1;
    return 0;
}

void geneology_lockbank_request(GeneologyLockBank* bank, Node** nodes, size_t num_nodes) {
    if (!bank || !nodes) return;
#ifdef _WIN32
    EnterCriticalSection(&bank->bank_mutex);
#else
    pthread_mutex_lock(&bank->bank_mutex);
#endif
    for (size_t i = 0; i < num_nodes; ++i) {
        Node* n = nodes[i];
        if (!n) continue;
        if (!lockbank_has_node(bank, n) && node_trylock(n)) {
            if (bank->num_locked == bank->cap_locked) {
                size_t new_cap = bank->cap_locked ? bank->cap_locked * 2 : 4;
                Node** tmp = guardian_realloc(NULL, bank->locked_nodes, new_cap * sizeof(Node*));
                if (!tmp) break;
                bank->locked_nodes = tmp;
                bank->cap_locked = new_cap;
            }
            bank->locked_nodes[bank->num_locked++] = n;
        }
    }
#ifdef _WIN32
    LeaveCriticalSection(&bank->bank_mutex);
#else
    pthread_mutex_unlock(&bank->bank_mutex);
#endif
}

int geneology_lockbank_confirm(GeneologyLockBank* bank, Node** nodes, size_t num_nodes) {
    if (!bank || !nodes) return 0;
#ifdef _WIN32
    EnterCriticalSection(&bank->bank_mutex);
#else
    pthread_mutex_lock(&bank->bank_mutex);
#endif
    int all_locked = 1;
    for (size_t i = 0; i < num_nodes; ++i) {
        if (!lockbank_has_node(bank, nodes[i])) { all_locked = 0; break; }
    }
#ifdef _WIN32
    LeaveCriticalSection(&bank->bank_mutex);
#else
    pthread_mutex_unlock(&bank->bank_mutex);
#endif
    return all_locked;
}

void geneology_lockbank_release(GeneologyLockBank* bank, Node** nodes, size_t num_nodes) {
    if (!bank || !nodes) return;
#ifdef _WIN32
    EnterCriticalSection(&bank->bank_mutex);
#else
    pthread_mutex_lock(&bank->bank_mutex);
#endif
    for (size_t i = 0; i < num_nodes; ++i) {
        Node* n = nodes[i];
        if (!n) continue;
        for (size_t j = 0; j < bank->num_locked; ++j) {
            if (bank->locked_nodes[j] == n) {
                node_unlock(n);
                bank->locked_nodes[j] = bank->locked_nodes[--bank->num_locked];
                break;
            }
        }
    }
#ifdef _WIN32
    LeaveCriticalSection(&bank->bank_mutex);
#else
    pthread_mutex_unlock(&bank->bank_mutex);
#endif
}

// Graph set operations
size_t graph_set_union(Node** a, size_t a_count, Node** b, size_t b_count, Node** out_union, size_t out_cap) {
    size_t count = 0;
    for (size_t i = 0; i < a_count && count < out_cap; ++i) out_union[count++] = a[i];
    for (size_t j = 0; j < b_count && count < out_cap; ++j) {
        int found = 0;
        for (size_t i = 0; i < a_count; ++i) if (a[i] == b[j]) { found = 1; break; }
        if (!found) out_union[count++] = b[j];
    }
    return count;
}

size_t graph_set_intersection(Node** a, size_t a_count, Node** b, size_t b_count, Node** out_inter, size_t out_cap) {
    size_t count = 0;
    for (size_t i = 0; i < a_count && count < out_cap; ++i) {
        for (size_t j = 0; j < b_count; ++j) {
            if (a[i] == b[j]) { out_inter[count++] = a[i]; break; }
        }
    }
    return count;
}

size_t graph_set_difference(Node** a, size_t a_count, Node** b, size_t b_count, Node** out_diff, size_t out_cap) {
    size_t count = 0;
    for (size_t i = 0; i < a_count && count < out_cap; ++i) {
        int found = 0;
        for (size_t j = 0; j < b_count; ++j) if (a[i] == b[j]) { found = 1; break; }
        if (!found) out_diff[count++] = a[i];
    }
    return count;
}

int graph_set_contains(Node** set, size_t set_count, Node* node) {
    for (size_t i = 0; i < set_count; ++i) if (set[i] == node) return 1;
    return 0;
}

// Node operator overrides
Node* node_add(const Node* a, const Node* b) { (void)a; (void)b; return NULL; /* TODO: implement */ }
Node* node_sub(const Node* a, const Node* b) { (void)a; (void)b; return NULL; /* TODO: implement */ }
Node* node_mul(const Node* a, const Node* b) { (void)a; (void)b; return NULL; /* TODO: implement */ }
Node* node_div(const Node* a, const Node* b) { (void)a; (void)b; return NULL; /* TODO: implement */ }
Node* node_add_scalar(const Node* a, double s) { (void)a; (void)s; return NULL; /* TODO: implement */ }
Node* node_mul_scalar(const Node* a, double s) { (void)a; (void)s; return NULL; /* TODO: implement */ }

// Geneology operator overrides
Geneology* geneology_union(const Geneology* a, const Geneology* b) { (void)a; (void)b; return NULL; /* TODO: implement */ }
Geneology* geneology_intersection(const Geneology* a, const Geneology* b) { (void)a; (void)b; return NULL; /* TODO: implement */ }
Geneology* geneology_difference(const Geneology* a, const Geneology* b) { (void)a; (void)b; return NULL; /* TODO: implement */ }
Geneology* geneology_symmetric_difference(const Geneology* a, const Geneology* b) { (void)a; (void)b; return NULL; /* TODO: implement */ }
Geneology* geneology_complement(const Geneology* a, const Geneology* universe) { (void)a; (void)universe; return NULL; /* TODO: implement */ }

