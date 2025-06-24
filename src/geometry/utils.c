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
static void* grow_array(void* array, size_t elem_size, size_t* cap) {
    size_t new_cap = (*cap == 0) ? 4 : (*cap * 2);
    void* new_arr = guardian_realloc_simple(array, new_cap * elem_size);
    if (new_arr) *cap = new_cap;
    return new_arr;
}

// Helper to add a node to a multidimensional array (e.g., parents, children, siblings)
static void node_add_to_stencil(Node**** arr, size_t** num_arr, size_t* num_dims, size_t dim, Node* target) {
    // Grow dimensions if needed
    if (dim >= *num_dims) {
        size_t old_dims = *num_dims;
        size_t new_dims = dim + 1;
        Node*** tmp_arr = guardian_realloc_simple(*arr, new_dims * sizeof(Node**));
        size_t* tmp_num = guardian_realloc_simple(*num_arr, new_dims * sizeof(size_t));
        if (!tmp_arr || !tmp_num) {
            guardian_free_simple(tmp_arr);
            guardian_free_simple(tmp_num);
            return;
        }
        *arr = tmp_arr;
        *num_arr = tmp_num;
        for (size_t d = old_dims; d < new_dims; ++d) {
            (*arr)[d] = NULL;
            (*num_arr)[d] = 0;
        }
        *num_dims = new_dims;
    }
    // Grow array in this dimension
    size_t idx = (*num_arr)[dim];
    Node** tmp_dim = guardian_realloc_simple((*arr)[dim], (idx + 1) * sizeof(Node*));
    if (!tmp_dim) return;
    (*arr)[dim] = tmp_dim;
    (*arr)[dim][idx] = target;
    (*num_arr)[dim]++;
}

// === Geneology Basic Operations ===

Geneology* geneology_create(void) {
    return guardian_calloc_simple(1, sizeof(Geneology));
}

void geneology_destroy(Geneology* g) {
    if (!g) return;
    guardian_free_simple(g->nodes);
    guardian_free_simple(g);
}

void geneology_add_node(Geneology* g, Node* node) {
    if (!g || !node) return;
    if (g->num_nodes == g->cap_nodes) {
        size_t new_cap = g->cap_nodes ? g->cap_nodes * 2 : 8;
        Node** tmp = realloc(g->nodes, new_cap * sizeof(Node*));
        if (!tmp) return;
        g->nodes = tmp;
        g->cap_nodes = new_cap;
    }
    g->nodes[g->num_nodes++] = node;
}

void geneology_remove_node(Geneology* g, Node* node) {
    if (!g || !node) return;
    for (size_t i = 0; i < g->num_nodes; ++i) {
        if (g->nodes[i] == node) {
            memmove(&g->nodes[i], &g->nodes[i + 1], (g->num_nodes - i - 1) * sizeof(Node*));
            g->num_nodes--;
            break;
        }
    }
}

size_t geneology_num_nodes(const Geneology* g) {
    return g ? g->num_nodes : 0;
}

Node* geneology_get_node(const Geneology* g, size_t idx) {
    if (!g || idx >= g->num_nodes) return NULL;
    return g->nodes[idx];
}

// === Node Data Structures ===

// All type definitions are provided in the public header.

// Traversal (DFS, BFS)
#include <stdbool.h>
void geneology_traverse_dfs(Geneology* g, Node* root, GeneologyVisitFn visit, void* user) {
    if (!g || !root || !visit) return;
    bool* visited = guardian_calloc_simple(g->num_nodes, sizeof(bool));
    size_t stack_cap = 16, stack_size = 0;
    Node** stack = guardian_malloc_simple(stack_cap * sizeof(Node*));
    stack[stack_size++] = root;
    while (stack_size) {
        Node* n = stack[--stack_size];
        size_t idx = 0;
        for (; idx < g->num_nodes; ++idx) if (g->nodes[idx] == n) break;
        if (idx == g->num_nodes || visited[idx]) continue;
        visited[idx] = true;
        visit(n, user);
        for (size_t i = 0; i < n->num_forward_links; ++i) {
            if (stack_size == stack_cap) {
                size_t new_cap = stack_cap * 2;
                Node** tmp = guardian_realloc_simple(stack, new_cap * sizeof(Node*));
                if (!tmp) break;
                stack = tmp;
                stack_cap = new_cap;
            }
            stack[stack_size++] = n->forward_links[i].node;
        }
    }
    guardian_free_simple(stack);
    guardian_free_simple(visited);
}

void geneology_traverse_bfs(Geneology* g, Node* root, GeneologyVisitFn visit, void* user) {
    if (!g || !root || !visit) return;
    bool* visited = guardian_calloc_simple(g->num_nodes, sizeof(bool));
    size_t queue_cap = 16, queue_size = 0, queue_head = 0;
    Node** queue = guardian_malloc_simple(queue_cap * sizeof(Node*));
    queue[queue_size++] = root;
    while (queue_head < queue_size) {
        Node* n = queue[queue_head++];
        size_t idx = 0;
        for (; idx < g->num_nodes; ++idx) if (g->nodes[idx] == n) break;
        if (idx == g->num_nodes || visited[idx]) continue;
        visited[idx] = true;
        visit(n, user);
        for (size_t i = 0; i < n->num_forward_links; ++i) {
            if (queue_size == queue_cap) {
                size_t new_cap = queue_cap * 2;
                Node** tmp = guardian_realloc_simple(queue, new_cap * sizeof(Node*));
                if (!tmp) break;
                queue = tmp;
                queue_cap = new_cap;
            }
            queue[queue_size++] = n->forward_links[i].node;
        }
    }
    guardian_free_simple(queue);
    guardian_free_simple(visited);
}

// Stubs for search/sort
typedef int (*GeneologyNodeCmp)(const Node*, const Node*);
void geneology_sort(Geneology* g, GeneologyNodeCmp cmp) {
    if (!g || !cmp || g->num_nodes < 2) return;
    for (size_t i = 0; i < g->num_nodes - 1; ++i) {
        for (size_t j = i + 1; j < g->num_nodes; ++j) {
            if (cmp(g->nodes[i], g->nodes[j]) > 0) {
                Node* tmp = g->nodes[i];
                g->nodes[i] = g->nodes[j];
                g->nodes[j] = tmp;
            }
        }
    }
}

Node* geneology_search(Geneology* g, int (*pred)(const Node*, void*), void* user) {
    if (!g || !pred) return NULL;
    for (size_t i = 0; i < g->num_nodes; ++i) {
        if (pred(g->nodes[i], user)) return g->nodes[i];
    }
    return NULL;
}

// === Node Lifecycle ===

Node* node_create(void) {
    Node* n = (Node*)guardian_calloc_simple(1, sizeof(Node));
    static uint64_t counter = 1;
    n->uid = counter++;
#ifdef _WIN32
    InitializeCriticalSection(&n->mutex);
#else
    pthread_mutex_init(&n->mutex, NULL);
#endif
    // Initialize multidimensional stencil arrays to NULL/0
    n->parents = NULL;
    n->num_parents = NULL;
    n->num_dims_parents = 0;
    n->children = NULL;
    n->num_children = NULL;
    n->num_dims_children = 0;
    n->left_siblings = NULL;
    n->num_left_siblings = NULL;
    n->num_dims_left_siblings = 0;
    n->right_siblings = NULL;
    n->num_right_siblings = NULL;
    n->num_dims_right_siblings = 0;
    return n;
}

void node_destroy(Node* node) {
    if (!node) return;
#ifdef _WIN32
    DeleteCriticalSection(&node->mutex);
#else
    pthread_mutex_destroy(&node->mutex);
#endif
    guardian_free_simple(node->id);
    guardian_free_simple(node->relations);
    if (node->features) {
        for (size_t i = 0; i < node->num_features; ++i) {
            guardian_free_simple(node->features[i]);
        }
    }
    guardian_free_simple(node->features);
    guardian_free_simple(node->exposures);
    guardian_free_simple(node->forward_links);
    guardian_free_simple(node->backward_links);
    // Free multidimensional stencil arrays
    if (node->parents) {
        for (size_t d = 0; d < node->num_dims_parents; ++d) guardian_free_simple(node->parents[d]);
        guardian_free_simple(node->parents);
        guardian_free_simple(node->num_parents);
    }
    if (node->children) {
        for (size_t d = 0; d < node->num_dims_children; ++d) guardian_free_simple(node->children[d]);
        guardian_free_simple(node->children);
        guardian_free_simple(node->num_children);
    }
    if (node->left_siblings) {
        for (size_t d = 0; d < node->num_dims_left_siblings; ++d) guardian_free_simple(node->left_siblings[d]);
        guardian_free_simple(node->left_siblings);
        guardian_free_simple(node->num_left_siblings);
    }
    if (node->right_siblings) {
        for (size_t d = 0; d < node->num_dims_right_siblings; ++d) guardian_free_simple(node->right_siblings[d]);
        guardian_free_simple(node->right_siblings);
        guardian_free_simple(node->num_right_siblings);
    }
    guardian_free_simple(node);
}

// === Node API ===

static size_t node_add_relation_full(Node* node, int type, NodeForwardFn forward, NodeBackwardFn backward, const char* name, void* context) {
    if (node->num_relations == node->cap_relations) {
        void* tmp = grow_array(node->relations, sizeof(NodeRelation), &node->cap_relations);
        if (!tmp) return (size_t)-1;
        node->relations = (NodeRelation*)tmp;
    }
    NodeRelation r = {type, forward, backward, strdup(name), context};
    size_t idx = node->num_relations;
    node->relations[idx] = r;
    node->num_relations++;

    // Default to 0th dimension for now
    size_t dim = 0;
    Node* partner = context ? (Node*)context : NULL;
    switch (type) {
        case EDGE_PARENT_CHILD_CONTIGUOUS:
            if (partner) node_add_to_stencil(&node->children, &node->num_children, &node->num_dims_children, dim, partner);
            break;
        case EDGE_CHILD_PARENT_CONTIGUOUS:
            if (partner) node_add_to_stencil(&node->parents, &node->num_parents, &node->num_dims_parents, dim, partner);
            break;
        case EDGE_SIBLING_LEFT_TO_RIGHT_CONTIGUOUS:
            if (partner) node_add_to_stencil(&node->right_siblings, &node->num_right_siblings, &node->num_dims_right_siblings, dim, partner);
            break;
        case EDGE_SIBLING_RIGHT_TO_LEFT_CONTIGUOUS:
            if (partner) node_add_to_stencil(&node->left_siblings, &node->num_left_siblings, &node->num_dims_left_siblings, dim, partner);
            break;
        case EDGE_SIBLING_SIBLING_NONCONTIGUOUS:
            if (partner) {
                node_add_to_stencil(&node->left_siblings, &node->num_left_siblings, &node->num_dims_left_siblings, dim, partner);
                node_add_to_stencil(&node->right_siblings, &node->num_right_siblings, &node->num_dims_right_siblings, dim, partner);
            }
            break;
        case EDGE_LINEAGE_NONCONTIGUOUS:
            if (partner) {
                node_add_to_stencil(&node->parents, &node->num_parents, &node->num_dims_parents, dim, partner);
                node_add_to_stencil(&node->children, &node->num_children, &node->num_dims_children, dim, partner);
            }
            break;
        case EDGE_ARBITRARY:
        default:
            break;
    }
    return idx;
}

size_t node_add_relation(Node* node, int type, NodeForwardFn forward, NodeBackwardFn backward) {
    return node_add_relation_full(node, type, forward, backward, "", NULL);
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

// === DAG Linkage ===

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

size_t node_add_bidirectional_link(Node* a, Node* b, int relation) {
    if (!a || !b) return (size_t)-1;
    // Add forward link for a -> b
    size_t idx1 = node_add_forward_link(a, b, relation);
    // Add to a's stencil (children, right_siblings, etc.)
    switch (relation) {
        case EDGE_PARENT_CHILD_CONTIGUOUS:
            node_add_to_stencil(&a->children, &a->num_children, &a->num_dims_children, 0, b);
            break;
        case EDGE_SIBLING_LEFT_TO_RIGHT_CONTIGUOUS:
            node_add_to_stencil(&a->right_siblings, &a->num_right_siblings, &a->num_dims_right_siblings, 0, b);
            break;
        default:
            break;
    }
    // Add backward link for b -> a (inverse relation)
    int inverse_relation = geneology_invert_relation(relation);
    if (inverse_relation >= (int)b->num_relations) inverse_relation = 0;
    size_t idx2 = node_add_backward_link(b, a, inverse_relation);
    // Add to b's stencil (parents, left_siblings, etc.)
    switch (inverse_relation) {
        case EDGE_CHILD_PARENT_CONTIGUOUS:
            node_add_to_stencil(&b->parents, &b->num_parents, &b->num_dims_parents, 0, a);
            break;
        case EDGE_SIBLING_RIGHT_TO_LEFT_CONTIGUOUS:
            node_add_to_stencil(&b->left_siblings, &b->num_left_siblings, &b->num_dims_left_siblings, 0, a);
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

Node* node_split(const Node* src) {
    if (!src) return NULL;
    Node* n = node_create();
    for (size_t i = 0; i < src->num_features; ++i)
        node_add_feature(n, src->features[i]);
    for (size_t i = 0; i < src->num_exposures; ++i)
        node_add_exposure(n, src->exposures[i].produce, src->exposures[i].reverse);
    for (size_t i = 0; i < src->num_relations; ++i)
        node_add_relation_full(n,
                              src->relations[i].type,
                              src->relations[i].forward,
                              src->relations[i].backward,
                              src->relations[i].name,
                              src->relations[i].context);
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

// === Additional Accessors ===

NodeRelation* node_get_relation(const Node* node, size_t index) {
    if (!node || index >= node->num_relations) return NULL;
    return &node->relations[index];
}

const char* node_get_feature(const Node* node, size_t index) {
    if (!node || index >= node->num_features) return NULL;
    return node->features[index];
}

NodeExposure* node_get_exposure(const Node* node, size_t index) {
    if (!node || index >= node->num_exposures) return NULL;
    return &node->exposures[index];
}

const NodeLink* node_get_forward_link(const Node* node, size_t index) {
    if (!node || index >= node->num_forward_links) return NULL;
    return &node->forward_links[index];
}

const NodeLink* node_get_backward_link(const Node* node, size_t index) {
    if (!node || index >= node->num_backward_links) return NULL;
    return &node->backward_links[index];
}

// === Recursive Traversal ===

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

// --- SimpleGraph implementation ---
SimpleGraph* simplegraph_create(Geneology* g) {
    SimpleGraph* graph = (SimpleGraph*)guardian_calloc_simple(1, sizeof(SimpleGraph));
    graph->geneology = g;
    return graph;
}

void simplegraph_destroy(SimpleGraph* graph) {
    if (!graph) return;
    guardian_free_simple(graph->edges);
    if (graph->feature_block) {
        // User is responsible for freeing tensor objects
        guardian_free_simple(graph->feature_block);
    }
    if (graph->node_feature_maps) {
        for (size_t i = 0; i < graph->num_nodes; ++i) {
            SimpleGraphFeatureMap* fmap = &graph->node_feature_maps[i];
            for (size_t j = 0; j < fmap->cap_entries; ++j) {
                guardian_free_simple(fmap->entries[j].key);
            }
            guardian_free_simple(fmap->entries);
        }
        guardian_free_simple(graph->node_feature_maps);
    }
    guardian_free_simple(graph);
}

static void simplegraph_grow_edges(SimpleGraph* graph) {
    if (graph->num_edges == graph->cap_edges) {
        size_t new_cap = graph->cap_edges ? graph->cap_edges * 2 : 4;
        SimpleGraphEdge* tmp = guardian_realloc_simple(graph->edges, new_cap * sizeof(SimpleGraphEdge));
        if (!tmp) return;
        graph->edges = tmp;
        graph->cap_edges = new_cap;
    }
}

void simplegraph_add_edge(SimpleGraph* graph, Node* src, Node* dst, SimpleGraphEdgeType type, int relation) {
    if (!graph || !src || !dst) return;
    simplegraph_grow_edges(graph);
    SimpleGraphEdge e = {src, dst, type, relation};
    graph->edges[graph->num_edges++] = e;
    // Optionally, also add to geneology if needed
}

static void simplegraph_grow_features(SimpleGraph* graph) {
    if (graph->num_features == graph->cap_features) {
        size_t new_cap = graph->cap_features ? graph->cap_features * 2 : 8;
        void** tmp = guardian_realloc_simple(graph->feature_block, new_cap * sizeof(void*));
        if (!tmp) return;
        graph->feature_block = tmp;
        graph->cap_features = new_cap;
    }
}

static void simplegraph_grow_node_maps(SimpleGraph* graph) {
    size_t g_nodes = geneology_num_nodes(graph->geneology);
    if (g_nodes > graph->cap_nodes) {
        size_t new_cap = graph->cap_nodes ? graph->cap_nodes * 2 : 8;
        if (new_cap < g_nodes) new_cap = g_nodes;
        SimpleGraphFeatureMap* tmp = guardian_realloc_simple(graph->node_feature_maps, new_cap * sizeof(SimpleGraphFeatureMap));
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

void simplegraph_add_feature(SimpleGraph* graph, Node* node, const char* feature_name, void* tensor_ptr) {
    if (!graph || !node || !feature_name) return;
    simplegraph_grow_features(graph);
    simplegraph_grow_node_maps(graph);
    // Find node index in geneology
    size_t idx = 0;
    for (; idx < graph->num_nodes; ++idx) if (geneology_get_node(graph->geneology, idx) == node) break;
    if (idx == graph->num_nodes) return;
    SimpleGraphFeatureMap* fmap = &graph->node_feature_maps[idx];
    // Grow feature map if needed
    if (fmap->num_entries * 2 >= fmap->cap_entries) {
        size_t new_cap = fmap->cap_entries ? fmap->cap_entries * 2 : 8;
        SimpleGraphFeatureEntry* new_entries = guardian_calloc_simple(new_cap, sizeof(SimpleGraphFeatureEntry));
        for (size_t i = 0; i < fmap->cap_entries; ++i) {
            if (fmap->entries[i].key) {
                size_t h = simplegraph_hash(fmap->entries[i].key, new_cap);
                while (new_entries[h].key) h = (h + 1) % new_cap;
                new_entries[h] = fmap->entries[i];
            }
        }
        guardian_free_simple(fmap->entries);
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
        fmap->entries = guardian_calloc_simple(fmap->cap_entries, sizeof(SimpleGraphFeatureEntry));
    }
    fmap->entries[h].key = strdup(feature_name);
    fmap->entries[h].value = tensor_ptr;
    fmap->num_entries++;
    graph->feature_block[graph->num_features++] = tensor_ptr;
}

void* simplegraph_get_feature(SimpleGraph* graph, Node* node, const char* feature_name) {
    if (!graph || !node || !feature_name) return NULL;
    simplegraph_grow_node_maps(graph);
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

void simplegraph_forward(SimpleGraph* graph) {
    if (!graph) return;
    for (size_t i = 0; i < graph->num_edges; ++i) {
        Node* src = graph->edges[i].src;
        Node* dst = graph->edges[i].dst;
        int rel = graph->edges[i].relation;
        if (src && rel >= 0 && (size_t)rel < src->num_relations) {
            NodeRelation* r = &src->relations[rel];
            if (r->forward) r->forward(dst, NULL); // or pass data as needed
        }
    }
}

void simplegraph_backward(SimpleGraph* graph) {
    if (!graph) return;
    for (size_t i = graph->num_edges; i-- > 0;) {
        Node* src = graph->edges[i].src;
        Node* dst = graph->edges[i].dst;
        int rel = graph->edges[i].relation;
        if (src && rel >= 0 && (size_t)rel < src->num_relations) {
            NodeRelation* r = &src->relations[rel];
            if (r->backward) r->backward(dst, NULL); // or pass data as needed
        }
    }
}

// --- DAG Manifest Structures Implementation ---
#include <stdlib.h>
#include <string.h>

Dag* dag_create(void) {
    Dag* dag = (Dag*)calloc(1, sizeof(Dag));
    return dag;
}

void dag_destroy(Dag* dag) {
    if (!dag) return;
    for (size_t i = 0; i < dag->num_manifests; ++i) {
        DagManifest* manifest = &dag->manifests[i];
        for (size_t l = 0; l < manifest->num_levels; ++l) {
            DagManifestLevel* level = &manifest->levels[l];
            for (size_t m = 0; m < level->num_mappings; ++m) {
                DagManifestMapping* mapping = &level->mappings[m];
                free(mapping->inputs);
                free(mapping->outputs);
            }
            free(level->mappings);
        }
        free(manifest->levels);
    }
    free(dag->manifests);
    free(dag);
}
void dag_add_manifest(Dag* dag, DagManifest* manifest) {
    if (!dag || !manifest) return;
    if (dag->num_manifests == dag->cap_manifests) {
        size_t new_cap = dag->cap_manifests ? dag->cap_manifests * 2 : 4;
        DagManifest* tmp = (DagManifest*)realloc(dag->manifests, new_cap * sizeof(DagManifest));
        if (!tmp) return;
        dag->manifests = tmp;
        dag->cap_manifests = new_cap;
    }
    dag->manifests[dag->num_manifests++] = *manifest;
}


// --- Emergence Implementation ---
Emergence* emergence_create(void) {
    Emergence* e = (Emergence*)calloc(1, sizeof(Emergence));
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
    free(e);
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
    NeuralNetwork* nn = (NeuralNetwork*)calloc(1, sizeof(NeuralNetwork));
    return nn;
}

void neuralnetwork_destroy(NeuralNetwork* nn) {
    if (!nn) return;
    for (size_t d = 0; d < nn->num_dags; ++d) {
        dag_destroy(nn->dags[d]);
        for (size_t s = 0; s < nn->num_steps[d]; ++s) {
            free(nn->steps[d][s]);
        }
    }
    free(nn);
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
        step = (NeuralNetworkStep*)calloc(1, sizeof(NeuralNetworkStep));
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
        NeighborEntry* tmp = guardian_realloc_simple(map->entries, new_cap * sizeof(NeighborEntry));
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
            free(node->neighbor_map.entries[i].label.label);
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
            free(node->neighbor_map.entries[i].label.label);
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
    char* name = (char*)guardian_malloc_simple(128);
    snprintf(name, 128, "%s_%zud_r%zu_pole%zu", type_str, dims, radius, pole_index);
    return name;
}

// --- StencilSet helpers ---
StencilSet* stencilset_wrap_single(GeneralStencil* stencil) {
    if (!stencil) return NULL;
    StencilSet* set = (StencilSet*)calloc(1, sizeof(StencilSet));
    set->stencils = (GeneralStencil**)calloc(1, sizeof(GeneralStencil*));
    set->stencils[0] = stencil;
    set->count = 1;
    set->cap = 1;
    // Allocate 1x1 relation matrix
    set->relation = (StencilRelation**)calloc(1, sizeof(StencilRelation*));
    set->relation[0] = (StencilRelation*)calloc(1, sizeof(StencilRelation));
    set->relation[0][0].type = STENCIL_RELATION_SAME_COORDINATE_SET;
    set->relation[0][0].context = NULL;
    set->relation[0][0].similarity_fn = NULL;
    return set;
}

void stencilset_init_fully_connected(StencilSet* set, int default_relation_type) {
    if (!set || set->count == 0) return;
    // Free any existing relation matrix
    if (set->relation) {
        for (size_t i = 0; i < set->count; ++i) free(set->relation[i]);
        free(set->relation);
    }
    set->relation = (StencilRelation**)calloc(set->count, sizeof(StencilRelation*));
    for (size_t i = 0; i < set->count; ++i) {
        set->relation[i] = (StencilRelation*)calloc(set->count, sizeof(StencilRelation));
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
    QuaternionHistory* hist = (QuaternionHistory*)guardian_calloc_simple(1, sizeof(QuaternionHistory));
    hist->window_size = window_size;
    hist->cap = window_size ? window_size : 16;
    hist->quats = (Quaternion*)guardian_calloc_simple(hist->cap, sizeof(Quaternion));
    hist->decay_fn = decay_fn;
    hist->decay_user_data = user_data;
    return hist;
}

void quaternion_history_destroy(QuaternionHistory* hist) {
    if (!hist) return;
    guardian_free_simple(hist->quats);
    guardian_free_simple(hist);
}

void quaternion_history_add(QuaternionHistory* hist, Quaternion q) {
    if (!hist) return;
    if (hist->window_size && hist->count == hist->window_size) {
        // Move window left
        memmove(hist->quats, hist->quats + 1, (hist->count - 1) * sizeof(Quaternion));
        hist->quats[hist->count - 1] = q;
    } else {
        if (hist->count == hist->cap) {
            size_t new_cap = hist->cap * 2;
            Quaternion* tmp = guardian_realloc_simple(hist->quats, new_cap * sizeof(Quaternion));
            if (!tmp) return;
            hist->quats = tmp;
            hist->cap = new_cap;
        }
        hist->quats[hist->count++] = q;
    }
}

// Simple weighted sum aggregate (normalize at end)
Quaternion quaternion_history_aggregate(const QuaternionHistory* hist) {
    Quaternion sum = {0, 0, 0, 0};
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
    double norm = sqrt(sum.w*sum.w + sum.x*sum.x + sum.y*sum.y + sum.z*sum.z);
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
    return (Quaternion){a.w + b.w, a.x + b.x, a.y + b.y, a.z + b.z};
}

Quaternion quaternion_sub(Quaternion a, Quaternion b) {
    return (Quaternion){a.w - b.w, a.x - b.x, a.y - b.y, a.z - b.z};
}

Quaternion quaternion_scale(Quaternion q, double s) {
    return (Quaternion){q.w * s, q.x * s, q.y * s, q.z * s};
}

Quaternion quaternion_div(Quaternion q, double s) {
    return (Quaternion){q.w / s, q.x / s, q.y / s, q.z / s};
}

Quaternion quaternion_conjugate(Quaternion q) {
    return (Quaternion){q.w, -q.x, -q.y, -q.z};
}

Quaternion quaternion_mul(Quaternion a, Quaternion b) {
    return (Quaternion){
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
    };
}

double quaternion_dot(Quaternion a, Quaternion b) {
    return a.w*b.w + a.x*b.x + a.y*b.y + a.z*b.z;
}

double quaternion_norm(Quaternion q) {
    return sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
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
        Quaternion result = quaternion_add(quaternion_scale(a, 1-t), quaternion_scale(b, t));
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
    double s = sqrt(1 - q.w*q.w);
    if (s < 1e-8) {
        axis_out[0] = 1; axis_out[1] = 0; axis_out[2] = 0;
    } else {
        axis_out[0] = q.x / s;
        axis_out[1] = q.y / s;
        axis_out[2] = q.z / s;
    }
}

Quaternion quaternion_from_axis_angle(const double* axis, double angle) {
    double s = sin(angle/2);
    return (Quaternion){cos(angle/2), axis[0]*s, axis[1]*s, axis[2]*s};
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
    } else {
        if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
            double s = 2.0 * sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]);
            q.w = (m[2][1] - m[1][2]) / s;
            q.x = 0.25 * s;
            q.y = (m[0][1] + m[1][0]) / s;
            q.z = (m[0][2] + m[2][0]) / s;
        } else if (m[1][1] > m[2][2]) {
            double s = 2.0 * sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]);
            q.w = (m[0][2] - m[2][0]) / s;
            q.x = (m[0][1] + m[1][0]) / s;
            q.y = 0.25 * s;
            q.z = (m[1][2] + m[2][1]) / s;
        } else {
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
        if (quaternion_angle_between(seq[i-1], seq[i]) > tol) return 0;
    }
    return 1;
}

int quaternion_is_gimbal_lock(Quaternion q, double tol) {
    // Gimbal lock if pitch is near +/-90 degrees
    double roll, pitch, yaw;
    quaternion_to_euler(q, &roll, &pitch, &yaw);
    return fabs(fabs(pitch) - M_PI/2) < tol;
}

int quaternion_history_is_smooth(const QuaternionHistory* hist, double tol) {
    if (!hist || hist->count < 2) return 1;
    return quaternion_is_continuous(hist->quats, hist->count, tol);
}

int quaternion_history_has_flips(const QuaternionHistory* hist, double tol) {
    if (!hist || hist->count < 2) return 0;
    for (size_t i = 1; i < hist->count; ++i) {
        if (quaternion_angle_between(hist->quats[i-1], hist->quats[i]) > tol) return 1;
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
    GeneologyLockBank* bank = guardian_calloc_simple(1, sizeof(GeneologyLockBank));
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
    guardian_free_simple(bank->locked_nodes);
    guardian_free_simple(bank);
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
                Node** tmp = guardian_realloc_simple(bank->locked_nodes, new_cap * sizeof(Node*));
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

