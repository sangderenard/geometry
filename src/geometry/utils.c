#include "geometry/utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

// === Utility ===
static void* grow_array(void* array, size_t elem_size, size_t* cap) {
    size_t new_cap = (*cap == 0) ? 4 : (*cap * 2);
    void* new_arr = realloc(array, new_cap * elem_size);
    if (new_arr) *cap = new_cap;
    return new_arr;
}

// === Node Data Structures ===

// All type definitions are provided in the public header.

// === Geneology Data Structure ===
typedef struct Geneology {
    Node** nodes;
    size_t num_nodes, cap_nodes;
} Geneology;

Geneology* geneology_create(void) {
    Geneology* g = (Geneology*)calloc(1, sizeof(Geneology));
    return g;
}

void geneology_destroy(Geneology* g) {
    if (!g) return;
    free(g->nodes);
    free(g);
}

void geneology_add_node(Geneology* g, Node* node) {
    if (!g || !node) return;
    if (g->num_nodes == g->cap_nodes) {
        void* tmp = grow_array(g->nodes, sizeof(Node*), &g->cap_nodes);
        if (!tmp) return;
        g->nodes = (Node**)tmp;
    }
    g->nodes[g->num_nodes++] = node;
}

void geneology_remove_node(Geneology* g, Node* node) {
    if (!g || !node) return;
    for (size_t i = 0; i < g->num_nodes; ++i) {
        if (g->nodes[i] == node) {
            g->nodes[i] = g->nodes[g->num_nodes - 1];
            g->num_nodes--;
            return;
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

// Traversal (DFS, BFS)
#include <stdbool.h>
void geneology_traverse_dfs(Geneology* g, Node* root, GeneologyVisitFn visit, void* user) {
    if (!g || !root || !visit) return;
    bool* visited = calloc(g->num_nodes, sizeof(bool));
    size_t stack_cap = 16, stack_size = 0;
    Node** stack = malloc(stack_cap * sizeof(Node*));
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
                stack_cap *= 2;
                stack = realloc(stack, stack_cap * sizeof(Node*));
            }
            stack[stack_size++] = n->forward_links[i].node;
        }
    }
    free(stack);
    free(visited);
}

void geneology_traverse_bfs(Geneology* g, Node* root, GeneologyVisitFn visit, void* user) {
    if (!g || !root || !visit) return;
    bool* visited = calloc(g->num_nodes, sizeof(bool));
    size_t queue_cap = 16, queue_size = 0, queue_head = 0;
    Node** queue = malloc(queue_cap * sizeof(Node*));
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
                queue_cap *= 2;
                queue = realloc(queue, queue_cap * sizeof(Node*));
            }
            queue[queue_size++] = n->forward_links[i].node;
        }
    }
    free(queue);
    free(visited);
}

// Stubs for search/sort
typedef int (*GeneologyNodeCmp)(const Node*, const Node*);
void geneology_sort(Geneology* g, GeneologyNodeCmp cmp) {
    // stub: implement as needed
}

Node* geneology_search(Geneology* g, int (*pred)(const Node*, void*), void* user) {
    // stub: implement as needed
    return NULL;
}

// === Node Lifecycle ===

Node* node_create(void) {
    Node* n = (Node*)calloc(1, sizeof(Node));
    static uint64_t counter = 1;
    n->uid = counter++;
#ifdef _WIN32
    InitializeCriticalSection(&n->mutex);
#else
    pthread_mutex_init(&n->mutex, NULL);
#endif
    return n;
}

void node_destroy(Node* node) {
    if (!node) return;
#ifdef _WIN32
    DeleteCriticalSection(&node->mutex);
#else
    pthread_mutex_destroy(&node->mutex);
#endif
    free(node->id);
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

// === Node API ===

static size_t node_add_relation_full(Node* node, int type, NodeForwardFn forward, NodeBackwardFn backward, const char* name, void* context) {
    if (node->num_relations == node->cap_relations) {
        void* tmp = grow_array(node->relations, sizeof(NodeRelation), &node->cap_relations);
        if (!tmp) return (size_t)-1;
        node->relations = (NodeRelation*)tmp;
    }
    NodeRelation r = {type, forward, backward, strdup(name), context};
    node->relations[node->num_relations] = r;
    return node->num_relations++;
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
    size_t idx1 = node_add_forward_link(a, b, relation);
    size_t idx2 = node_add_backward_link(b, a, relation);
    if (idx1 == (size_t)-1 || idx2 == (size_t)-1) return (size_t)-1;
    return idx1;
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
    SimpleGraph* graph = (SimpleGraph*)calloc(1, sizeof(SimpleGraph));
    graph->geneology = g;
    return graph;
}

void simplegraph_destroy(SimpleGraph* graph) {
    if (!graph) return;
    free(graph->edges);
    if (graph->feature_block) {
        // User is responsible for freeing tensor objects
        free(graph->feature_block);
    }
    if (graph->node_feature_maps) {
        for (size_t i = 0; i < graph->num_nodes; ++i) {
            SimpleGraphFeatureMap* fmap = &graph->node_feature_maps[i];
            for (size_t j = 0; j < fmap->cap_entries; ++j) {
                free(fmap->entries[j].key);
            }
            free(fmap->entries);
        }
        free(graph->node_feature_maps);
    }
    free(graph);
}

static void simplegraph_grow_edges(SimpleGraph* graph) {
    if (graph->num_edges == graph->cap_edges) {
        size_t new_cap = graph->cap_edges ? graph->cap_edges * 2 : 4;
        graph->edges = (SimpleGraphEdge*)realloc(graph->edges, new_cap * sizeof(SimpleGraphEdge));
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
        graph->feature_block = (void**)realloc(graph->feature_block, new_cap * sizeof(void*));
        graph->cap_features = new_cap;
    }
}

static void simplegraph_grow_node_maps(SimpleGraph* graph) {
    size_t g_nodes = geneology_num_nodes(graph->geneology);
    if (g_nodes > graph->cap_nodes) {
        size_t new_cap = graph->cap_nodes ? graph->cap_nodes * 2 : 8;
        if (new_cap < g_nodes) new_cap = g_nodes;
        graph->node_feature_maps = (SimpleGraphFeatureMap*)realloc(graph->node_feature_maps, new_cap * sizeof(SimpleGraphFeatureMap));
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
        SimpleGraphFeatureEntry* new_entries = calloc(new_cap, sizeof(SimpleGraphFeatureEntry));
        for (size_t i = 0; i < fmap->cap_entries; ++i) {
            if (fmap->entries[i].key) {
                size_t h = simplegraph_hash(fmap->entries[i].key, new_cap);
                while (new_entries[h].key) h = (h + 1) % new_cap;
                new_entries[h] = fmap->entries[i];
            }
        }
        free(fmap->entries);
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
        fmap->entries = calloc(fmap->cap_entries, sizeof(SimpleGraphFeatureEntry));
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
