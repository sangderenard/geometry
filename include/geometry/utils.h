#ifndef GEOMETRY_UTILS_H
#define GEOMETRY_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#ifdef _WIN32
#include <windows.h>
typedef CRITICAL_SECTION node_mutex_t;
#else
#include <pthread.h>
typedef pthread_mutex_t node_mutex_t;
#endif

#include "geometry/stencil.h"
#include "geometry/dag.h"

struct Node;

typedef void (*NodeForwardFn)(struct Node* self, void* out);
typedef void (*NodeBackwardFn)(struct Node* self, void* grad);

typedef struct {
    int type;
    NodeForwardFn forward;
    NodeBackwardFn backward;
    char* name;
    void* context;
} NodeRelation;

typedef void (*NodeProduceFn)(struct Node* self, void* product);
typedef void (*NodeReverseFn)(struct Node* self, void* product);

typedef struct {
    NodeProduceFn produce;
    NodeReverseFn reverse;
} NodeExposure;

typedef struct {
    struct Node* node;
    int relation;
} NodeLink;

// --- StencilSet abstraction ---
// Relationship type enum
typedef enum {
    STENCIL_RELATION_SAME_COORDINATE_SET = 0,
    STENCIL_RELATION_DIFFERENT_COORDINATE_SET = 1,
    STENCIL_RELATION_ORTHOGONAL = 2,
    STENCIL_RELATION_COSINE_SIMILARITY = 3,
    STENCIL_RELATION_NODE_SHARING = 4,
    STENCIL_RELATION_CUSTOM = 100
} StencilRelationType;

// Relationship struct for each pair
typedef struct StencilRelation {
    int type; // StencilRelationType
    void* context; // e.g. node set, parameters
    double (*similarity_fn)(const GeneralStencil*, const GeneralStencil*, void* context); // optional
} StencilRelation;

typedef struct StencilSet {
    GeneralStencil** stencils;
    size_t count, cap;
    // Relationship matrix: relation[i][j] describes the relationship from stencil i to j
    StencilRelation** relation;
} StencilSet;

// Helper to wrap a single stencil as a set
StencilSet* stencilset_wrap_single(GeneralStencil* stencil);
// Initialize a fully connected relationship graph for the set (all-to-all, with a default rule)
void stencilset_init_fully_connected(StencilSet* set, int default_relation_type);

// Negotiate a bond between two nodes at given stencil pole indices
// Returns the relationship type (enum), fills out relation if not NULL
int stencilset_negotiate_bond(const StencilSet* a, size_t pole_a, const StencilSet* b, size_t pole_b, StencilRelation* out_relation);

// A label for a neighbor relationship (can be explicit or auto-generated)
typedef struct NeighborLabel {
    char* label;           // e.g. "parent", "child", "neighbor_0", etc.
    size_t pole_index;     // Index in the stencil (if any)
} NeighborLabel;

// A neighbor entry: maps a label/pole to a neighbor node
typedef struct NeighborEntry {
    NeighborLabel label;
    struct Node* neighbor;
} NeighborEntry;

// A flexible neighbor map for a node
typedef struct NeighborMap {
    NeighborEntry* entries;
    size_t count, cap;
    // StencilSet for neighbor arrangement (can be a set of stencils, not just one)
    StencilSet* stencil_set; // The set of stencils defining the neighbor arrangement (optional)
} NeighborMap;

// --- Quaternion and QuaternionHistory ---
typedef struct Quaternion {
    double w, x, y, z;
} Quaternion;

typedef double (*QuaternionDecayFn)(size_t idx, size_t window_size, void* user_data);

typedef struct QuaternionHistory {
    Quaternion* quats;         // Array of quaternions
    size_t count;              // Number of valid quaternions
    size_t cap;                // Allocated capacity
    size_t window_size;        // Max window size (0 = unlimited)
    QuaternionDecayFn decay_fn;// Optional decay function (NULL = moving window)
    void* decay_user_data;     // Optional user data for decay
} QuaternionHistory;

// Quaternion history API
QuaternionHistory* quaternion_history_create(size_t window_size, QuaternionDecayFn decay_fn, void* user_data);
void quaternion_history_destroy(QuaternionHistory* hist);
void quaternion_history_add(QuaternionHistory* hist, Quaternion q);
Quaternion quaternion_history_aggregate(const QuaternionHistory* hist);

// --- Quaternion Operators and Orientation Validators ---
// Basic arithmetic
Quaternion quaternion_add(Quaternion a, Quaternion b);
Quaternion quaternion_sub(Quaternion a, Quaternion b);
Quaternion quaternion_mul(Quaternion a, Quaternion b); // Hamilton product
Quaternion quaternion_scale(Quaternion q, double s);
Quaternion quaternion_div(Quaternion q, double s);

// Properties and transforms
Quaternion quaternion_conjugate(Quaternion q);
Quaternion quaternion_inverse(Quaternion q);
double quaternion_norm(Quaternion q);
Quaternion quaternion_normalize(Quaternion q);
double quaternion_dot(Quaternion a, Quaternion b);

// Interpolation
Quaternion quaternion_slerp(Quaternion a, Quaternion b, double t);

// Conversion
void quaternion_to_axis_angle(Quaternion q, double* axis_out, double* angle_out);
Quaternion quaternion_from_axis_angle(const double* axis, double angle);
void quaternion_to_euler(Quaternion q, double* roll, double* pitch, double* yaw);
Quaternion quaternion_from_euler(double roll, double pitch, double yaw);
void quaternion_to_matrix(Quaternion q, double m[3][3]);
Quaternion quaternion_from_matrix(const double m[3][3]);

// Orientation validators
int quaternion_is_normalized(Quaternion q, double tol);
double quaternion_angle_between(Quaternion a, Quaternion b);
int quaternion_is_continuous(const Quaternion* seq, size_t count, double tol);
int quaternion_is_gimbal_lock(Quaternion q, double tol);

// Quaternion history evaluators
int quaternion_history_is_smooth(const QuaternionHistory* hist, double tol);
int quaternion_history_has_flips(const QuaternionHistory* hist, double tol);
Quaternion quaternion_history_average(const QuaternionHistory* hist);

typedef struct Node {
    char* id;
    unsigned long long uid;
    size_t activation_count;
    double activation_sum;
    double activation_sq_sum;

    // Multidimensional stencil arrays for relationships
    struct Node*** parents;      // [dim][index] array of parent pointers
    size_t* num_parents;        // [dim] number of parents per dimension
    size_t num_dims_parents;    // number of parent dimensions

    struct Node*** children;     // [dim][index] array of child pointers
    size_t* num_children;        // [dim] number of children per dimension
    size_t num_dims_children;    // number of child dimensions

    struct Node*** left_siblings;  // [dim][index] array of left sibling pointers
    size_t* num_left_siblings;     // [dim] number of left siblings per dimension
    size_t num_dims_left_siblings; // number of left sibling dimensions

    struct Node*** right_siblings; // [dim][index] array of right sibling pointers
    size_t* num_right_siblings;    // [dim] number of right siblings per dimension
    size_t num_dims_right_siblings;// number of right sibling dimensions

    // Canonical relationship structure: links only
    NodeLink* forward_links;
    NodeLink* backward_links;
    size_t num_forward_links, cap_forward_links;
    size_t num_backward_links, cap_backward_links;

    NodeRelation* relations;
    size_t num_relations, cap_relations;

    char** features;
    size_t num_features, cap_features;

    NodeExposure* exposures;
    size_t num_exposures, cap_exposures;

    node_mutex_t mutex;

    struct Emergence* emergence;

    NeighborMap neighbor_map; // Flexible, labeled neighbor mapping

    QuaternionHistory* orientation_history; // Optional orientation history
} Node;

// --- Emergence structure for node-level adaptation ---
typedef struct Emergence Emergence;

struct Emergence {
    // Statistics
    double activation_sum;
    double activation_sq_sum;
    size_t activation_count;
    uint64_t last_global_step;
    uint64_t last_timestamp;

    // Thread lock for parallel split
    node_mutex_t thread_lock;
    int is_locked;

    // Decision hooks
    int (*should_split)(Emergence* self, Node* node);
    int (*should_apoptose)(Emergence* self, Node* node);
    int (*should_metastasize)(Emergence* self, Node* node);

    // Action hooks
    void (*split)(Emergence* self, Node* node);
    void (*apoptose)(Emergence* self, Node* node);
    void (*metastasize)(Emergence* self, Node* node);

    // User data for custom policies
    void* user_data;
};

Emergence* emergence_create(void);
void emergence_destroy(Emergence* e);
void emergence_lock(Emergence* e);
void emergence_release(Emergence* e);
void emergence_resolve(Emergence* e);
void emergence_update(Emergence* e, Node* node, double activation, uint64_t global_step, uint64_t timestamp);

// --- Geneology structure ---

// --- Path hash dictionary for relationship path caching ---
// This dictionary will map a path hash (e.g., from a path vector or encoding)
// to a relationship descriptor or cached result. Use a suitable hash map structure.
// Implementation and type details to be defined later.
typedef struct PathHashDict PathHashDict;

typedef struct Geneology {
    Node** nodes;
    size_t num_nodes, cap_nodes;
    // Path hash dictionary for fast relationship queries
    PathHashDict* path_hash_dict; // Guidance: implement as a hash map from path hash to relationship info
} Geneology;

Geneology* geneology_create(void);
void geneology_destroy(Geneology* g);
void geneology_add_node(Geneology* g, Node* node);
void geneology_remove_node(Geneology* g, Node* node);
size_t geneology_num_nodes(const Geneology* g);
Node* geneology_get_node(const Geneology* g, size_t idx);

// --- Additional Geneology faculties ---
void geneology_merge(Geneology* dest, const Geneology* src);
void geneology_find_ancestors(const Geneology* g, const Node* node, Node** out, size_t* out_count);
void geneology_find_descendants(const Geneology* g, const Node* node, Node** out, size_t* out_count);
void geneology_extract_lineage(const Geneology* g, const Node* node, Node** out, size_t* out_count);
void geneology_extract_slice_2d(const Geneology* g, const Node* root, size_t gen_start, size_t gen_end, size_t sib_start, size_t sib_end, Node** out, size_t* out_count);
Node* geneology_clone_subtree(const Geneology* g, const Node* node);

// Traversal (DFS, BFS)
typedef void (*GeneologyVisitFn)(Node* node, void* user);
void geneology_traverse_dfs(Geneology* g, Node* root, GeneologyVisitFn visit, void* user);
void geneology_traverse_bfs(Geneology* g, Node* root, GeneologyVisitFn visit, void* user);

// Stubs for search/sort
void geneology_sort(Geneology* g, int (*cmp)(const Node*, const Node*));
Node* geneology_search(Geneology* g, int (*pred)(const Node*, void*), void* user);

Node* node_create(void);
void node_destroy(Node* node);
Node* node_split(const Node* src);
int node_should_split(Node* n);
void node_record_activation(Node* n, double act);

size_t node_add_relation(Node* node, int type, NodeForwardFn forward, NodeBackwardFn backward);
NodeRelation* node_get_relation(const Node* node, size_t index);

size_t node_add_feature(Node* node, const char* feature);
const char* node_get_feature(const Node* node, size_t index);

size_t node_add_exposure(Node* node, NodeProduceFn produce, NodeReverseFn reverse);
NodeExposure* node_get_exposure(const Node* node, size_t index);

size_t node_add_forward_link(Node* node, Node* link, int relation);
size_t node_add_backward_link(Node* node, Node* link, int relation);
size_t node_add_bidirectional_link(Node* a, Node* b, int relation);
int geneology_invert_relation(int relation_type);
const NodeLink* node_get_forward_link(const Node* node, size_t index);
const NodeLink* node_get_backward_link(const Node* node, size_t index);

typedef void (*NodeVisitFn)(Node* node, int relation, void* user);
void node_for_each_forward(Node* node, NodeVisitFn visit, void* user);
void node_for_each_backward(Node* node, NodeVisitFn visit, void* user);

void node_scatter_to_siblings(Node* node, void* data);
void node_gather_from_siblings(Node* node, void* out);
void node_scatter_to_descendants(Node* node, void* data);
void node_gather_from_ancestors(Node* node, void* out);

// --- Neighbor Map API ---
// Attach a neighbor at a given pole (by index) with a label (explicit or auto-generated)
int node_attach_neighbor(Node* node, Node* neighbor, size_t pole_index, const char* label);
// Detach a neighbor by pole or label
int node_detach_neighbor(Node* node, size_t pole_index);
int node_detach_neighbor_by_label(Node* node, const char* label);
// Query a neighbor by pole or label
Node* node_get_neighbor(const Node* node, size_t pole_index);
Node* node_get_neighbor_by_label(const Node* node, const char* label);
// Ensure bidirectional link (or hold/drop if not possible)
int node_ensure_bidirectional_neighbor(Node* node, Node* neighbor, size_t pole_index, const char* label, int require_bidirectional);

// Procedural relationship naming for stencil-based relationships
// Returns a malloc'd string describing the relationship (caller must free)
char* node_generate_stencil_relation_name(const GeneralStencil* stencil, size_t pole_index, const char* base_type);

// =====================
// LOCKING POLICY AND DESIGN
// =====================
/*
Locking in this system is designed for both fine-grained (node-level) and coarse-grained (subgraph/geneology-level) concurrency control.

Node Locking:
- Each Node contains a node_mutex_t mutex for exclusive access.
- Node locking functions (lock, unlock, trylock, is_locked) provide direct, thread-safe access to individual nodes.
- Nodes should be locked before any mutation or critical read, and unlocked as soon as possible.

Geneology Lock Bank:
- The GeneologyLockBank manages locks for sets of nodes (subgraphs) and coordinates concurrent access across the entire geneology.
- The lock bank maintains a queue of lock requests (LockRequestQueue). Requests for overlapping subgraphs are queued and block until all required nodes are available.
- Non-overlapping lock requests may be granted out-of-order for maximum concurrency.
- When a lock request is granted, the lock bank issues an asynchronous notification ("your table is ready" alarm) to the requesting thread, which may be blocked waiting for the lock. This allows the thread to proceed as soon as the lock is available, without polling.
- The lock bank must prevent deadlocks, ensure fairness, and support lock escalation (from node to subgraph) as needed.
- Subgraph set operations (union, intersection, difference) are provided to efficiently manage and compare lock requests.

Design Requirements:
- All lock acquisition and release must be coordinated through the lock bank for subgraph operations.
- The lock bank must be able to confirm lock status from any thread, not just the bank's own thread.
- The system must support both blocking and non-blocking lock requests, with asynchronous notification for blocking requests.
- Locking policies must be clearly documented and enforced to avoid deadlocks and starvation.
*/
// =====================
// Node Locking API
// =====================

// Lock the node's mutex
void node_lock(Node* node);
// Unlock the node's mutex
void node_unlock(Node* node);
// Try to lock the node's mutex (non-blocking), returns 1 if successful, 0 otherwise
int node_trylock(Node* node);
// Check if the node is currently locked (non-blocking)
int node_is_locked(const Node* node);

// =====================
// Geneology Lock Bank
// =====================

// Forward declaration for lock request queue
struct LockRequestQueue;

// Structure for managing locks across a geneology (subgraph lock bank)
typedef struct GeneologyLockBank {
    // Queue of pending lock requests
    struct LockRequestQueue* request_queue;
    // Set of currently locked nodes (could be a hash set or array)
    Node** locked_nodes;
    size_t num_locked, cap_locked;
    // Mutex for synchronizing access to the lock bank
    node_mutex_t bank_mutex;
} GeneologyLockBank;

// Initialize/destroy the lock bank
GeneologyLockBank* geneology_lockbank_create(void);
void geneology_lockbank_destroy(GeneologyLockBank* bank);

// Request locks for a set of nodes (subgraph); non-blocking, will queue if not immediately available
void geneology_lockbank_request(GeneologyLockBank* bank, Node** nodes, size_t num_nodes);
// Confirm if a set of nodes is locked (can be called outside the bank thread)
int geneology_lockbank_confirm(GeneologyLockBank* bank, Node** nodes, size_t num_nodes);
// Release locks for a set of nodes
void geneology_lockbank_release(GeneologyLockBank* bank, Node** nodes, size_t num_nodes);

// =====================
// Graph Set Operations (for subgraph management)
// =====================
/*
GraphSetOps provides a general interface for set operations on node sets (subgraphs),
including union, intersection, difference, and membership checks. These are useful for
lock management, relationship queries, and subgraph analysis.
*/
typedef struct GraphSetOps {
    // Union of two sets (a ? b)
    size_t (*set_union)(Node** a, size_t a_count, Node** b, size_t b_count, Node** out_union, size_t out_cap);
    // Intersection of two sets (a ? b)
    size_t (*set_intersection)(Node** a, size_t a_count, Node** b, size_t b_count, Node** out_inter, size_t out_cap);
    // Difference of two sets (a \ b)
    size_t (*set_difference)(Node** a, size_t a_count, Node** b, size_t b_count, Node** out_diff, size_t out_cap);
    // Check if a set contains a node
    int (*set_contains)(Node** set, size_t set_count, Node* node);
    // Guidance: add more set operations as needed (e.g., symmetric difference, subset, etc.)
} GraphSetOps;

// Default set operation implementations (declarations only)
size_t graph_set_union(Node** a, size_t a_count, Node** b, size_t b_count, Node** out_union, size_t out_cap);
size_t graph_set_intersection(Node** a, size_t a_count, Node** b, size_t b_count, Node** out_inter, size_t out_cap);
size_t graph_set_difference(Node** a, size_t a_count, Node** b, size_t b_count, Node** out_diff, size_t out_cap);
int graph_set_contains(Node** set, size_t set_count, Node* node);

// --- SimpleGraph edge types ---
typedef enum {
    EDGE_PARENT_CHILD_CONTIGUOUS,
    EDGE_CHILD_PARENT_CONTIGUOUS,
    EDGE_SIBLING_LEFT_TO_RIGHT_CONTIGUOUS,
    EDGE_SIBLING_RIGHT_TO_LEFT_CONTIGUOUS,
    EDGE_SIBLING_SIBLING_NONCONTIGUOUS,
    EDGE_LINEAGE_NONCONTIGUOUS,
    EDGE_ARBITRARY
} SimpleGraphEdgeType;

typedef struct {
    Node* src;
    Node* dst;
    SimpleGraphEdgeType type;
    int relation; // index or type for the relationship
} SimpleGraphEdge;

// --- Feature hash map (simple open addressing) ---
typedef struct {
    char* key;
    void* value; // pointer to tensor (ONNX/Eigen)
} SimpleGraphFeatureEntry;

typedef struct {
    SimpleGraphFeatureEntry* entries;
    size_t num_entries, cap_entries;
} SimpleGraphFeatureMap;

// --- SimpleGraph structure ---
typedef struct {
    Geneology* geneology;
    SimpleGraphEdge* edges;
    size_t num_edges, cap_edges;

    // Contiguous feature storage (array of pointers to tensors)
    void** feature_block;
    size_t num_features, cap_features;

    // Per-node feature hash maps (indexed by node index in geneology)
    SimpleGraphFeatureMap* node_feature_maps;
    size_t num_nodes, cap_nodes;
} SimpleGraph;

SimpleGraph* simplegraph_create(Geneology* g);
void simplegraph_destroy(SimpleGraph* graph);
void simplegraph_add_edge(SimpleGraph* graph, Node* src, Node* dst, SimpleGraphEdgeType type, int relation);
void simplegraph_add_feature(SimpleGraph* graph, Node* node, const char* feature_name, void* tensor_ptr);
void* simplegraph_get_feature(SimpleGraph* graph, Node* node, const char* feature_name);

void simplegraph_forward(SimpleGraph* graph);
void simplegraph_backward(SimpleGraph* graph);

// Manifest management
// DagManifestMapping, DagManifestLevel, DagManifest, and Dag are fully
// defined in <geometry/dag.h>.  We only provide the management function
// prototypes here to prevent duplicate type definitions across headers.
Dag* dag_create(void);
void dag_destroy(Dag* dag);
void dag_add_manifest(Dag* dag, DagManifest* manifest);

#ifdef __cplusplus
}
#endif

#endif /* GEOMETRY_UTILS_H */
