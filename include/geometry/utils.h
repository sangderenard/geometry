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

#include "geometry/dag.h"
#include "geometry/graph_ops.h"
#include "geometry/stencil.h"


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

struct GuardianNode;

typedef struct GuardianToken {
    unsigned long token; // Unique token for the guardian
    size_t size;         // Size of the memory block managed by this token
} GuardianToken;;

typedef struct GuardianObjectSet {
	GuardianToken guardian_pointer_token; // Unique token for the object
	TokenGuardian* guardian; // Pointer to the guardian managing this object
	GuardianToken guardian_lock_token; // Lock token for thread safety
} GuardianObjectSet;

typedef struct TokenGuardian {
    node_mutex_t mutex;
    size_t num_threads, cap_threads;
    size_t min_allocation, max_allocation;
} TokenGuardian;

typedef struct GuardianPointerToken {
    GuardianToken token; // Unique token for the pointer
	size_t size, span; // Size of the unit / size of the memory block (in units)
} GuardianPointerToken;

typedef struct GuardianLinkedList {
    GuardianToken left;
	GuardianToken right;
	size_t max_size; // Maximum size of the linked list
	boolean is_contiguous; // Flag for contiguous allocation
	size_t list_size; // Current size of the linked list
} GuardianLinkedList;

// --- GuardianDict structure ---
typedef struct GuardianDict {
    size_t total_allocation_size; // Total allocated size
    size_t used_allocation_size;  // Number of used entries
    GuardianLinkedList keys;
    GuardianLinkedList values;                // Internal representation of the set (e.g., hash table or tree)
} GuardianDict;

// --- GuardianList structure ---
typedef struct GuardianList {
    size_t total_allocation_size; // Total allocated size
    size_t used_allocation_size;  // Number of used entries
    GuardianLinkedList entry_set;              // Dictionary for non-contiguous entries (indices to pointer tokens)
} GuardianList;

typedef struct GuardianParallelList {
    GuardianList lists;
} GuardianParallelList;

// --- GuardianSet structure ---
typedef struct GuardianSet {
    size_t total_allocation_size; // Total allocated size
    size_t used_allocation_size;  // Number of used entries
    GuardianLinkedList entry_set;  // Internal representation of the set (e.g., hash table or tree)
} GuardianSet;

typedef struct GuardianMap {
    GuardianList a_to_b_hash; //all origins and exits implied by valid paths plus directional metadata with their composed pointer sequences (the first function pointer -> any translation function pointer (optional) -> [list of next funtions receiving that input] -> valid output, etc.);
	GuardianParallelList paths_and_subweights; // List of valid paths using pole features from stencil set of the edge and the valid function flows
    GuardianSet poles; // the inputs or outputs or both that describe the stencil of the edge - it's agnostic connection points
	GuardianSet function_flows; // valid function pointer sequences by parameter types and available translations

} GuardianMap;

struct Node;
struct GuardianEdge;

typedef struct GuardianStencil {
	GeneralStencil stencil; // GeneralStencil
} GuardianStencil;

enum NodeOrientationNature {
	NODE_ORIENTATION_DOMAIN_PARALLEL = 0, // Parallel to domain
	NODE_ORIENTATION_FIXED = 1, // Fixed orientation to be set by user
	NODE_ORIENTATION_ITERATIVE = 2, // Iterative orientation solved by physics engine
	NODE_ORIENTATION_DOMAIN_TRANSFORM = 3, // Domain transformation
	NODE_ORIENTATION_SPACE_FILLING_PATTERN = 4, // Space filling orientation (ie tetrahedral, hexahedral, etc. patterns)
    
};

typedef struct GuardianStencilSet {
    GuardianParallelList stencils_orthagonalities_orientations; // what are the stencils, their relationships, and orientation modes
} GuardianStencilSet;

typedef struct Node {
    char* id;
	GuardianStencilSet stencil_set; // Stencil set for the node
	GuardianObjectSet object_set; // Object set for the node listing the thread lock token, guardian pointer,  and the object pointer token

	GuardianObjectSet feature_set; // Set of features associated with the node as an arbitrary guardian object
	GuardianSet compatible_edges; // Set of compatible edge_type identities for at least one point on one stencil this node
	GuardianList internal_edges; // List of internal edges (edges that are not exposed to the outside world, but are used internally for node processing)

    struct Emergence* emergence = NULL;
} Node;
// Define EdgeType structs using tokens
typedef struct EdgeAttribute {
    char* name; // Name of the attribute
    double weight; // Weight of the attribute
	GuardianSet groups; // Dictionary of groups
    GuardianDict function_graphs;
} EdgeAttribute;

typedef struct Subedge {
    EdgeAttribute attribute;
	GuardianMap poles_functions_map; // Map of poles to function flows to poles, bidirectional reference
	GuardianList effective_subedges; // List of effective subedges
};

typedef struct EdgeType {
    char* identifier; // Unique identifier for the edge type
	GuardianList subedges; // List of subedges associated with this edge type
	GuardianStencil edge_stencil; // Stencil for the edge type
} EdgeType;

typedef struct Edge {
    GuardianParallelList subedjes_and_attachments_list;
} Edge;

typedef struct GuardianEdge {
    Edge edge;
} GuardianEdge;

typedef struct GuardianGeneology {
    GuardianObjectSet self; // Self-reference for the geneology object
	int kernel_radius; // Radius of the kernel for cross-stencil node kernels in terms of recursive stencil points
    GuardianParallelList cross_stencil_node_kernels_correlated_with_edges; // Cross-stencil node kernels correlated with edges
} GuardianGeneology;

// --- SimpleGraph structure ---
typedef struct GuardianSimpleGraph {
	GuardianObjectSet self; // Self-reference for the graph object
    GuardianGeneology geneology; // a guardian object containing tiling relationships for node fields
	GuardianSet nodes; // Set of nodes in the graph
	GuardianSet edges; // Set of edges in the graph

} GuardianSimpleGraph;

// --- Dag structure ---
typedef struct Dag {
    DagManifest* manifests;
    size_t num_manifests, cap_manifests;
} Dag;

typedef struct DagManifest {
    DagManifestLevel* levels;
    size_t num_levels;
} DagManifest;

typedef struct DagManifestLevel {
    DagManifestMapping* mappings;
    size_t num_mappings;
    int level_index;
} DagManifestLevel;

typedef struct DagManifestMapping {
    Node** inputs;
    size_t num_inputs;
    Node** outputs;
    size_t num_outputs;
} DagManifestMapping;

// --- NeuralNetwork structure ---
typedef struct NeuralNetwork {
    Dag* dags[NN_MAX_DAGS];
    size_t num_dags;
    NeuralNetworkStep* steps[NN_MAX_DAGS][NN_MAX_STEPS];
    size_t num_steps[NN_MAX_DAGS];
    NeuralNetworkFunctionRepo function_repo;
} NeuralNetwork;

typedef struct NeuralNetworkStep {
    NNForwardFn forward;
    NNBackwardFn backward;
    void* user_data;
} NeuralNetworkStep;

typedef struct NeuralNetworkFunctionRepo {
    NeuralNetworkFunctionEntry entries[NN_MAX_FUNCTIONS];
    size_t num_entries;
} NeuralNetworkFunctionRepo;

typedef struct NeuralNetworkFunctionEntry {
    const char* name;
    NNForwardFn forward;
    NNBackwardFn backward;
} NeuralNetworkFunctionEntry;

// === Object Types Enum ===
typedef enum {
    OBJECT_TYPE_NODE = 1,
    OBJECT_TYPE_EDGE,
    OBJECT_TYPE_POINTER_TOKEN,
    OBJECT_TYPE_STENCIL,
    OBJECT_TYPE_PARAMETRIC_DOMAIN
} ObjectType;

#ifdef __cplusplus
}
#endif

#endif // GEOMETRY_UTILS_H
