#ifndef GEOMETRY_UTILS_H
#define GEOMETRY_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __BOOLEAN_DEFINED
#define __BOOLEAN_DEFINED

typedef unsigned char boolean;
#define true 1
#define false 0

#endif

#include <limits.h>
#define MAX_U8  ((uint8_t)0xFF)
#define MAX_U16 ((uint16_t)0xFFFF)
#define MAX_U32 ((uint32_t)0xFFFFFFFF)
#define MAX_U64 ((uint64_t)0xFFFFFFFFFFFFFFFFULL)
#define MAX_I8  ((int8_t)0x7F)
#define MAX_I16 ((int16_t)0x7FFF)
#define MAX_I32 ((int32_t)0x7FFFFFFF)
#define MAX_I64 ((int64_t)0x7FFFFFFFFFFFFFFFULL)
#define MIN_U8  ((uint8_t)0x00)
#define MIN_U16 ((uint16_t)0x0000)
#define MIN_U32 ((uint32_t)0x00000000)
#define MIN_U64 ((uint64_t)0x0000000000000000ULL)
#define MIN_I8  ((int8_t)0x80)
#define MIN_I16 ((int16_t)0x8000)
#define MIN_I32 ((int32_t)0x80000000)
#define MIN_I64 ((int64_t)0x8000000000000000ULL)

#include <stddef.h>
#include <stdint.h>

#include "geometry/guardian_platform.h"

#include "geometry/dag.h"
#include "geometry/graph_ops.h"
#include "geometry/stencil.h"
#include "geometry/parametric_domain.h"

/* Forward declarations for types used before definition */
typedef struct GuardianLinkedList GuardianLinkedList;
typedef struct GuardianList GuardianList;
typedef struct GuardianParallelList GuardianParallelList;
typedef struct GuardianDict GuardianDict;
typedef struct GuardianSet GuardianSet;
typedef struct GuardianMap GuardianMap;
typedef struct GuardianStencilSet GuardianStencilSet;
typedef struct GuardianGeneology GuardianGeneology;
typedef struct GuardianHeap GuardianHeap;
typedef struct GuardianThread GuardianThread;
typedef struct TokenGuardian TokenGuardian;

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

typedef enum {
    GUARDIAN_TOKEN_NONE          = 0,
    GUARDIAN_TOKEN_LOCK          = 1 << 0,
    GUARDIAN_TOKEN_MEMORY        = 1 << 1,
    GUARDIAN_TOKEN_OBJECT        = 1 << 2,
    GUARDIAN_TOKEN_OBJECT_FREE   = 1 << 3,
    GUARDIAN_TOKEN_LOCK_FREE     = 1 << 4,
    GUARDIAN_TOKEN_MEMORY_FREE   = 1 << 5
} GuardianTokenFlags;

typedef struct GuardianToken {
    boolean is_locked;
    boolean is_initialized;

    TokenGuardian* guardian;

    unsigned long long token;
    size_t size;

    mutex_t* ___lock;
    void* ___memory;
    void* ___object;
} GuardianToken;


typedef struct GuardianObjectSet {
	GuardianToken guardian_pointer_token; // Unique token for the object
	TokenGuardian* guardian; // Pointer to the guardian managing this object
	GuardianToken guardian_lock_token; // Lock token for thread safety
} GuardianObjectSet;

typedef struct GuardianStack {
	GuardianToken token; // Unique token for the stack
	size_t size; // Size of the stack
	size_t max_size; // Maximum size of the stack
	NodeFeatureType type; // Type of the stack (e.g., NODE_FEATURE_TYPE_FLOAT)
	GuardianPointerToken data_pointer_token; // Pointer to the data array
	void* data; // Pointer to the actual data array
	TokenGuardian* guardian; // Pointer to the guardian managing this stack
	double priority; // Priority for scheduling or processing
} GuardianStack;

typedef enum {
	NODE_FEATURE_TYPE_INT = 0, // Integer feature
	NODE_FEATURE_TYPE_FLOAT = 1, // Float feature
	NODE_FEATURE_TYPE_DOUBLE = 2, // Double feature
	NODE_FEATURE_TYPE_STRING = 3, // String feature
	NODE_FEATURE_TYPE_BOOLEAN = 4, // Boolean feature
	NODE_FEATURE_TYPE_POINTER = 5, // Pointer feature
	NODE_FEATURE_TYPE_VECTOR_INT = 6, // Vector feature
	NODE_FEATURE_TYPE_VECTOR_FLOAT = 7, // Vector feature
	NODE_FEATURE_TYPE_VECTOR_DOUBLE = 8, // Vector feature
	NODE_FEATURE_TYPE_VECTOR_STRING = 9, // Vector feature
	NODE_FEATURE_TYPE_VECTOR_BOOLEAN = 10, // Vector feature
	NODE_FEATURE_TYPE_VECTOR_POINTER = 11, // Vector feature
	NODE_FEATURE_TYPE_TENSOR_INT = 12, // Tensor feature
	NODE_FEATURE_TYPE_TENSOR_FLOAT = 13, // Tensor feature
	NODE_FEATURE_TYPE_TENSOR_DOUBLE = 14, // Tensor feature
	NODE_FEATURE_TYPE_TENSOR_STRING = 15, // Tensor feature
	NODE_FEATURE_TYPE_TENSOR_BOOLEAN = 16, // Tensor feature
	NODE_FEATURE_TYPE_TENSOR_POINTER = 17, // Tensor feature
	NODE_FEATURE_TYPE_TENSOR_VECTOR_INT = 18, // Tensor of vectors of integers
	NODE_FEATURE_TYPE_TENSOR_VECTOR_FLOAT = 19, // Tensor of vectors of floats
	NODE_FEATURE_TYPE_TENSOR_VECTOR_DOUBLE = 20, // Tensor of vectors of doubles
	NODE_FEATURE_TYPE_TENSOR_VECTOR_STRING = 21, // Tensor of vectors of strings
	NODE_FEATURE_TYPE_TENSOR_VECTOR_BOOLEAN = 22, // Tensor of vectors of booleans
	NODE_FEATURE_TYPE_TENSOR_VECTOR_POINTER = 23, // Tensor of vectors of pointers
	NODE_FEATURE_TYPE_NODE = 24, // Node feature type
	NODE_FEATURE_TYPE_EDGE = 25, // Edge feature type
	NODE_FEATURE_TYPE_STENCIL = 26, // Stencil feature type
	NODE_FEATURE_TYPE_GENEALOGY = 27, // Genealogy feature type
	NODE_FEATURE_TYPE_EMERGENCE = 28, // Emergence feature type
	NODE_FEATURE_TYPE_LINKED_LIST = 29, // Linked list feature type
	NODE_FEATURE_TYPE_DICTIONARY = 30, // Dictionary feature type
	NODE_FEATURE_TYPE_SET = 31, // Set feature type
	NODE_FEATURE_TYPE_MAP = 32, // Map feature type
	NODE_FEATURE_TYPE_PARALLEL_LIST = 33, // Parallel list feature type
	NODE_FEATURE_TYPE_LIST = 34, // List feature type
	NODE_FEATURE_TYPE_POINTER_TOKEN = 35, // Pointer token feature type
	NODE_FEATURE_TYPE_GUARDIAN = 36, // Guardian feature type
	NODE_FEATURE_TYPE_TOKEN = 37, // Guardian token feature type
	NODE_FEATURE_TYPE_OBJECT_SET = 38, // Guardian object set feature type
	NODE_FEATURE_TYPE_MEMORY_TOKEN = 39, // Memory token feature type
	NODE_FEATURE_TYPE_TOKEN_LOCK = 40, // Token lock feature type
	NODE_FEATURE_TYPE_MEMORY_MAP = 41, // Memory map feature type
	NODE_FEATURE_TYPE_MESSAGE = 42, // Guardian message feature type
	NODE_FEATURE_TYPE_CUSTOM = 43, // Custom feature type for user-defined features

} NodeFeatureType;

typedef struct GuardianMessage {
	GuardianToken self; // Self-reference for the message object
	size_t to_id; // ID of the recipient thread or object
	size_t from_id; // ID of the sender thread or object
	size_t size; // Size of the message data
	void * data; // Pointer to the message data
	double priority; // Priority of the message for processing
	double timestamp; // Timestamp of when the message was created
	double priority; // Priority for processing the message
	double patience; // Patience for waiting on resources
	boolean is_locked; // Flag for whether the message is locked
} GuardianMessage;

typedef struct GuardianMailbox {
	GuardianObjectSet self; // Self-reference for the mailbox object
	size_t thread_id; // ID of the thread owning this mailbox
	GuardianMessage * message_head; // Pointer to the head of the message queue
	GuardianMessage * message_tail; // Pointer to the tail of the message queue
	GuardianLinkedList message_list; // Linked list of messages in the mailbox
	size_t max_size; // Maximum size of the mailbox
	size_t message_count; // Count of messages in the mailbox
	enum MailboxDisposalPolicy {
		MAILBOX_DISPOSAL_POLICY_NONE = 0, // No disposal policy
		MAILBOX_DISPOSAL_POLICY_FIFO = 1, // First In First Out
		MAILBOX_DISPOSAL_POLICY_LIFO = 2, // Last In First Out
		MAILBOX_DISPOSAL_POLICY_PRIORITY = 3 // Priority-based disposal
	} disposal_policy; // Disposal policy for the mailbox
	boolean is_locked; // Flag for whether the mailbox is locked
	boolean is_initialized; // Flag for whether the mailbox is initialized
	boolean is_contiguous; // Flag for whether the mailbox is contiguous
	boolean is_synchronized; // Flag for whether the mailbox is synchronized
} GuardianMailbox;
const long long GUARDIAN_NOT_USED = LLONG_MAX; // Constant for unused guardian
typedef struct GuardianThread {
	GuardianObjectSet self; // Self-reference for the thread object
	size_t thread_id; // Unique identifier for the thread
	size_t stack_size; // Size of the stack allocated for this thread
	size_t max_stack_size; // Maximum size of the stack for this thread
	size_t min_allocation; // Minimum allocation size for the thread
	size_t max_allocation; // Maximum allocation size for the thread
	double allocation_growth_factor; // Growth factor for memory allocation
	double priority; // Priority for scheduling or processing
	double patience; // Patience for waiting on resources
	GuardianSet mailbox; // Set of messages in the thread's mailbox
	GuardianMessage * message_head; // Pointer to the head of the message queue
	GuardianMessage * message_tail; // Pointer to the tail of the message queue
	boolean auto_contiguous_allocation; // Flag for automatic contiguous allocation	
	boolean auto_synchronization; // Flag for automatic synchronization
	boolean using_buffered_io; // Flag for buffered I/O operations
	boolean is_locked; // Flag for whether the thread is locked
	boolean is_initialized; // Flag for whether the thread is initialized
	boolean is_contiguous; // Flag for whether the thread is contiguous
	boolean is_synchronized; // Flag for whether the thread is synchronized
	GuardianStack * stack; // Pointer to the stack allocated for this thread
} GuardianThread;

typedef struct GuardianHeap {
	GuardianToken self; // Self-reference for the heap object
	size_t size; // Size of the heap
	size_t min_allocation; // Minimum allocation size for the heap
	size_t max_allocation; // Maximum allocation size for the heap
	double allocation_growth_factor; // Growth factor for memory allocation
	boolean auto_contiguous_allocation; // Flag for automatic contiguous allocation
	boolean auto_synchronization; // Flag for automatic synchronization
	boolean using_buffered_io; // Flag for buffered I/O operations
	boolean is_locked; // Flag for whether the heap is locked
	boolean is_initialized; // Flag for whether the heap is initialized
	boolean is_contiguous; // Flag for whether the heap is contiguous
	boolean is_synchronized; // Flag for whether the heap is synchronized
	GuardianStack ** stray_data; // Pointer to stray data in the heap
	size_t stray_data_count; // Count of stray data in the heap
	size_t max_stray_data_count; // Maximum count of stray data in the heap
	GuardianObjectSet object_set; // Object set for the heap listing the thread lock token, guardian pointer, and the object pointer token
	GuardianToken main_lock; // Main lock token for the heap
	GuardianThread my_thread; // Pointer to the thread managing this heap
	GuardianToken my_thread_lock; // Lock token for the thread managing this heap
	GuardianToken heap_lock; // Lock token for thread safety
	GuardianToken heap_buffer_lock; // Lock token for buffered I/O operations
	GuardianParallelList thread_tokens_locks_buffers_threads; // the manager database for threads
	GuardianList stalled_threads; // List of stalled threads waiting for resources
	GuardianList active_threads; // List of active threads using this heap
	GuardianList free_threads; // List of free threads available for use
	GuardianList in_heap_stacks; // List of stacks allocated in this heap
	GuardianList out_heap_stacks; // List of stacks allocated outside this heap
	GuardianParallelList threads; // List of threads associated with this heap
	// Maps are enriched with bidirectional references to allow for efficient lookups and updates
	GuardianMap in_heap_memory_map; // Map of memory blocks managed by this heap
	GuardianMap out_heap_memory_map; // Map of memory blocks allocated outside this heap
	GuardianMap in_heap_stacks_map; // Map of stacks allocated in this heap
	GuardianMap out_heap_stacks_map; // Map of stacks allocated outside this heap
	GuardianMap in_heap_objects_map; // Map of objects allocated in this heap
	GuardianMap out_heap_objects_map; // Map of objects allocated outside this heap
	GuardianMap in_heap_tokens_map; // Map of tokens allocated in this heap
	GuardianMap out_heap_tokens_map; // Map of tokens allocated outside this heap
	GuardianMap in_heap_locks_map; // Map of locks allocated in this heap
	GuardianMap out_heap_locks_map; // Map of locks allocated outside this heap
	GuardianMap in_heap_messages_map; // Map of messages allocated in this heap
	GuardianMap out_heap_messages_map; // Map of messages allocated outside this heap
	GuardianMap in_heap_custom_features_map; // Map of custom features allocated in this heap
	GuardianMap out_heap_custom_features_map; // Map of custom features allocated outside this heap
	GuardianMap in_heap_guardians_map; // Map of guardians allocated in this heap
	GuardianMap out_heap_guardians_map; // Map of guardians allocated outside this heap
	GuardianList maps; // List of maps associated with this heap
	GuardianDict custom_features; // List of custom features associated with this heap
} GuardianHeap;
struct TokenGuardian;

typedef struct TokenGuardian {
	boolean is_locked; // Flag for whether the guardian is locked
	boolean is_initialized; // Flag for whether the guardian is initialized
	boolean is_contiguous; // Flag for whether the guardian is contiguous
	boolean is_synchronized; // Flag for whether the guardian is synchronized
	GuardianObjectSet self; // Self-reference for the guardian object
	GuardianToken main_lock; // Main lock token for the guardian
    size_t max_threads;
    size_t min_allocation, max_allocation;
	byte* data; // Pointer to the data managed by this guardian
	GuardianHeap* heap; // Pointer to the heap managed by this guardian
	double allocation_growth_factor; // Growth factor for memory allocation 
	int concurrent_threads; // Number of concurrent threads using this guardian
	boolean using_buffered_io; // Flag for buffered I/O operations
	boolean auto_contiguous_allocation; // Flag for automatic contiguous allocation
	boolean auto_synchronization; // Flag for automatic synchronization
	GuardianParallelList thread_tokens_locks_buffers_threads; // the manager database for threads
	GuardianMap stack_memory_map; // Map of memory blocks managed by this guardian
	size_t heap_size; // Size of the heap managed by this guardian
	int max_stack_count; // Number of stacks managed by this guardian
	size_t max_stack_size; // Size of each stack managed by this guardian
	NodeFeatureType default_feature_type; // Default feature type for nodes
	GuardianStencilSet default_stencil_set; // Default stencil set for nodes
	int default_node_orientation; // Default node orientation nature
	ParametricDomain default_parametric_domain; // Default parametric domain for nodes
	GuardianGeneology default_geneology; // Default geneology for nodes
	GuardianToken next_token; // Next available token ID for this guardian
	TokenGuardian* token_host; // Pointer to the host guardian managing this token
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
    GuardianPointerToken* data; // Pointer to the data array
	TokenGuardian* guardian; // Pointer to the guardian managing this linked list
	NodeFeatureType feature_type; // Feature type for the linked list
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
