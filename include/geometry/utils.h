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
#include "geometry/stencil.h"
#include "geometry/parametric_domain.h"
#include "geometry/graph_ops.h"
#include "geometry/graph_ops_handler.h"

#include "geometry/guardian_link_cache.h" // Include the new primitive layer

/* Forward declarations for types used before definition */
typedef struct GuardianList GuardianList;
typedef struct GuardianDict GuardianDict;
typedef struct GuardianSet GuardianSet;
typedef struct GuardianMap GuardianMap;
typedef struct GuardianStencilSet GuardianStencilSet;
typedef struct GuardianGeneology GuardianGeneology;
typedef struct GuardianHeap GuardianHeap;
typedef struct GuardianThread GuardianThread;
typedef struct TokenGuardian TokenGuardian;
typedef struct GuardianParallelList GuardianParallelList;
typedef struct GuardianObjectSet GuardianObjectSet;	
typedef struct GuardianLinkedList GuardianLinkedList;
typedef struct GuardianLinkNode GuardianLinkNode;

// --- Emergence structure for node-level adaptation ---
// Forward declarations
typedef struct Node Node;
typedef struct Emergence Emergence;


typedef struct Emergence {  
    // Statistics  
    double activation_sum;  
    double activation_sq_sum;  
    size_t activation_count;  
    uint64_t last_global_step;  
    uint64_t last_timestamp;  

    // Thread lock for parallel split  
    mutex_t thread_lock;  
    int is_locked;  

    // Decision hooks  
    int (*should_split)(struct Emergence* self, struct Node* node);  
    int (*should_apoptose)(struct Emergence* self, struct Node* node);  
    int (*should_metastasize)(struct Emergence* self, struct Node* node);  

    // Action hooks  
    void (*split)(struct Emergence* self, struct Node* node);  
    void (*apoptose)(struct Emergence* self, struct Node* node);  
    void (*metastasize)(struct Emergence* self, struct Node* node);  

    // User data for custom policies  
    void* user_data;  
} Emergence;

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
	GuardianToken* guardian_pointer_token; // Unique token for the object
	TokenGuardian* guardian; // Pointer to the guardian managing this object
	GuardianToken* guardian_lock_token; // Lock token for thread safety
} GuardianObjectSet;

typedef struct GuardianStack {
	GuardianToken token; // Unique token for the stack
	size_t size; // Size of the stack
	size_t max_size; // Maximum size of the stack
	NodeFeatureType type; // Type of the stack (e.g., NODE_FEATURE_TYPE_FLOAT)
	GuardianToken data_pointer_token; // Pointer to the data array
	void* data; // Pointer to the actual data array
	TokenGuardian* guardian; // Pointer to the guardian managing this stack
	double priority; // Priority for scheduling or processing
} GuardianStack;



typedef struct GuardianMessage {
	GuardianToken self; // Self-reference for the message object
	size_t to_id; // ID of the recipient thread or object
	size_t from_id; // ID of the sender thread or object
	size_t size; // Size of the message data
	void * data; // Pointer to the message data
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
	GuardianList * message_list; // List of messages in the mailbox
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
	GuardianSet * mailbox; // Set of messages in the thread's mailbox
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
	GuardianToken * self; // Self-reference for the heap object
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
	GuardianObjectSet * object_set; // Object set for the heap listing the thread lock token, guardian pointer, and the object pointer token
	GuardianToken * main_lock; // Main lock token for the heap
	GuardianThread * my_thread; // Pointer to the thread managing this heap
	GuardianToken * my_thread_lock; // Lock token for the thread managing this heap
	GuardianToken * heap_lock; // Lock token for thread safety
	GuardianToken * heap_buffer_lock; // Lock token for buffered I/O operations
	GuardianParallelList * thread_tokens_locks_buffers_threads; // the manager database for threads
	GuardianList * stalled_threads; // List of stalled threads waiting for resources
	GuardianList * active_threads; // List of active threads using this heap
	GuardianList * free_threads; // List of free threads available for use
	GuardianList * in_heap_stacks; // List of stacks allocated in this heap
	GuardianList * out_heap_stacks; // List of stacks allocated outside this heap
	GuardianParallelList * threads; // List of threads associated with this heap
	// Maps are enriched with bidirectional references to allow for efficient lookups and updates
	GuardianMap * in_heap_memory_map; // Map of memory blocks managed by this heap
	GuardianMap * out_heap_memory_map; // Map of memory blocks allocated outside this heap
	GuardianMap * in_heap_stacks_map; // Map of stacks allocated in this heap
	GuardianMap * out_heap_stacks_map; // Map of stacks allocated outside this heap
	GuardianMap * in_heap_objects_map; // Map of objects allocated in this heap
	GuardianMap * out_heap_objects_map; // Map of objects allocated outside this heap
	GuardianMap * in_heap_tokens_map; // Map of tokens allocated in this heap
	GuardianMap * out_heap_tokens_map; // Map of tokens allocated outside this heap
	GuardianMap * in_heap_locks_map; // Map of locks allocated in this heap
	GuardianMap * out_heap_locks_map; // Map of locks allocated outside this heap
	GuardianMap * in_heap_messages_map; // Map of messages allocated in this heap
	GuardianMap * out_heap_messages_map; // Map of messages allocated outside this heap
	GuardianMap * in_heap_custom_features_map; // Map of custom features allocated in this heap
	GuardianMap * out_heap_custom_features_map; // Map of custom features allocated outside this heap
	GuardianMap * in_heap_guardians_map; // Map of guardians allocated in this heap
	GuardianMap * out_heap_guardians_map; // Map of guardians allocated outside this heap
	GuardianList * maps; // List of maps associated with this heap
	GuardianDict * custom_features; // List of custom features associated with this heap
} GuardianHeap;
struct TokenGuardian;

typedef struct TokenGuardian {
	boolean is_locked; // Flag for whether the guardian is locked
	boolean is_initialized; // Flag for whether the guardian is initialized
	boolean is_contiguous; // Flag for whether the guardian is contiguous
	boolean is_synchronized; // Flag for whether the guardian is synchronized
	GuardianObjectSet * self; // Self-reference for the guardian object
	GuardianToken * main_lock; // Main lock token for the guardian
    size_t max_threads;
    size_t min_allocation, max_allocation;
	byte* data; // Pointer to the data managed by this guardian
	GuardianHeap* heap; // Pointer to the heap managed by this guardian
	double allocation_growth_factor; // Growth factor for memory allocation 
	int concurrent_threads; // Number of concurrent threads using this guardian
	boolean using_buffered_io; // Flag for buffered I/O operations
	boolean auto_contiguous_allocation; // Flag for automatic contiguous allocation
	boolean auto_synchronization; // Flag for automatic synchronization
	GuardianParallelList * thread_tokens_locks_buffers_threads; // the manager database for threads
	GuardianMap * stack_memory_map; // Map of memory blocks managed by this guardian
	size_t heap_size; // Size of the heap managed by this guardian
	int max_stack_count; // Number of stacks managed by this guardian
	size_t max_stack_size; // Size of each stack managed by this guardian
	NodeFeatureType default_feature_type; // Default feature type for nodes
	GuardianStencilSet * default_stencil_set; // Default stencil set for nodes
	int default_node_orientation; // Default node orientation nature
	ParametricDomain * default_parametric_domain; // Default parametric domain for nodes
	GuardianGeneology * default_geneology; // Default geneology for nodes
	GuardianToken * next_token; // Next available token ID for this guardian
	TokenGuardian* token_host; // Pointer to the host guardian managing this token
} TokenGuardian;

typedef struct GuardianPointerToken {
    GuardianToken * token; // Unique token for the pointer
	size_t size, span; // Size of the unit / size of the memory block (in units)
} GuardianPointerToken;
typedef struct GuardianLinkedList {
	GuardianPointerToken * left; // Pointer to the left end of the linked list
	GuardianPointerToken * right; // Pointer to the right end of the linked list
	size_t size; // Size of the linked list
	size_t max_size; // Maximum size of the linked list
	NodeFeatureType feature_type; // Type of payload in this linked list
	TokenGuardian* guardian; // The guardian that owns this linked list
} GuardianLinkedList;	
typedef struct GuardianLinkNode {
	GuardianPointerToken* pointer_token; // Pointer to the payload
	GuardianLinkNode* next; // Pointer to the next node in the list
	GuardianLinkNode* prev; // Pointer to the previous node in the list
	NodeFeatureType feature_type; // Type of payload in this node
} GuardianLinkNode;
GuardianToken * guardian_create_pointer_token(TokenGuardian* g, void* ptr, NodeFeatureType type);
// A GuardianList is a container for a doubly-linked list of payloads.
// It is built upon the primitive GuardianLinkNode from the global cache.
typedef struct GuardianList {
    size_t count;
	GuardianLinkedList* index_to_pointer; // Index to pointer mapping for fast access
	GuardianLinkedList* pointer_to_index; // Pointer to index mapping for fast access
	GuardianMap* index_to_pointer_map; // Map of indices to pointers for fast access
    TokenGuardian* guardian; // The guardian that owns this list
    NodeFeatureType feature_type; // The type of payload in the list
} GuardianList;

typedef struct GuardianParallelList {
    GuardianList * lists;
} GuardianParallelList;

// --- GuardianDict structure ---
// A dictionary mapping keys to values. Implemented with two parallel lists.
typedef struct GuardianDict {
    GuardianList * keys;
    GuardianList * values;
} GuardianDict;

// --- GuardianSet structure ---
// An unordered collection of unique items. Implemented as a single list.
// Hashing/uniqueness must be handled by the functions operating on it.
typedef struct GuardianSet {
    GuardianList * entries;
} GuardianSet;

typedef struct GuardianMap {
    GuardianList * a_to_b_hash; //all origins and exits implied by valid paths plus directional metadata with their composed pointer sequences (the first function pointer -> any translation function pointer (optional) -> [list of next funtions receiving that input] -> valid output, etc.);
	GuardianParallelList * paths_and_subweights; // List of valid paths using pole features from stencil set of the edge and the valid function flows
    GuardianSet * poles; // the inputs or outputs or both that describe the stencil of the edge - it's agnostic connection points
	GuardianSet * function_flows; // valid function pointer sequences by parameter types and available translations

} GuardianMap;

struct Node;
struct GuardianEdge;

typedef struct GuardianStencil {
	GuardianParallelList point_flags; // Set of points in the stencil
	GuardianSet stencil_flags; // Set of flags for the stencil
} GuardianStencil;

typedef enum NodeOrientationNature {
	NODE_ORIENTATION_DOMAIN_PARALLEL = 0, // Parallel to domain
	NODE_ORIENTATION_FIXED = 1, // Fixed orientation to be set by user
	NODE_ORIENTATION_ITERATIVE = 2, // Iterative orientation solved by physics engine
	NODE_ORIENTATION_DOMAIN_TRANSFORM = 3, // Domain transformation
	NODE_ORIENTATION_SPACE_FILLING_PATTERN = 4, // Space filling orientation (ie tetrahedral, hexahedral, etc. patterns)
    
} NodeOrientationNature;

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

	struct Emergence* emergence;
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
    GuardianObjectSet* self; // Self-reference for the geneology object
	int kernel_radius; // Radius of the kernel for cross-stencil node kernels in terms of recursive stencil points
    GuardianParallelList* cross_stencil_node_kernels_correlated_with_edges; // Cross-stencil node kernels correlated with edges
	GuardianStencilSet* stencil_set; // Stencil set for the geneology
	ParametricDomain* domain; // Parametric domain for the geneology
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

// Add missing function prototypes
TokenGuardian* find_token_authority(TokenGuardian* g);
TokenGuardian* guardian_initialize(TokenGuardian* parent, size_t num_threads);
GuardianToken* guardian_create_pointer_token(TokenGuardian* g, void* ptr, NodeFeatureType type);
GuardianToken* guardian_create_lock_token(TokenGuardian* g);
boolean guardian_lock_with_timeout(TokenGuardian* g, GuardianToken guardian_lock_token, int duration, boolean reentrant);
int guardian_try_lock(TokenGuardian* g, unsigned long lock_token);
void guardian_lock(TokenGuardian* g, unsigned long lock_token);
void guardian_unlock(TokenGuardian* g, unsigned long lock_token);
int guardian_is_locked(TokenGuardian* g, unsigned long lock_token);
void* guardian_dereference_object(TokenGuardian* g, unsigned long pointer_token);

TokenGuardian* ___guardian_create_internal_(void);
void ___guardian_destroy_internal_(TokenGuardian* g);
unsigned long ___guardian_register_thread_internal_(TokenGuardian* g);
void ___guardian_unregister_thread_internal_(TokenGuardian* g, unsigned long token);
void* ___guardian_alloc_internal_(TokenGuardian* g, size_t size, unsigned long* token_out);
void ___guardian_free_internal_(TokenGuardian* g, unsigned long token);
void ___guardian_send_internal_(TokenGuardian* g, unsigned long from, unsigned long to, const void* data, size_t size);
size_t ___guardian_receive_internal_(TokenGuardian* g, unsigned long to, void* buffer, size_t max_size);
unsigned long ___guardian_create_object_internal_(TokenGuardian* g, int type, int count, GuardianPointerToken referrent, GuardianList params);
void ___guardian_destroy_object_internal_(TokenGuardian* g, unsigned long token);
unsigned long ___guardian_parse_nested_object_internal_(TokenGuardian* g, const char* request);

#ifdef __cplusplus
}
#endif

#endif // GEOMETRY_UTILS_H
