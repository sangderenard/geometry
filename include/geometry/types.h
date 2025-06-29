#ifndef GEOMETRY_TYPES_H
#define GEOMETRY_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/* Basic boolean and byte types */
typedef unsigned char boolean;
#define true 1
#define false 0

typedef unsigned char byte;

/* Platform neutral time structure */
typedef struct {
    int64_t milliseconds;
    int64_t microseconds;
    int64_t nanoseconds;
} guardian_time_t;



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
// duplicate includes of <stddef.h> and <stdint.h> removed

#include "geometry/guardian_platform_types.h"
#include "assembly_backend/thread_ops.h"

#include "geometry/dag.h"
#include "geometry/stencil.h"
#include "geometry/parametric_domain.h"
#include "geometry/graph_ops.h"
#include "geometry/graph_ops_handler.h"

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
typedef struct GuardianPointerToken GuardianPointerToken;
typedef struct GuardianLinkNode GuardianLinkNode;

// --- Emergence structure for node-level adaptation ---
// Forward declarations
typedef struct Node Node;
typedef struct Emergence Emergence;


typedef enum {
    NODE_FEATURE_IDX_INT = 0,
    NODE_FEATURE_IDX_FLOAT,
    NODE_FEATURE_IDX_DOUBLE,
    NODE_FEATURE_IDX_STRING,
    NODE_FEATURE_IDX_BOOLEAN,
    NODE_FEATURE_IDX_POINTER,
    NODE_FEATURE_IDX_COMPLEX,
    NODE_FEATURE_IDX_COMPLEX_DOUBLE,
    NODE_FEATURE_IDX_VECTOR,
    NODE_FEATURE_IDX_TENSOR,
    NODE_FEATURE_IDX_NODE,
    NODE_FEATURE_IDX_EDGE,
    NODE_FEATURE_IDX_STENCIL,
    NODE_FEATURE_IDX_GENEALOGY,
    NODE_FEATURE_IDX_EMERGENCE,
    NODE_FEATURE_IDX_LINKED_LIST,
    NODE_FEATURE_IDX_DICTIONARY,
    NODE_FEATURE_IDX_SET,
    NODE_FEATURE_IDX_MAP,
    NODE_FEATURE_IDX_PARALLEL_LIST,
    NODE_FEATURE_IDX_LIST,
    NODE_FEATURE_IDX_POINTER_TOKEN,
    NODE_FEATURE_IDX_GUARDIAN,
    NODE_FEATURE_IDX_STACK,
    NODE_FEATURE_IDX_HEAP,
    NODE_FEATURE_IDX_TOKEN,
    NODE_FEATURE_IDX_OBJECT_SET,
    NODE_FEATURE_IDX_MEMORY_TOKEN,
    NODE_FEATURE_IDX_TOKEN_LOCK,
    NODE_FEATURE_IDX_MEMORY_MAP,
    NODE_FEATURE_IDX_MESSAGE,
    NODE_FEATURE_IDX_CUSTOM,
    NODE_FEATURE_IDX_BITFIELD,
    NODE_FEATURE_IDX_BYTEFIELD,
    NODE_FEATURE_IDX_RADIX_ENCODING,
    NODE_FEATURE_IDX_PATTERN_PALETTE_STREAM,
    NODE_FEATURE_IDX_ENCODING_ENGINE,
    NODE_FEATURE_IDX_COUNT,
    NODE_FEATURE_IDX_MUTEX_T,
    NODE_FEATURE_IDX_STENCIL_SET,
} NodeFeatureIndex;

typedef enum {
    NODE_CATEGORY_PRIMITIVE  = 1ULL << 16,
    NODE_CATEGORY_VECTOR     = 1ULL << 17,
    NODE_CATEGORY_TENSOR     = 1ULL << 18,
    NODE_CATEGORY_GRAPH      = 1ULL << 19,
    NODE_CATEGORY_COLLECTION = 1ULL << 20,
    NODE_CATEGORY_GUARDIAN   = 1ULL << 21,
    NODE_CATEGORY_ENCODING   = 1ULL << 22,
    NODE_CATEGORY_CUSTOM     = 1ULL << 23
} NodeFeatureCategory;

#define NODE_FEATURE_INDEX_MASK 0xFFFFULL

typedef unsigned long long NodeFeatureType;

#define NODE_FEATURE_TYPE_INT (NODE_CATEGORY_PRIMITIVE | NODE_FEATURE_IDX_INT)
#define NODE_FEATURE_TYPE_FLOAT (NODE_CATEGORY_PRIMITIVE | NODE_FEATURE_IDX_FLOAT)
#define NODE_FEATURE_TYPE_DOUBLE (NODE_CATEGORY_PRIMITIVE | NODE_FEATURE_IDX_DOUBLE)
#define NODE_FEATURE_TYPE_STRING (NODE_CATEGORY_PRIMITIVE | NODE_FEATURE_IDX_STRING)
#define NODE_FEATURE_TYPE_BOOLEAN (NODE_CATEGORY_PRIMITIVE | NODE_FEATURE_IDX_BOOLEAN)
#define NODE_FEATURE_TYPE_POINTER (NODE_CATEGORY_PRIMITIVE | NODE_FEATURE_IDX_POINTER)
#define NODE_FEATURE_TYPE_VECTOR (NODE_CATEGORY_COLLECTION | NODE_CATEGORY_VECTOR | NODE_FEATURE_IDX_VECTOR)
#define NODE_FEATURE_TYPE_TENSOR (NODE_CATEGORY_COLLECTION | NODE_CATEGORY_TENSOR | NODE_FEATURE_IDX_TENSOR)
#define NODE_FEATURE_TYPE_NODE (NODE_CATEGORY_GRAPH | NODE_FEATURE_IDX_NODE)
#define NODE_FEATURE_TYPE_EDGE (NODE_CATEGORY_GRAPH | NODE_FEATURE_IDX_EDGE)
#define NODE_FEATURE_TYPE_STENCIL (NODE_CATEGORY_GRAPH | NODE_FEATURE_IDX_STENCIL)
#define NODE_FEATURE_TYPE_GENEALOGY (NODE_CATEGORY_GRAPH | NODE_FEATURE_IDX_GENEALOGY)
#define NODE_FEATURE_TYPE_EMERGENCE (NODE_CATEGORY_GRAPH | NODE_FEATURE_IDX_EMERGENCE)
#define NODE_FEATURE_TYPE_LINKED_LIST (NODE_CATEGORY_COLLECTION | NODE_FEATURE_IDX_LINKED_LIST)
#define NODE_FEATURE_TYPE_DICTIONARY (NODE_CATEGORY_COLLECTION | NODE_FEATURE_IDX_DICTIONARY)
#define NODE_FEATURE_TYPE_SET (NODE_CATEGORY_COLLECTION | NODE_FEATURE_IDX_SET)
#define NODE_FEATURE_TYPE_MAP (NODE_CATEGORY_COLLECTION | NODE_FEATURE_IDX_MAP)
#define NODE_FEATURE_TYPE_PARALLEL_LIST (NODE_CATEGORY_COLLECTION | NODE_FEATURE_IDX_PARALLEL_LIST)
#define NODE_FEATURE_TYPE_LIST (NODE_CATEGORY_COLLECTION | NODE_FEATURE_IDX_LIST)
#define NODE_FEATURE_TYPE_POINTER_TOKEN (NODE_CATEGORY_GUARDIAN | NODE_FEATURE_IDX_POINTER_TOKEN)
#define NODE_FEATURE_TYPE_GUARDIAN (NODE_CATEGORY_GUARDIAN | NODE_FEATURE_IDX_GUARDIAN)
#define NODE_FEATURE_TYPE_TOKEN (NODE_CATEGORY_GUARDIAN | NODE_FEATURE_IDX_TOKEN)
#define NODE_FEATURE_TYPE_OBJECT_SET (NODE_CATEGORY_GUARDIAN | NODE_FEATURE_IDX_OBJECT_SET)
#define NODE_FEATURE_TYPE_MEMORY_TOKEN (NODE_CATEGORY_GUARDIAN | NODE_FEATURE_IDX_MEMORY_TOKEN)
#define NODE_FEATURE_TYPE_TOKEN_LOCK (NODE_CATEGORY_GUARDIAN | NODE_FEATURE_IDX_TOKEN_LOCK)
#define NODE_FEATURE_TYPE_MEMORY_MAP (NODE_CATEGORY_GUARDIAN | NODE_FEATURE_IDX_MEMORY_MAP)
#define NODE_FEATURE_TYPE_MESSAGE (NODE_CATEGORY_GUARDIAN | NODE_FEATURE_IDX_MESSAGE)
#define NODE_FEATURE_TYPE_CUSTOM (NODE_CATEGORY_CUSTOM | NODE_FEATURE_IDX_CUSTOM)
#define NODE_FEATURE_TYPE_COMPLEX (NODE_CATEGORY_PRIMITIVE | NODE_FEATURE_IDX_COMPLEX)
#define NODE_FEATURE_TYPE_COMPLEX_DOUBLE (NODE_CATEGORY_PRIMITIVE | NODE_FEATURE_IDX_COMPLEX_DOUBLE)
#define NODE_FEATURE_TYPE_BITFIELD (NODE_CATEGORY_ENCODING | NODE_FEATURE_IDX_BITFIELD)
#define NODE_FEATURE_TYPE_BYTEFIELD (NODE_CATEGORY_ENCODING | NODE_FEATURE_IDX_BYTEFIELD)
#define NODE_FEATURE_TYPE_RADIX_ENCODING (NODE_CATEGORY_ENCODING | NODE_FEATURE_IDX_RADIX_ENCODING)
#define NODE_FEATURE_TYPE_PATTERN_PALETTE_STREAM (NODE_CATEGORY_ENCODING | NODE_FEATURE_IDX_PATTERN_PALETTE_STREAM)
#define NODE_FEATURE_TYPE_ENCODING_ENGINE (NODE_CATEGORY_ENCODING | NODE_FEATURE_IDX_ENCODING_ENGINE)
#define NODE_FEATURE_TYPE_COUNT NODE_FEATURE_IDX_COUNT
#define NODE_FEATURE_TYPE_MUTEX_T (NODE_CATEGORY_GUARDIAN | NODE_FEATURE_IDX_MUTEX_T

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
    GuardianObjectSet * self; // Self-reference for the heap object
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
typedef struct GuardianLinkNode {
    GuardianObjectSet * self; // Self-reference for the node object
    struct GuardianPointerToken * next; // Pointer to the next node in the linked list
    struct GuardianPointerToken * prev; // Pointer to the previous node in the linked list
    size_t size; // Size of the payload in this node
    long long index; // Index of this node in the linked list
    NodeFeatureType feature_type; // Type of payload in this node
    struct GuardianPointerToken * payload; // Pointer to the payload data
    GuardianLinkedList * linked_list; // Pointer to the linked list this node belongs to
} GuardianLinkNode;
typedef struct GuardianLinkedList {
    GuardianPointerToken * left; // Pointer to the left end of the linked list
    GuardianPointerToken * right; // Pointer to the right end of the linked list
    size_t size; // Size of the linked list
    size_t max_size; // Maximum size of the linked list
    NodeFeatureType feature_type; // Type of payload in this linked list
    TokenGuardian* guardian; // The guardian that owns this linked list
} GuardianLinkedList;	

// A GuardianList is a container for a doubly-linked list of payloads.
// It is built upon the primitive GuardianLinkNode from the global cache.
typedef struct GuardianList {
    size_t count;
    GuardianLinkedList* index_to_pointer; // Index to pointer mapping for fast access
    GuardianLinkedList* pointer_to_index; // Pointer to index mapping for fast access
    GuardianMap* index_to_pointer_map; // Map of indices to pointers for fast access
    GuardianObjectSet * self; // Self-reference for the list object
} GuardianList;

typedef struct GuardianParallelList {
    GuardianList * lists;
    int num_lists;
    GuardianObjectSet * self; // Self-reference for the parallel list object
    
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
    GuardianParallelList * stencils_orthagonalities_orientations; // what are the stencils, their relationships, and orientation modes
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



/* ======================= */
/* Parametric Domain Types */
/* ======================= */
#define PD_MAX_DIM 16

typedef enum {
    PD_BOUNDARY_INCLUSIVE,
    PD_BOUNDARY_EXCLUSIVE,
    PD_BOUNDARY_POS_INFINITY,
    PD_BOUNDARY_NEG_INFINITY
} BoundaryType;

typedef enum {
    PD_PERIODIC_NONE,
    PD_PERIODIC_SIMPLE,
    PD_PERIODIC_MIRROR,
    PD_PERIODIC_CYCLIC,
    PD_PERIODIC_POLARIZED
} PeriodicityType;

typedef enum {
    PD_BC_DIRICHLET,
    PD_BC_NEUMANN,
    PD_BC_CUSTOM
} BoundaryConditionType;

typedef struct DiscontinuityNode {
    double position;
    BoundaryType type;
    struct DiscontinuityNode* next;
    struct DiscontinuityNode* prev;
} DiscontinuityNode;

typedef struct AxisDescriptor {
    double start;
    double end;
    bool periodic;
    PeriodicityType periodicity_type;
    BoundaryType start_boundary;
    BoundaryType end_boundary;

    BoundaryConditionType bc_type;
    void (*bc_function)(double*, size_t);
    DiscontinuityNode* discontinuities;

    double (*extrapolate_neg)(double);
    double (*extrapolate_pos)(double);
} AxisDescriptor;

typedef struct ParametricDomain {
    size_t dim;
    AxisDescriptor axes[PD_MAX_DIM];
} ParametricDomain;

/* ===== DAG / Neural Network Types ===== */

typedef struct DAGNode DAGNode;
typedef struct DAGEdge DAGEdge;

typedef void (*DAGForwardFn)(DAGNode* self);
typedef void (*DAGBackwardFn)(DAGNode* self);

typedef struct {
    void* data;
} DAGParams;

struct DAGNode {
    DAGForwardFn forward;
    DAGBackwardFn backward;
    DAGNode** inputs;
    size_t num_inputs;
    DAGParams* params;
    void* output;
    void* grad;
    DAGNode* next;
};

typedef struct {
    struct Node** inputs;
    size_t num_inputs;
    struct Node** outputs;
    size_t num_outputs;
} DagManifestMapping;

typedef struct {
    DagManifestMapping* mappings;
    size_t num_mappings;
    int level_index;
} DagManifestLevel;

typedef struct {
    DagManifestLevel* levels;
    size_t num_levels;
} DagManifest;

typedef struct Dag {
    DagManifest* manifests;
    size_t num_manifests, cap_manifests;
} Dag;

typedef struct NeuralNetwork NeuralNetwork;

typedef void (*NNForwardFn)(struct Node** inputs, size_t num_inputs, struct Node** outputs, size_t num_outputs, void* user);
typedef void (*NNBackwardFn)(struct Node** inputs, size_t num_inputs, struct Node** outputs, size_t num_outputs, void* user);

typedef struct {
    DagManifestMapping* mapping;
    NNForwardFn forward;
    NNBackwardFn backward;
    void* user_data;
} NeuralNetworkStep;

#define NN_MAX_FUNCTIONS 32

typedef struct {
    const char* name;
    NNForwardFn forward;
    NNBackwardFn backward;
} NeuralNetworkFunctionEntry;

typedef struct {
    NeuralNetworkFunctionEntry entries[NN_MAX_FUNCTIONS];
    size_t num_entries;
} NeuralNetworkFunctionRepo;

#define NN_MAX_DAGS 8
#define NN_MAX_STEPS 256

typedef struct NeuralNetwork {
    Dag* dags[NN_MAX_DAGS];
    size_t num_dags;
    NeuralNetworkStep* steps[NN_MAX_DAGS][NN_MAX_STEPS];
    size_t num_steps[NN_MAX_DAGS];
    NeuralNetworkFunctionRepo function_repo;
} NeuralNetwork;

/* ===== Execution Graph ===== */
#define EXEC_GRAPH_MAX_NODES 64

typedef struct {
    DAGNode* nodes[EXEC_GRAPH_MAX_NODES];
    size_t num_nodes;
} ExecutionGraph;

/* ===== Graph Ops Types ===== */

typedef NodeFeatureType OperationSuiteType;

/* Function pointer typedefs for container operations */
typedef void*   (*OpCreateFn)(void);
typedef void    (*OpDestroyFn)(void* container);
typedef void    (*OpClearFn)(void* container);
typedef void    (*OpPushFn)(void* container, void* element);
typedef void*   (*OpPopFn)(void* container);
typedef void*   (*OpShiftFn)(void* container);
typedef void    (*OpUnshiftFn)(void* container, void* element);
typedef void    (*OpInsertFn)(void* container, size_t index, void* element);
typedef void*   (*OpRemoveFn)(void* container, size_t index);
typedef void*   (*OpGetFn)(void* container, size_t index);
typedef void    (*OpSetFn)(void* container, size_t index, void* element);
typedef size_t  (*OpSizeFn)(const void* container);
typedef void    (*OpSortFn)(void* container, int (*cmp)(const void*, const void*));
typedef void*   (*OpSearchFn)(void* container, int (*pred)(const void*, void*), void* user_data);
typedef void    (*OpSliceFn)(void* container, size_t start, size_t end, void** out_array);
typedef void    (*OpStencilFn)(void* container, const size_t* indices, size_t count, void** out_array);
typedef void    (*OpForEachFn)(void* container, void (*fn)(void* element, void* user_data), void* user_data);
typedef void*   (*OpMapFn)(void* container, void* (*map_fn)(void* element, void* user_data), void* user_data);
typedef void*   (*OpFilterFn)(void* container, int (*pred)(void* element, void* user_data), void* user_data);
typedef void*   (*OpReduceFn)(void* container, void* (*reduce_fn)(void* acc, void* element, void* user_data), void* user_data, void* initial);
typedef void*   (*OpCloneFn)(const void* container);
typedef void    (*OpMergeFn)(void* dest, const void* src);
typedef int     (*OpSerializeFn)(const void* container, void* out_buffer, size_t buffer_size);
typedef void*   (*OpDeserializeFn)(const void* in_buffer, size_t buffer_size);
typedef void    (*OpLockFn)(void* container);
typedef void    (*OpUnlockFn)(void* container);
typedef int     (*OpGetRelationshipFn)(void* container, void* a, void* b);
typedef void*   (*OpDiffFn)(void* container);
typedef void*   (*OpIntegrateFn)(void* container);
typedef void*   (*OpInterpolateFn)(void* container, double t);
typedef void*   (*OpProbabilisticSelectFn)(void* container, void* histogram);
typedef void*   (*OpDiscretizeFn)(void* container, double resolution);
typedef void*   (*OpLaplaceFn)(void* container);
typedef void*   (*OpStepFn)(void* container, double t);
typedef void*   (*OpQuantizeFn)(void* container);
typedef void*   (*OpNormalizeFn)(void* container);
typedef void*   (*OpHistogramFn)(void* container, size_t bins);
typedef void*   (*OpQuantileHistogramFn)(void* container, size_t quantiles);
typedef void*   (*OpRankOrderNormalizeFn)(void* container);

/* Graph Math Operations */
typedef enum {
    GRAPH_OP_REGION_DOMAIN = 0,
    GRAPH_OP_REGION_SUBDOMAIN,
    GRAPH_OP_REGION_SLICE,
    GRAPH_OP_REGION_KERNEL
} GraphOpRegion;

typedef void* (*graph_add_fn)(void* a, void* b, GraphOpRegion region);
typedef void* (*graph_subtract_fn)(void* a, void* b, GraphOpRegion region);
typedef void* (*graph_multiply_fn)(void* a, void* b, GraphOpRegion region);
typedef void* (*graph_divide_fn)(void* a, void* b, GraphOpRegion region);
typedef void* (*graph_negate_fn)(void* a, GraphOpRegion region);

typedef void* (*graph_transpose_fn)(void* a, GraphOpRegion region);
typedef void* (*graph_matmul_fn)(void* a, void* b, GraphOpRegion region);
typedef void* (*graph_inverse_fn)(void* a, GraphOpRegion region);
typedef void* (*graph_determinant_fn)(void* a, GraphOpRegion region);

typedef void* (*graph_matrix_pad_fn)(void* a, size_t top, size_t bottom, size_t left, size_t right, GraphOpRegion region);
typedef void* (*graph_make_symmetric_fn)(void* a, GraphOpRegion region);

typedef void* (*graph_union_fn)(void* a, void* b, GraphOpRegion region);
typedef void* (*graph_intersection_fn)(void* a, void* b, GraphOpRegion region);
typedef void* (*graph_difference_fn)(void* a, void* b, GraphOpRegion region);
typedef int   (*graph_subset_fn)(void* a, void* b, GraphOpRegion region);
typedef boolean (*graph_gap_inventory_fn)(void* container);

typedef boolean  (*graph_make_contiguous_fn)(void* container);
typedef boolean  (*graph_make_contiguous_no_wait_fn)(void* container);
typedef boolean  (*graph_make_contiguous_wait_fn)(void* container);
typedef boolean  (*graph_make_contiguous_wait_timeout_fn)(void* container);
typedef boolean  (*graph_make_contiguous_force_fn)(void* container);

typedef void  (*graph_sync_fn)(void* container);
typedef void* (*graph_factorize_fn)(void* a, GraphOpRegion region);
typedef void* (*graph_gcf_fn)(void* a, void* b, GraphOpRegion region);
typedef void* (*graph_lcm_fn)(void* a, void* b, GraphOpRegion region);

/* Bitwise Operations */
typedef void* (*graph_bit_and_fn)(void* a, void* b, GraphOpRegion region);
typedef void* (*graph_bit_or_fn)(void* a, void* b, GraphOpRegion region);
typedef void* (*graph_bit_xor_fn)(void* a, void* b, GraphOpRegion region);
typedef void* (*graph_bit_not_fn)(void* a, GraphOpRegion region);
typedef void* (*graph_bit_shift_left_fn)(void* a, int shift, GraphOpRegion region);
typedef void* (*graph_bit_shift_right_fn)(void* a, int shift, GraphOpRegion region);

typedef enum {
    DIFFUSION_MODEL_AUXIN = 0,
    DIFFUSION_MODEL_SLIME_MOLD,
    DIFFUSION_MODEL_DEPTH_PRESSURE,
    DIFFUSION_MODEL_BREADTH_PRESSURE
} DiffusionModel;

typedef enum {
    DIFFUSION_SOURCE_SINGLE = 0,
    DIFFUSION_SOURCE_UBIQUITOUS,
    DIFFUSION_SOURCE_MAP,
    DIFFUSION_SOURCE_DAG,
    DIFFUSION_SOURCE_BOUNDARY_Y
} DiffusionSourceMode;

typedef enum {
    DIFFUSION_SINK_SINGLE = 0,
    DIFFUSION_SINK_UBIQUITOUS,
    DIFFUSION_SINK_MAP,
    DIFFUSION_SINK_DAG,
    DIFFUSION_SINK_BOUNDARY_Y
} DiffusionSinkMode;

typedef void* (*graph_diffuse_fn)(void* graph, DiffusionModel model, DiffusionSourceMode source, DiffusionSinkMode sink, double rate, int iterations, GraphOpRegion region);

typedef struct {
    graph_add_fn         add;
    graph_subtract_fn    subtract;
    graph_multiply_fn    multiply;
    graph_divide_fn      divide;
    graph_negate_fn      negate;
    graph_transpose_fn   transpose;
    graph_matmul_fn      matmul;
    graph_inverse_fn     inverse;
    graph_determinant_fn determinant;
    graph_matrix_pad_fn  pad;
    graph_make_symmetric_fn make_symmetric;
    graph_union_fn       set_union;
    graph_intersection_fn set_intersection;
    graph_difference_fn  set_difference;
    graph_subset_fn      is_subset;
    graph_gap_inventory_fn gap_inventory;
    graph_make_contiguous_fn make_contiguous;
    graph_make_contiguous_no_wait_fn make_contiguous_no_wait;
    graph_make_contiguous_wait_fn make_contiguous_wait;
    graph_make_contiguous_wait_timeout_fn make_contiguous_wait_timeout;
    graph_make_contiguous_force_fn make_contiguous_force;
    graph_sync_fn         sync;
    graph_factorize_fn    factorize;
    graph_gcf_fn          gcf;
    graph_lcm_fn          lcm;
    graph_diffuse_fn     diffuse;
} GraphMathOps;

typedef struct {
    graph_bit_and_fn        bit_and;
    graph_bit_or_fn         bit_or;
    graph_bit_xor_fn        bit_xor;
    graph_bit_not_fn        bit_not;
    graph_bit_shift_left_fn shift_left;
    graph_bit_shift_right_fn shift_right;
} GraphBitOps;

typedef struct {
    void *         translate_ptr;

    OpCreateFn     create;
    OpDestroyFn    destroy;
    OpClearFn      clear;

    OpPushFn       push;
    OpPopFn        pop;
    OpShiftFn      shift;
    OpUnshiftFn    unshift;
    OpInsertFn     insert;
    OpRemoveFn     remove;

    OpGetFn        get;
    OpSetFn        set;
    OpSizeFn       size;

    OpSortFn       sort;
    OpSearchFn     search;
    OpSliceFn      slice;
    OpStencilFn    stencil;
    OpForEachFn    for_each;
    OpMapFn        map;
    OpFilterFn     filter;
    OpReduceFn     reduce;
    OpCloneFn      clone;
    OpMergeFn      merge;
    OpSerializeFn  serialize;
    OpDeserializeFn deserialize;
    OpLockFn       lock;
    OpUnlockFn     unlock;
    OpGetRelationshipFn      get_relationship;
    OpDiffFn                diff;
    OpIntegrateFn           integrate;
    OpInterpolateFn         interpolate;
    OpProbabilisticSelectFn probabilistic_select;
    OpDiscretizeFn          discretize;
    OpLaplaceFn             laplace;
    OpStepFn                step;
    OpQuantizeFn            quantize;
    OpNormalizeFn           normalize;
    OpHistogramFn           histogram;
    OpQuantileHistogramFn   quantile_histogram;
    OpRankOrderNormalizeFn  rank_order_normalize;

    GraphMathOps math_ops;
    GraphBitOps  bit_ops;
} OperationSuite;


#ifdef __cplusplus
}
#endif

#endif /* GEOMETRY_TYPES_H */
