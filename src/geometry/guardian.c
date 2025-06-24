#include "geometry/guardian.h"
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

// All dynamic memory management and access must use guardian APIs
// guardian_alloc, guardian_free, guardian_realloc, etc. are the only allowed methods for dynamic primitives
// All traversal, neighbor access, and dereferencing must use guardian_deref_neighbor and token-based APIs
// Add any missing concurrency-checked dereference helpers as needed

static void guardian_mutex_init(node_mutex_t* m) {
#ifdef _WIN32
    InitializeCriticalSection(m);
#else
    pthread_mutex_init(m, NULL);
#endif
}

static void guardian_mutex_destroy(node_mutex_t* m) {
#ifdef _WIN32
    DeleteCriticalSection(m);
#else
    pthread_mutex_destroy(m);
#endif
}

static void guardian_mutex_lock(node_mutex_t* m) {
#ifdef _WIN32
    EnterCriticalSection(m);
#else
    pthread_mutex_lock(m);
#endif
}

static void guardian_mutex_unlock(node_mutex_t* m) {
#ifdef _WIN32
    LeaveCriticalSection(m);
#else
    pthread_mutex_unlock(m);
#endif
}

TokenGuardian* guardian_create(void) {
    TokenGuardian* g = calloc(1, sizeof(TokenGuardian));
    if (!g) return NULL;
    guardian_mutex_init(&g->mutex);
    return g;
}

void guardian_destroy(TokenGuardian* g) {
    if (!g) return;
    guardian_mutex_lock(&g->mutex);
    free(g->threads);
    for (size_t i = 0; i < g->num_memories; ++i) free(g->memories[i].block);
    free(g->memories);
    while (g->msg_head) {
        GuardianMessage* tmp = g->msg_head;
        g->msg_head = tmp->next;
        free(tmp->data);
        free(tmp);
    }
    guardian_mutex_unlock(&g->mutex);
    guardian_mutex_destroy(&g->mutex);
    free(g);
}

static unsigned long guardian_next_id(TokenGuardian* g, unsigned long* counter) {
    return ++(*counter);
}

unsigned long guardian_register_thread(TokenGuardian* g) {
    if (!g) return 0;
    static unsigned long thread_counter = 0;
    guardian_mutex_lock(&g->mutex);
    if (g->num_threads == g->cap_threads) {
        size_t new_cap = g->cap_threads ? g->cap_threads * 2 : 4;
        g->threads = realloc(g->threads, new_cap * sizeof(GuardianThreadToken));
        g->cap_threads = new_cap;
    }
    unsigned long id = guardian_next_id(g, &thread_counter);
#ifdef _WIN32
    void* handle = GetCurrentThread();
#else
    pthread_t handle = pthread_self();
#endif
    g->threads[g->num_threads++] = (GuardianThreadToken){id, (void*)handle};
    guardian_mutex_unlock(&g->mutex);
    return id;
}

void guardian_unregister_thread(TokenGuardian* g, unsigned long token) {
    if (!g || token == 0) return;
    guardian_mutex_lock(&g->mutex);
    for (size_t i = 0; i < g->num_threads; ++i) {
        if (g->threads[i].id == token) {
            g->threads[i] = g->threads[g->num_threads - 1];
            g->num_threads--;
            break;
        }
    }
    guardian_mutex_unlock(&g->mutex);
}

static void* guardian_alloc_internal(TokenGuardian* g, size_t size, unsigned long* token_out) {
    if (!g || size == 0) return NULL;
    static unsigned long mem_counter = 0;
    void* block = calloc(1, size);
    if (!block) return NULL;
    guardian_mutex_lock(&g->mutex);
    if (g->num_memories == g->cap_memories) {
        size_t new_cap = g->cap_memories ? g->cap_memories * 2 : 4;
        g->memories = realloc(g->memories, new_cap * sizeof(GuardianMemoryToken));
        g->cap_memories = new_cap;
    }
    unsigned long id = guardian_next_id(g, &mem_counter);
    g->memories[g->num_memories++] = (GuardianMemoryToken){id, block, size};
    guardian_mutex_unlock(&g->mutex);
    if (token_out) *token_out = id;
    return block;
}

static void guardian_free_internal(TokenGuardian* g, unsigned long token) {
    if (!g || token == 0) return;
    guardian_mutex_lock(&g->mutex);
    for (size_t i = 0; i < g->num_memories; ++i) {
        if (g->memories[i].id == token) {
            free(g->memories[i].block);
            g->memories[i] = g->memories[g->num_memories - 1];
            g->num_memories--;
            break;
        }
    }
    guardian_mutex_unlock(&g->mutex);
}

void guardian_send(TokenGuardian* g, unsigned long from, unsigned long to, const void* data, size_t size) {
    if (!g || !data || size == 0) return;
    GuardianMessage* msg = malloc(sizeof(GuardianMessage));
    msg->from_id = from;
    msg->to_id = to;
    msg->data = malloc(size);
    memcpy(msg->data, data, size);
    msg->size = size;
    msg->next = NULL;
    guardian_mutex_lock(&g->mutex);
    if (g->msg_tail) {
        g->msg_tail->next = msg;
    } else {
        g->msg_head = msg;
    }
    g->msg_tail = msg;
    guardian_mutex_unlock(&g->mutex);
}

size_t guardian_receive(TokenGuardian* g, unsigned long to, void* buffer, size_t max_size) {
    if (!g || !buffer) return 0;
    guardian_mutex_lock(&g->mutex);
    GuardianMessage* prev = NULL;
    GuardianMessage* curr = g->msg_head;
    while (curr) {
        if (curr->to_id == to || to == 0) {
            size_t copy = curr->size < max_size ? curr->size : max_size;
            memcpy(buffer, curr->data, copy);
            if (prev) prev->next = curr->next; else g->msg_head = curr->next;
            if (g->msg_tail == curr) g->msg_tail = prev;
            free(curr->data);
            free(curr);
            guardian_mutex_unlock(&g->mutex);
            return copy;
        }
        prev = curr;
        curr = curr->next;
    }
    guardian_mutex_unlock(&g->mutex);
    return 0;
}

// Add concurrency-checked dereference API for neighbor access
// Returns 1 on success, 0 on dead end or error
int guardian_deref_neighbor(TokenGuardian* g, unsigned long container_token, size_t neighbor_index, unsigned long* neighbor_token_out) {
    if (!g || !neighbor_token_out) return 0;
    // Example: look up the container by token, then get the neighbor token at neighbor_index
    // This is a stub; actual implementation depends on how containers and neighbors are tracked
    // For now, always fail (to be implemented with actual container/neighbor tracking)
    *neighbor_token_out = 0;
    return 0;
}

// Add factory functions for Guardian-managed objects

// Enumeration for object types managed by the Guardian
typedef enum {
    GUARDIAN_OBJECT_LIST,
    GUARDIAN_OBJECT_DICT,
    GUARDIAN_OBJECT_SET,
    GUARDIAN_OBJECT_NODE,
    GUARDIAN_OBJECT_EDGE,
    GUARDIAN_OBJECT_GENELOGY,
    GUARDIAN_OBJECT_PARAMETRIC_DOMAIN,
    GUARDIAN_OBJECT_SIMPLEGRAPH,
    GUARDIAN_OBJECT_DAG,
    GUARDIAN_OBJECT_DAWG,
    GUARDIAN_OBJECT_NN,
    GUARDIAN_OBJECT_MODSPACE,
    GUARDIAN_OBJECT_MESHGRID,
    GUARDIAN_OBJECT_MESH,
    GUARDIAN_OBJECT_TENSOR,
    GUARDIAN_OBJECT_VECTOR,
    GUARDIAN_OBJECT_FUNCTION_SEQUENCE
} GuardianObjectType;

// Factory function declarations
unsigned long guardian_create_object(TokenGuardian* g, GuardianObjectType type);
void guardian_destroy_object(TokenGuardian* g, unsigned long token);

// Factory function definitions
unsigned long guardian_create_object(TokenGuardian* g, GuardianObjectType type) {
    if (!g) return 0;
    unsigned long token;
    void* object = NULL;

    switch (type) {
        case GUARDIAN_OBJECT_LIST:
            object = calloc(1, sizeof(void*)); // Placeholder for list
            break;
        case GUARDIAN_OBJECT_DICT:
            object = calloc(1, sizeof(void*)); // Placeholder for dict
            break;
        case GUARDIAN_OBJECT_SET:
            object = calloc(1, sizeof(void*)); // Placeholder for set
            break;
        case GUARDIAN_OBJECT_NODE:
            object = calloc(1, sizeof(void*)); // Placeholder for node
            break;
        case GUARDIAN_OBJECT_EDGE:
            object = calloc(1, sizeof(void*)); // Placeholder for edge
            break;
        case GUARDIAN_OBJECT_GENELOGY:
            object = calloc(1, sizeof(void*)); // Placeholder for geneology
            break;
        case GUARDIAN_OBJECT_PARAMETRIC_DOMAIN:
            object = calloc(1, sizeof(void*)); // Placeholder for parametric domain
            break;
        case GUARDIAN_OBJECT_SIMPLEGRAPH:
            object = calloc(1, sizeof(void*)); // Placeholder for simplegraph
            break;
        case GUARDIAN_OBJECT_DAG:
            object = calloc(1, sizeof(void*)); // Placeholder for dag
            break;
        case GUARDIAN_OBJECT_DAWG:
            object = calloc(1, sizeof(void*)); // Placeholder for dawg
            break;
        case GUARDIAN_OBJECT_NN:
            object = calloc(1, sizeof(void*)); // Placeholder for nn
            break;
        case GUARDIAN_OBJECT_MODSPACE:
            object = calloc(1, sizeof(void*)); // Placeholder for modspace
            break;
        case GUARDIAN_OBJECT_MESHGRID:
            object = calloc(1, sizeof(void*)); // Placeholder for meshgrid
            break;
        case GUARDIAN_OBJECT_MESH:
            object = calloc(1, sizeof(void*)); // Placeholder for mesh
            break;
        case GUARDIAN_OBJECT_TENSOR:
            object = calloc(1, sizeof(void*)); // Placeholder for tensor
            break;
        case GUARDIAN_OBJECT_VECTOR:
            object = calloc(1, sizeof(void*)); // Placeholder for vector
            break;
        case GUARDIAN_OBJECT_FUNCTION_SEQUENCE:
            object = calloc(1, sizeof(void*)); // Placeholder for function sequence
            break;
        default:
            return 0;
    }

    if (!object) return 0;
    guardian_alloc_internal(g, sizeof(object), &token);
    return token;
}

void guardian_destroy_object(TokenGuardian* g, unsigned long token) {
    if (!g || token == 0) return;
    guardian_free_internal(g, token);
}

// Add a recursive parser for nested object requests

// Recursive parser for nested object requests
unsigned long guardian_parse_nested_object(TokenGuardian* g, const char* request) {
    if (!g || !request) return 0;

    // Placeholder for parsing logic
    // This function will recursively parse the request string and build the desired structure
    // using Guardian-managed tokens. The structure types include:
    // - list: [item, item]
    // - dict: {key:item}
    // - set: (dict, dict, dict)
    // - vector: <item, item, item>
    // - tensor: ~/batch, channel, a, b, c, ..., n/~/type/~
    // - graph: !![node^item^, node^item^, ..., n], [edge^edgeattr^, ..., m]!!

    // Example pseudocode for parsing:
    // 1. Identify the structure type based on delimiters (e.g., [], {}, (), <>, ~/~/, !! !!).
    // 2. Recursively parse nested items within the structure.
    // 3. Use guardian_create_object() to create tokens for each item.
    // 4. Return the token for the top-level structure.

    // Interface with contiguous()
    // The contiguous() function will handle buffer management for contiguous composition.
    // It will interface with deeply internal factors of buffer before rebuild length and
    // buffer length growth and decay factors. This mechanism ensures efficient memory
    // management and dynamic resizing of buffers.
    // Note: contiguous() is not implemented here, but its role is assumed in the parsing process.

    // Placeholder return value
    return 0;
}
