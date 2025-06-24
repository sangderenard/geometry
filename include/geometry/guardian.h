#ifndef GEOMETRY_GUARDIAN_H
#define GEOMETRY_GUARDIAN_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include "geometry/utils.h"

// Token types for threads and memory blocks
typedef struct {
    unsigned long id;
    void* handle; // platform thread handle
} GuardianThreadToken;

typedef struct {
    unsigned long id;
    void* block;
    size_t size;
} GuardianMemoryToken;

typedef struct GuardianMessage {
    unsigned long from_id;
    unsigned long to_id;
    void* data;
    size_t size;
    struct GuardianMessage* next;
} GuardianMessage;

typedef struct {
    GuardianThreadToken* threads;
    size_t num_threads, cap_threads;

    GuardianMemoryToken* memories;
    size_t num_memories, cap_memories;

    GuardianMessage* msg_head;
    GuardianMessage* msg_tail;

    node_mutex_t mutex;
} TokenGuardian;

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

// All dynamic memory management and access must use guardian APIs.
// guardian_alloc, guardian_free, guardian_realloc, etc. are the only allowed methods for dynamic primitives.
// All traversal, neighbor access, and dereferencing must use guardian_deref_neighbor and token-based APIs.
// Add any missing concurrency-checked dereference helpers as needed.

TokenGuardian* guardian_create(void);
void guardian_destroy(TokenGuardian* g);

unsigned long guardian_register_thread(TokenGuardian* g);
void guardian_unregister_thread(TokenGuardian* g, unsigned long token);

// All access to dynamic primitives must be concurrency-checked and token-based.
// The following API provides concurrency-checked dereferencing for neighbor access.
// Returns 1 on success, 0 on dead end or error.
int guardian_deref_neighbor(TokenGuardian* g, unsigned long container_token, size_t neighbor_index, unsigned long* neighbor_token_out);

void guardian_send(TokenGuardian* g, unsigned long from, unsigned long to, const void* data, size_t size);
size_t guardian_receive(TokenGuardian* g, unsigned long to, void* buffer, size_t max_size);

// Factory function declarations
unsigned long guardian_create_object(TokenGuardian* g, GuardianObjectType type);
void guardian_destroy_object(TokenGuardian* g, unsigned long token);

// Declare the recursive parser for nested object requests
unsigned long guardian_parse_nested_object(TokenGuardian* g, const char* request);

#ifdef __cplusplus
}
#endif

#endif /* GEOMETRY_GUARDIAN_H */
