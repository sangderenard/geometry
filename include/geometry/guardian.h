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

TokenGuardian* guardian_create(void);
void guardian_destroy(TokenGuardian* g);

unsigned long guardian_register_thread(TokenGuardian* g);
void guardian_unregister_thread(TokenGuardian* g, unsigned long token);

void* guardian_alloc(TokenGuardian* g, size_t size, unsigned long* token_out);
void guardian_free(TokenGuardian* g, unsigned long token);

void guardian_send(TokenGuardian* g, unsigned long from, unsigned long to, const void* data, size_t size);
size_t guardian_receive(TokenGuardian* g, unsigned long to, void* buffer, size_t max_size);

/* Simple thread-safe wrappers for raw memory operations */
void* guardian_malloc_simple(size_t size);
void* guardian_calloc_simple(size_t count, size_t size);
void* guardian_realloc_simple(void* ptr, size_t size);
void guardian_free_simple(void* ptr);

#ifdef __cplusplus
}
#endif

#endif /* GEOMETRY_GUARDIAN_H */
