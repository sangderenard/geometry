#include "geometry/guardian.h"
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

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

void* guardian_alloc(TokenGuardian* g, size_t size, unsigned long* token_out) {
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

void guardian_free(TokenGuardian* g, unsigned long token) {
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
