#include "assembly_backend/thread_ops.h"
#include "assembly_backend/core.h"
#include <string.h>

#ifndef THREAD_OPS_DEFAULT_CAPACITY
#define THREAD_OPS_DEFAULT_CAPACITY 1024
#endif

typedef struct {
    long long token;
    mutex_t* mutex;
    guardian_thread_handle_t thread;
    void* pointer;
    int used;
} ThreadOpsEntry;

static ThreadOpsEntry* entries = NULL;
static size_t entry_capacity = 0;
static size_t entry_count = 0;
static mutex_t* entries_lock = NULL;

static int find_entry(long long token) {
    for (size_t i = 0; i < entry_count; ++i) {
        if (entries[i].used && entries[i].token == token) {
            return (int)i;
        }
    }
    return -1;
}

void thread_ops_init(size_t capacity) {
    if (entries) return;
    if (capacity == 0) capacity = THREAD_OPS_DEFAULT_CAPACITY;
    entries = (ThreadOpsEntry*)mg_alloc(sizeof(ThreadOpsEntry) * capacity);
    if (!entries) return;
    memset(entries, 0, sizeof(ThreadOpsEntry) * capacity);
    entry_capacity = capacity;
    entry_count = 0;
    entries_lock = guardian_mutex_init();
}

static ThreadOpsEntry* allocate_entry(long long token) {
    if (!entries) thread_ops_init(0);
    if (entry_count >= entry_capacity) {
        return NULL;
    }
    ThreadOpsEntry* e = &entries[entry_count++];
    memset(e, 0, sizeof(ThreadOpsEntry));
    e->used = 1;
    e->token = token;
    return e;
}

void thread_ops_register_mutex(long long token, mutex_t* mutex) {
    if (!entries) thread_ops_init(0);
    guardian_mutex_lock(entries_lock);
    int idx = find_entry(token);
    ThreadOpsEntry* e = NULL;
    if (idx >= 0) {
        e = &entries[idx];
    } else {
        e = allocate_entry(token);
    }
    if (e) {
        e->mutex = mutex;
    }
    guardian_mutex_unlock(entries_lock);
}

mutex_t* thread_ops_get_mutex(long long token) {
    if (!entries) return NULL;
    guardian_mutex_lock(entries_lock);
    int idx = find_entry(token);
    mutex_t* mtx = (idx >= 0) ? entries[idx].mutex : NULL;
    guardian_mutex_unlock(entries_lock);
    return mtx;
}

void thread_ops_register_thread(long long token, guardian_thread_handle_t thread) {
    if (!entries) thread_ops_init(0);
    guardian_mutex_lock(entries_lock);
    int idx = find_entry(token);
    ThreadOpsEntry* e = NULL;
    if (idx >= 0) {
        e = &entries[idx];
    } else {
        e = allocate_entry(token);
    }
    if (e) {
        e->thread = thread;
    }
    guardian_mutex_unlock(entries_lock);
}

guardian_thread_handle_t thread_ops_get_thread(long long token) {
    if (!entries) return (guardian_thread_handle_t)0;
    guardian_mutex_lock(entries_lock);
    int idx = find_entry(token);
    guardian_thread_handle_t t = (idx >= 0) ? entries[idx].thread : (guardian_thread_handle_t)0;
    guardian_mutex_unlock(entries_lock);
    return t;
}

void thread_ops_register_pointer(long long token, void* ptr) {
    if (!entries) thread_ops_init(0);
    guardian_mutex_lock(entries_lock);
    int idx = find_entry(token);
    ThreadOpsEntry* e = NULL;
    if (idx >= 0) {
        e = &entries[idx];
    } else {
        e = allocate_entry(token);
    }
    if (e) {
        e->pointer = ptr;
    }
    guardian_mutex_unlock(entries_lock);
}

void* thread_ops_get_pointer(long long token) {
    if (!entries) return NULL;
    guardian_mutex_lock(entries_lock);
    int idx = find_entry(token);
    void* p = (idx >= 0) ? entries[idx].pointer : NULL;
    guardian_mutex_unlock(entries_lock);
    return p;
}

int guardian_try_lock(TokenGuardian* g, unsigned long lock_token) {
    (void)g;
    mutex_t* mutex = thread_ops_get_mutex((long long)lock_token);
    if (!mutex) return 0;
    return guardian_mutex_trylock(mutex);
}

void guardian_lock(TokenGuardian* g, unsigned long lock_token) {
    (void)g;
    mutex_t* mutex = thread_ops_get_mutex((long long)lock_token);
    if (!mutex) return;
    guardian_mutex_lock(mutex);
}

void guardian_unlock(TokenGuardian* g, unsigned long lock_token) {
    (void)g;
    mutex_t* mutex = thread_ops_get_mutex((long long)lock_token);
    if (!mutex) return;
    guardian_mutex_unlock(mutex);
}

int guardian_is_locked(TokenGuardian* g, unsigned long lock_token) {
    (void)g;
    mutex_t* mutex = thread_ops_get_mutex((long long)lock_token);
    if (!mutex) return 0;
    if (guardian_mutex_trylock(mutex)) {
        guardian_mutex_unlock(mutex);
        return 0;
    }
    return 1;
}

