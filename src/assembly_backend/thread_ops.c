#include "assembly_backend/thread_ops.h"
#include "assembly_backend/core.h"
#include "geometry/guardian_platform.h" 
#include "geometry/types.h"
#include "geometry/graph_ops_handler.h"
#include <string.h>

#ifndef THREAD_OPS_DEFAULT_CAPACITY
#define THREAD_OPS_DEFAULT_CAPACITY 1024
#endif


static GuardianThread** lake_threads = NULL;
static GuardianThread** sluice_threads = NULL;
static GuardianThread** finished_threads = NULL;
static enum ThreadOpsExitCode {
    EXIT_CODE_CLEAR = 0,
    EXIT_CODE_ERROR = 1,
    EXIT_CODE_RUNNING = 2,
    EXIT_CODE_PAUSED = 3,
    EXIT_CODE_STOPPED = 4
} ThreadOpsExitCode;
static size_t lake_capacity = 0;
static size_t lake_count = 0;
static size_t sluice_count = 0;
static size_t sluice_capacity = 0;
static mutex_t* sluice_lock = NULL;
static int ** sluice_states = NULL;
static size_t finished_capacity = 0;
static size_t finished_count = 0;
static boolean locked = false;
static int status = 0; // 0 = idle, 1 = running, 2 = paused, 3 = stopped
static GuardianThread** operator_bins = NULL; // List of operators for the threads
static GuardianThread** worker_list = NULL;
// thread loop that checcks sluice status, opens and closes,
// and manages the thread pool "lake" so it's well composed
// for the effective number of threads

static int insert_lake_threads_into_sluice(GuardianThread** threads_to_run, GuardianThread** runners, size_t threads_to_run_count, size_t runners_count) {
    boolean success = lock_sluice_gates();
    int sluice_index = 0;
    int * threads_already_run_mask = NULL;
    threads_already_run_mask = (int*)mg_alloc(sizeof(int) * threads_to_run_count);
    mg_init(threads_already_run_mask, 0, sizeof(int) * threads_to_run_count);
    while (threads_to_run_count > 0 && success) {
        if ( threads_already_run_mask[sluice_index] ) {
            // If this sluice is already running a thread, skip it
            sluice_index = (sluice_index + 1) % sluice_count;
            continue;
        }
        GuardianThread* sluice_to_check = worker_list[sluice_index];
        GuardianOperatorType sluice_type = sluice_to_check->operator_type;
        if (sluice_type == GUARDIAN_OPERATOR_TYPE_NONE) {
            // If the sluice is not assigned, assign it a thread from the lake
            sluice_to_check = threads_to_run[--threads_to_run_count];
            worker_list[sluice_index] = sluice_to_check;
            threads_already_run_mask[sluice_index] = 1; // Mark this sluice as running
            threads_to_run_count--;
        }
        
        
        sluice_index = (sluice_index + 1) % sluice_count;
            
    }
    success = unlock_sluice_gates();
    if (!success) {
        GuardianThread* threads_to_kill = (GuardianThread*)memops_apply_mask(threads_to_run, threads_already_run_mask, threads_to_run_count, 1);
        mg_free(threads_to_kill);
        return EXIT_CODE_ERROR
    }
    
    GuardianThread* threads_to_clear = (GuardianThread*)memops_apply_mask(threads_to_run, threads_already_run_mask, threads_to_run_count, 0);
    mg_free(threads_to_clear);
    return EXIT_CODE_CLEAR;
}

static int find_entry(long long token) {
    if (!token) return -1;
    for (size_t i = 0; i < lake_count; ++i) {
        if (lake_threads[i] && lake_threads[i]->self->guardian_pointer_token->token == token) {
            return (int)i;
        } 
    }
    return -1;
}

mutex_t* thread_ops_mutex_init() {
    mutex_t* mtx = guardian_mutex_init();
    if (!mtx) return NULL;
    return mtx;
}

void thread_ops_init(size_t capacity) {
    if (lake_threads) return;
    if (lake_capacity == 0) lake_capacity = THREAD_OPS_DEFAULT_CAPACITY;
    lake_threads = (GuardianThread**)mg_alloc(sizeof(GuardianThread*) * lake_capacity);
    
    if (!lake_threads) return;
    mg_bulk_initialize(lake_threads, 0, sizeof(GuardianThread*) * capacity);
    sluice_threads = (GuardianThread**)mg_alloc(sizeof(GuardianThread*) * sluice_capacity);
    
    if(!sluice_threads) return;
    mg_bulk_initialize(sluice_threads, 0, sizeof(GuardianThread*) * sluice_capacity);
    finished_threads = (GuardianThread**)mg_alloc(sizeof(GuardianThread*) * finished_capacity);

    if (!finished_threads) return;
    mg_bulk_initialize(finished_threads, 0, sizeof(GuardianThread*) * finished_capacity);
    
    sluice_lock = thread_ops_mutex_init();
    if (!sluice_lock) return;
    sluice_states = (int**)mg_alloc(sizeof(int*) * sluice_capacity);
    mg_bulk_initialize(sluice_states, 0, sizeof(int*) * sluice_capacity);
    
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

int try_lock(TokenGuardian* g, unsigned long lock_token) {
    (void)g;
    mutex_t* mutex = thread_ops_get_mutex((long long)lock_token);
    if (!mutex) return 0;
    return guardian_mutex_trylock(mutex);
}

void lock(TokenGuardian* g, unsigned long lock_token) {
    (void)g;
    mutex_t* mutex = thread_ops_get_mutex((long long)lock_token);
    if (!mutex) return;
    guardian_mutex_lock(mutex);
}

void unlock(TokenGuardian* g, unsigned long lock_token) {
    (void)g;
    mutex_t* mutex = thread_ops_get_mutex((long long)lock_token);
    if (!mutex) return;
    guardian_mutex_unlock(mutex);
}

int is_locked(TokenGuardian* g, unsigned long lock_token) {
    (void)g;
    mutex_t* mutex = thread_ops_get_mutex((long long)lock_token);
    if (!mutex) return 0;
    if (guardian_mutex_trylock(mutex)) {
        guardian_mutex_unlock(mutex);
        return 0;
    }
    return 1;
}

