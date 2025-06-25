#include "geometry/utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif


// === Node Data Structures ===

// All type definitions are provided in the public header.


// --- DAG Manifest Structures Implementation ---
#include <stdlib.h>
#include <string.h>

Dag* dag_create(void) {
    Dag* dag = (Dag*)calloc(1, sizeof(Dag));
    return dag;
}

void dag_destroy(Dag* dag) {
    if (!dag) return;
    for (size_t i = 0; i < dag->num_manifests; ++i) {
        DagManifest* manifest = &dag->manifests[i];
        for (size_t l = 0; l < manifest->num_levels; ++l) {
            DagManifestLevel* level = &manifest->levels[l];
            for (size_t m = 0; m < level->num_mappings; ++m) {
                DagManifestMapping* mapping = &level->mappings[m];
                free(mapping->inputs);
                free(mapping->outputs);
            }
            free(level->mappings);
        }
        free(manifest->levels);
    }
    free(dag->manifests);
    free(dag);
}
void dag_add_manifest(Dag* dag, DagManifest* manifest) {
    if (!dag || !manifest) return;
    if (dag->num_manifests == dag->cap_manifests) {
        size_t new_cap = dag->cap_manifests ? dag->cap_manifests * 2 : 4;
        DagManifest* tmp = (DagManifest*)realloc(dag->manifests, new_cap * sizeof(DagManifest));
        if (!tmp) return;
        dag->manifests = tmp;
        dag->cap_manifests = new_cap;
    }
    dag->manifests[dag->num_manifests++] = *manifest;
}

size_t dag_num_manifests(const Dag* dag) {
    return dag ? dag->num_manifests : 0;
}

DagManifest* dag_get_manifest(const Dag* dag, size_t idx) {
    if (!dag || idx >= dag->num_manifests) return NULL;
    return &dag->manifests[idx];
}

size_t dag_manifest_num_levels(const DagManifest* manifest) {
    return manifest ? manifest->num_levels : 0;
}

DagManifestLevel* dag_manifest_get_level(const DagManifest* manifest, size_t level_idx) {
    if (!manifest || level_idx >= manifest->num_levels) return NULL;
    return &manifest->levels[level_idx];
}

size_t dag_level_num_mappings(const DagManifestLevel* level) {
    return level ? level->num_mappings : 0;
}

DagManifestMapping* dag_level_get_mapping(const DagManifestLevel* level) {
    return level ? level->mappings : NULL;
}

void dag_gather(const DagManifestMapping* mapping, void* out) {
    (void)mapping;
    (void)out;
    /* TODO: gather data from inputs */
}

void dag_scatter(const DagManifestMapping* mapping, void* data) {
    (void)mapping;
    (void)data;
    /* TODO: scatter data to outputs */
}

// --- NeuralNetwork implementation ---
#include <string.h>

NeuralNetwork* neuralnetwork_create(void) {
    NeuralNetwork* nn = (NeuralNetwork*)calloc(1, sizeof(NeuralNetwork));
    return nn;
}

void neuralnetwork_destroy(NeuralNetwork* nn) {
    if (!nn) return;
    for (size_t d = 0; d < nn->num_dags; ++d) {
        dag_destroy(nn->dags[d]);
        for (size_t s = 0; s < nn->num_steps[d]; ++s) {
            free(nn->steps[d][s]);
        }
    }
    free(nn);
}

void neuralnetwork_register_function(NeuralNetwork* nn, const char* name, NNForwardFn forward, NNBackwardFn backward) {
    if (!nn || !name) return;
    if (nn->function_repo.num_entries >= NN_MAX_FUNCTIONS) return;
    NeuralNetworkFunctionEntry* e = &nn->function_repo.entries[nn->function_repo.num_entries++];
    e->name = name;
    e->forward = forward;
    e->backward = backward;
}

void neuralnetwork_set_step_function(NeuralNetwork* nn, size_t dag_idx, size_t step_idx, const char* function_name, void* user_data) {
    if (!nn || dag_idx >= nn->num_dags || step_idx >= NN_MAX_STEPS) return;
    NeuralNetworkStep* step = nn->steps[dag_idx][step_idx];
    if (!step) {
        step = (NeuralNetworkStep*)calloc(1, sizeof(NeuralNetworkStep));
        nn->steps[dag_idx][step_idx] = step;
        if (step_idx >= nn->num_steps[dag_idx]) nn->num_steps[dag_idx] = step_idx + 1;
    }
    for (size_t i = 0; i < nn->function_repo.num_entries; ++i) {
        NeuralNetworkFunctionEntry* e = &nn->function_repo.entries[i];
        if (strcmp(e->name, function_name) == 0) {
            step->forward = e->forward;
            step->backward = e->backward;
            step->user_data = user_data;
            break;
        }
    }
}

void neuralnetwork_forward(NeuralNetwork* nn) {
    (void)nn; // TODO
}

void neuralnetwork_backward(NeuralNetwork* nn) {
    (void)nn; // TODO
}

void neuralnetwork_forwardstep(NeuralNetwork* nn, size_t dag_idx, size_t step_idx) {
    (void)nn; (void)dag_idx; (void)step_idx; // TODO
}

void neuralnetwork_backwardstep(NeuralNetwork* nn, size_t dag_idx, size_t step_idx) {
    (void)nn; (void)dag_idx; (void)step_idx; // TODO
}

// === Consolidated Guardian Implementation ===

// === GuardianNode structure ===

// === GuardianToken structure ===

// === GuardianObjectSet structure ===

// === TokenGuardian structure ===

// === GuardianPointerToken structure ===

// === GuardianLinkedList structure ===

// === GuardianDict structure ===

// === GuardianList structure ===

// === GuardianParallelList structure ===

// === GuardianSet structure ===

// === GuardianMap structure ===

// === GuardianStencil structure ===

// === NodeOrientationNature enum ===

// === GuardianStencilSet structure ===

// === Node structure ===

// === EdgeAttribute structure ===

// === Subedge structure ===

// === EdgeType structure ===

// === Edge structure ===

// === GuardianEdge structure ===

// === GuardianGeneology structure ===

// === GuardianSimpleGraph structure ===



static void ___guardian_mutex_init_internal_ifyouareseeingthislineyouareviolatingcodingstandards(node_mutex_t* m) {
#ifdef _WIN32
    InitializeCriticalSection(m);
#else
    pthread_mutex_init(m, NULL);
#endif
}

static void ___guardian_mutex_destroy_internal_ifyouareseeingthislineyouareviolatingcodingstandards(node_mutex_t* m) {
#ifdef _WIN32
    DeleteCriticalSection(m);
#else
    pthread_mutex_destroy(m);
#endif
}

static void ___guardian_mutex_lock_internal_ifyouareseeingthislineyouareviolatingcodingstandards(node_mutex_t* m) {
#ifdef _WIN32
    EnterCriticalSection(m);
#else
    pthread_mutex_lock(m);
#endif
}

static void ___guardian_mutex_unlock_internal_ifyouareseeingthislineyouareviolatingcodingstandards(node_mutex_t* m) {
#ifdef _WIN32
    LeaveCriticalSection(m);
#else
    pthread_mutex_unlock(m);
#endif
}

TokenGuardian* ___guardian_create_internal_ifyouareseeingthislineyouareviolatingcodingstandards(void) {
    TokenGuardian* g = calloc(1, sizeof(TokenGuardian));
    if (!g) return NULL;
    ___guardian_mutex_init_internal_ifyouareseeingthislineyouareviolatingcodingstandards(&g->mutex);
    return g;
}

void ___guardian_destroy_internal_ifyouareseeingthislineyouareviolatingcodingstandards(TokenGuardian* g) {
    if (!g) return;
    ___guardian_mutex_lock_internal_ifyouareseeingthislineyouareviolatingcodingstandards(&g->mutex);
    free(g->threads);
    for (size_t i = 0; i < g->num_memories; ++i) free(g->memories[i].block);
    free(g->memories);
    while (g->msg_head) {
        GuardianMessage* tmp = g->msg_head;
        g->msg_head = tmp->next;
        free(tmp->data);
        free(tmp);
    }
    ___guardian_mutex_unlock_internal_ifyouareseeingthislineyouareviolatingcodingstandards(&g->mutex);
    ___guardian_mutex_destroy_internal_ifyouareseeingthislineyouareviolatingcodingstandards(&g->mutex);
    free(g);
}

static unsigned long thread_counter = 0;

static unsigned long ___guardian_next_id_internal_ifyouareseeingthislineyouareviolatingcodingstandards(TokenGuardian* g, unsigned long* counter) {
    return ++(*counter);
}

unsigned long ___guardian_register_thread_internal_ifyouareseeingthislineyouareviolatingcodingstandards(TokenGuardian* g) {
    if (!g) return 0;
    ___guardian_mutex_lock_internal_ifyouareseeingthislineyouareviolatingcodingstandards(&g->mutex);
    if (g->num_threads == g->cap_threads) {
        size_t new_cap = g->cap_threads ? g->cap_threads * 2 : 4;
        g->threads = realloc(g->threads, new_cap * sizeof(GuardianThreadToken));
        g->cap_threads = new_cap;
    }
#ifdef _WIN32
    void* handle = GetCurrentThread();
#else
    pthread_t handle = pthread_self();
#endif
    unsigned long id = ___guardian_next_id_internal_ifyouareseeingthislineyouareviolatingcodingstandards(g, &thread_counter);
    g->threads[g->num_threads++] = (GuardianThreadToken){id, handle};
    ___guardian_mutex_unlock_internal_ifyouareseeingthislineyouareviolatingcodingstandards(&g->mutex);
    return id;
}

void ___guardian_unregister_thread_internal_ifyouareseeingthislineyouareviolatingcodingstandards(TokenGuardian* g, unsigned long token) {
    if (!g || token == 0) return;
    ___guardian_mutex_lock_internal_ifyouareseeingthislineyouareviolatingcodingstandards(&g->mutex);
    for (size_t i = 0; i < g->num_threads; ++i) {
        if (g->threads[i].id == token) {
            g->threads[i] = g->threads[g->num_threads - 1];
            g->num_threads--;
            break;
        }
    }
    ___guardian_mutex_unlock_internal_ifyouareseeingthislineyouareviolatingcodingstandards(&g->mutex);
}

static void* ___guardian_alloc_internal_ifyouareseeingthislineyouareviolatingcodingstandards(TokenGuardian* g, size_t size, unsigned long* token_out) {
    if (!g || size == 0) return NULL;
    static unsigned long mem_counter = 0;
    void* block = calloc(1, size);
    if (!block) return NULL;
    ___guardian_mutex_lock_internal_ifyouareseeingthislineyouareviolatingcodingstandards(&g->mutex);
    if (g->num_memories == g->cap_memories) {
        size_t new_cap = g->cap_memories ? g->cap_memories * 2 : 4;
        g->memories = realloc(g->memories, new_cap * sizeof(GuardianMemoryToken));
        g->cap_memories = new_cap;
    }
    unsigned long id = ___guardian_next_id_internal_ifyouareseeingthislineyouareviolatingcodingstandards(g, &mem_counter);
    g->memories[g->num_memories++] = (GuardianMemoryToken){id, block, size};
    ___guardian_mutex_unlock_internal_ifyouareseeingthislineyouareviolatingcodingstandards(&g->mutex);
    if (token_out) *token_out = id;
    return block;
}

static void ___guardian_free_internal_ifyouareseeingthislineyouareviolatingcodingstandards(TokenGuardian* g, unsigned long token) {
    if (!g || token == 0) return;
    ___guardian_mutex_lock_internal_ifyouareseeingthislineyouareviolatingcodingstandards(&g->mutex);
    for (size_t i = 0; i < g->num_memories; ++i) {
        if (g->memories[i].id == token) {
            free(g->memories[i].block);
            g->memories[i] = g->memories[g->num_memories - 1];
            g->num_memories--;
            break;
        }
    }
    ___guardian_mutex_unlock_internal_ifyouareseeingthislineyouareviolatingcodingstandards(&g->mutex);
}

void ___guardian_send_internal_ifyouareseeingthislineyouareviolatingcodingstandards(TokenGuardian* g, unsigned long from, unsigned long to, const void* data, size_t size) {
    if (!g || !data || size == 0) return;
    GuardianMessage* msg = guardian_alloc_internal(g, sizeof(GuardianMessage), NULL);
    msg->from_id = from;
    msg->to_id = to;
    msg->data = guardian_alloc_internal(g, size, NULL);
    memcpy(msg->data, data, size);
    msg->size = size;
    msg->next = NULL;
    guardian_mutex_lock(&g->mutex);
    if (g->msg_tail) g->msg_tail->next = msg; else g->msg_head = msg;
    g->msg_tail = msg;
    guardian_mutex_unlock(&g->mutex);
}

size_t ___guardian_receive_internal_ifyouareseeingthislineyouareviolatingcodingstandards(TokenGuardian* g, unsigned long to, void* buffer, size_t max_size) {
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
            guardian_free_internal(g, curr->data);
            guardian_free_internal(g, curr);
            guardian_mutex_unlock(&g->mutex);
            return copy;
        }
        prev = curr;
        curr = curr->next;
    }
    ___guardian_mutex_unlock_internal_ifyouareseeingthislineyouareviolatingcodingstandards(&g->mutex);
    return 0;
}

int ___guardian_deref_neighbor_internal_ifyouareseeingthislineyouareviolatingcodingstandards(TokenGuardian* g, unsigned long container_token, size_t neighbor_index, unsigned long* neighbor_token_out) {
    if (!g || !neighbor_token_out) return 0;
    *neighbor_token_out = 0;
    return 0;
}

// Factory functions for Guardian-managed objects
unsigned long ___guardian_create_object_internal_ifyouareseeingthislineyouareviolatingcodingstandards(
    TokenGuardian* g = NULL,
    int type = OBJECT_TYPE_NODE,
    int count = 1,
    void* referrent = NULL,
    void** params = NULL,
    int param_count = 1
) {
    if (!g) {
        g = ___guardian_create_internal_ifyouareseeingthislineyouareviolatingcodingstandards();
    }
    switch (type) {
        case OBJECT_TYPE_NODE:)
            return ___guardian_create_node_internal_ifyouareseeingthislineyouareviolatingcodingstandards(g, count, params, param_count);
        case OBJECT_TYPE_EDGE:
            return ___guardian_create_edge_internal_ifyouareseeingthislineyouareviolatingcodingstandards(g, count, params, param_count);
        case OBJECT_TYPE_POINTER_TOKEN:
            return ___guardian_create_pointer_token_ifyouareseeingthislineyouareviolatingcodingstandards(g, referrent, params, param_count);
        case OBJECT_TYPE_STENCIL:
            return ___guardian_create_stencil_internal_ifyouareseeingthislineyouareviolatingcodingstandards(g, referrent, params, param_count);
        case OBJECT_TYPE_PARAMETRIC_DOMAIN:
            return ___guardian_create_parametric_domain_internal_ifyouareseeingthislineyouareviolatingcodingstandards(g, referrent, params, param_count);
        default:
            return 1; // Unsupported type
    }
}

void ___guardian_destroy_object_internal_ifyouareseeingthislineyouareviolatingcodingstandards(TokenGuardian* g, unsigned long token) {
    if (!g || token == 0) return;
    ___guardian_free_internal_ifyouareseeingthislineyouareviolatingcodingstandards(g, token);
}

unsigned long ___guardian_parse_nested_object_internal_ifyouareseeingthislineyouareviolatingcodingstandards(TokenGuardian* g, const char* request) {
    if (!g || !request) return 0;
    return 0;
}
// === End Guardian Integration ===

// Public Guardian API wrappers

typedef unsigned long guardianToken;

guardianToken getObject(int type) {
    TokenGuardian* g = ___guardian_create_internal_ifyouareseeingthislineyouareviolatingcodingstandards();
    if (!g) return 0;
    return ___guardian_create_object_internal_ifyouareseeingthislineyouareviolatingcodingstandards(g, type);
}

guardianToken parseNestedObjectString(const char* str) {
    TokenGuardian* g = ___guardian_create_internal_ifyouareseeingthislineyouareviolatingcodingstandards();
    if (!g) return 0;
    return ___guardian_parse_nested_object_internal_ifyouareseeingthislineyouareviolatingcodingstandards(g, str);
}

guardianToken getNestedObject(const char* name) {
    // alias for parseNestedObjectString for now
    return parseNestedObjectString(name);
}

int returnObject(guardianToken token) {
    TokenGuardian* g = ___guardian_create_internal_ifyouareseeingthislineyouareviolatingcodingstandards();
    if (!g || token == 0) return 0;
    ___guardian_destroy_object_internal_ifyouareseeingthislineyouareviolatingcodingstandards(g, token);
    return 1;
}

// === Friendly Functions for Guardian ===

// Generate a token for a pointer without exposing the pointer itself
unsigned long guardian_create_pointer_token(TokenGuardian* g, void* ptr) {
    if (!g || !ptr) return 0;
    unsigned long token;
    void* block = ___guardian_alloc_internal_ifyouareseeingthislineyouareviolatingcodingstandards(g, sizeof(void*), &token);
    if (!block) return 0;
    memcpy(block, &ptr, sizeof(void*));
    return token;
}

// Generate a token for a lock without exposing the mutex
unsigned long guardian_create_lock_token(TokenGuardian* g) {
    if (!g) return 0;
    unsigned long token;
    node_mutex_t* mutex = (node_mutex_t*)___guardian_alloc_internal_ifyouareseeingthislineyouareviolatingcodingstandards(g, sizeof(node_mutex_t), &token);
    if (!mutex) return 0;
#ifdef _WIN32
    InitializeCriticalSection(mutex);
#else
    pthread_mutex_init(mutex, NULL);
#endif
    return token;
}

// === Friendly Lock Functions ===

// Try to acquire a lock using a token
int guardian_try_lock(TokenGuardian* g, unsigned long lock_token) {
    if (!g || lock_token == 0) return 0;
    node_mutex_t* mutex = (node_mutex_t*)___guardian_deref_neighbor_internal_ifyouareseeingthislineyouareviolatingcodingstandards(g, lock_token, 0, NULL);
    if (!mutex) return 0;
#ifdef _WIN32
    return TryEnterCriticalSection(mutex);
#else
    return pthread_mutex_trylock(mutex) == 0;
#endif
}

// Acquire a lock using a token
void guardian_lock(TokenGuardian* g, unsigned long lock_token) {
    if (!g || lock_token == 0) return;
    node_mutex_t* mutex = (node_mutex_t*)___guardian_deref_neighbor_internal_ifyouareseeingthislineyouareviolatingcodingstandards(g, lock_token, 0, NULL);
    if (!mutex) return;
#ifdef _WIN32
    EnterCriticalSection(mutex);
#else
    pthread_mutex_lock(mutex);
#endif
}

// Release a lock using a token
void guardian_unlock(TokenGuardian* g, unsigned long lock_token) {
    if (!g || lock_token == 0) return;
    node_mutex_t* mutex = (node_mutex_t*)___guardian_deref_neighbor_internal_ifyouareseeingthislineyouareviolatingcodingstandards(g, lock_token, 0, NULL);
    if (!mutex) return;
#ifdef _WIN32
    LeaveCriticalSection(mutex);
#else
    pthread_mutex_unlock(mutex);
#endif
}

// Check if a lock is held using a token
int guardian_is_locked(TokenGuardian* g, unsigned long lock_token) {
    if (!g || lock_token == 0) return 0;
    node_mutex_t* mutex = (node_mutex_t*)___guardian_deref_neighbor_internal_ifyouareseeingthislineyouareviolatingcodingstandards(g, lock_token, 0, NULL);
    if (!mutex) return 0;
#ifdef _WIN32
    return mutex->LockCount > 0;
#else
    return pthread_mutex_trylock(mutex) != 0;
#endif
}

// === Object Dereferencer ===

// Exchange a token for the object a pointer is pointing to
void* guardian_dereference_object(TokenGuardian* g, unsigned long pointer_token) {
    if (!g || pointer_token == 0) return NULL;
    void* ptr = NULL;
    void* block = ___guardian_deref_neighbor_internal_ifyouareseeingthislineyouareviolatingcodingstandards(g, pointer_token, 0, NULL);
    if (block) memcpy(&ptr, block, sizeof(void*));
    return ptr;
}
// === Friendly Functions for Guardian ===

// Generate a token for a pointer without exposing the pointer itself
unsigned long guardian_create_pointer_token(TokenGuardian* g, void* ptr) {
    if (!g || !ptr) return 0;
    unsigned long token;
    void* block = ___guardian_alloc_internal_ifyouareseeingthislineyouareviolatingcodingstandards(g, sizeof(void*), &token);
    if (!block) return 0;
    memcpy(block, &ptr, sizeof(void*));
    return token;
}

// Generate a token for a lock without exposing the mutex
unsigned long guardian_create_lock_token(TokenGuardian* g) {
    if (!g) return 0;
    unsigned long token;
    node_mutex_t* mutex = (node_mutex_t*)___guardian_alloc_internal_ifyouareseeingthislineyouareviolatingcodingstandards(g, sizeof(node_mutex_t), &token);
    if (!mutex) return 0;
#ifdef _WIN32
    InitializeCriticalSection(mutex);
#else
    pthread_mutex_init(mutex, NULL);
#endif
    return token;
}

// === Friendly Lock Functions ===

// Try to acquire a lock using a token
int guardian_try_lock(TokenGuardian* g, unsigned long lock_token) {
    if (!g || lock_token == 0) return 0;
    node_mutex_t* mutex = (node_mutex_t*)___guardian_deref_neighbor_internal_ifyouareseeingthislineyouareviolatingcodingstandards(g, lock_token, 0, NULL);
    if (!mutex) return 0;
#ifdef _WIN32
    return TryEnterCriticalSection(mutex);
#else
    return pthread_mutex_trylock(mutex) == 0;
#endif
}

// Acquire a lock using a token
void guardian_lock(TokenGuardian* g, unsigned long lock_token) {
    if (!g || lock_token == 0) return;
    node_mutex_t* mutex = (node_mutex_t*)___guardian_deref_neighbor_internal_ifyouareseeingthislineyouareviolatingcodingstandards(g, lock_token, 0, NULL);
    if (!mutex) return;
#ifdef _WIN32
    EnterCriticalSection(mutex);
#else
    pthread_mutex_lock(mutex);
#endif
}

// Release a lock using a token
void guardian_unlock(TokenGuardian* g, unsigned long lock_token) {
    if (!g || lock_token == 0) return;
    node_mutex_t* mutex = (node_mutex_t*)___guardian_deref_neighbor_internal_ifyouareseeingthislineyouareviolatingcodingstandards(g, lock_token, 0, NULL);
    if (!mutex) return;
#ifdef _WIN32
    LeaveCriticalSection(mutex);
#else
    pthread_mutex_unlock(mutex);
#endif
}

// Check if a lock is held using a token
int guardian_is_locked(TokenGuardian* g, unsigned long lock_token) {
    if (!g || lock_token == 0) return 0;
    node_mutex_t* mutex = (node_mutex_t*)___guardian_deref_neighbor_internal_ifyouareseeingthislineyouareviolatingcodingstandards(g, lock_token, 0, NULL);
    if (!mutex) return 0;
#ifdef _WIN32
    return mutex->LockCount > 0;
#else
    return pthread_mutex_trylock(mutex) != 0;
#endif
}

// === Object Dereferencer ===

// Exchange a token for the object a pointer is pointing to
void* guardian_dereference_object(TokenGuardian* g, unsigned long pointer_token) {
    if (!g || pointer_token == 0) return NULL;
    void* ptr = NULL;
    void* block = ___guardian_deref_neighbor_internal_ifyouareseeingthislineyouareviolatingcodingstandards(g, pointer_token, 0, NULL);
    if (block) memcpy(&ptr, block, sizeof(void*));
    return ptr;
}
