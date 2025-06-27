#include "geometry/guardian_link_cache.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// --- Global Cache Internals ---
typedef struct {
    GuardianLinkNode* pool;      // The contiguous block of memory for all nodes
    GuardianLinkNode* free_list; // A singly-linked list of available nodes (using the 'next' pointer)
    size_t capacity;
    mutex_t* lock;
    int is_initialized;
} GlobalLinkCache;

static GlobalLinkCache g_cache = {0};

// --- API Implementation ---

void guardian_link_cache_init(size_t initial_capacity) {
    if (g_cache.is_initialized) {
        return; // Already initialized
    }

    if (initial_capacity == 0) {
        initial_capacity = 65536; // Default to a reasonable number
    }

    g_cache.lock = guardian_mutex_init();
    if (!g_cache.lock) {
        // Critical error: cannot create mutex
        perror("Failed to initialize link cache mutex");
        return;
    }

    g_cache.pool = (GuardianLinkNode*)malloc(initial_capacity * sizeof(GuardianLinkNode));
    if (!g_cache.pool) {
        perror("Failed to allocate memory for link cache pool");
        guardian_mutex_destroy(g_cache.lock);
        g_cache.lock = NULL;
        return;
    }

    g_cache.capacity = initial_capacity;

    // Chain all nodes into the free list
    for (size_t i = 0; i < initial_capacity - 1; ++i) {
        g_cache.pool[i].next = &g_cache.pool[i + 1];
    }
    g_cache.pool[initial_capacity - 1].next = NULL;

    g_cache.free_list = &g_cache.pool[0];
    g_cache.is_initialized = 1;
}

void guardian_link_cache_shutdown() {
    if (!g_cache.is_initialized) {
        return;
    }
    free(g_cache.pool);
    guardian_mutex_destroy(g_cache.lock);
    memset(&g_cache, 0, sizeof(GlobalLinkCache));
}

GuardianLinkNode* guardian_link_alloc() {
    if (!g_cache.is_initialized) {
        guardian_link_cache_init(0); // Lazy initialization with default size
        if (!g_cache.is_initialized) return NULL;
    }

    guardian_mutex_lock(g_cache.lock);

    GuardianLinkNode* node = g_cache.free_list;
    if (node) {
        g_cache.free_list = node->next;
    }
    // NOTE: A more robust implementation could expand the pool if it's exhausted.

    guardian_mutex_unlock(g_cache.lock);

    if (node) {
        memset(node, 0, sizeof(GuardianLinkNode)); // Zero out the node before returning
    }
    return node;
}

void guardian_link_free(GuardianLinkNode* node) {
    if (!node || !g_cache.is_initialized) return;

    guardian_mutex_lock(g_cache.lock);
    node->next = g_cache.free_list;
    g_cache.free_list = node;
    guardian_mutex_unlock(g_cache.lock);
}