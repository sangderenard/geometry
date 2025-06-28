#ifndef GUARDIAN_LINK_CACHE_H
#define GUARDIAN_LINK_CACHE_H

#include "geometry/guardian_platform_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// The fundamental building block of all linked data in the Guardian system.
// A raw, untyped, bidirectional link, exactly as you described.
typedef struct GuardianLinkNode {
    struct GuardianLinkNode* prev;
    struct GuardianLinkNode* next;
    void* payload; // Points to the actual data
} GuardianLinkNode;

// Initializes the global link cache.
// Must be called once at application startup. Not thread-safe.
void guardian_link_cache_init(size_t initial_capacity);

// Shuts down and frees the global link cache. Not thread-safe.
void guardian_link_cache_shutdown();

// Allocates a single link node from the global cache. Thread-safe.
GuardianLinkNode* guardian_link_alloc();

// Frees a single link node, returning it to the global cache. Thread-safe.
void guardian_link_free(GuardianLinkNode* node);

#ifdef __cplusplus
}
#endif

#endif // GUARDIAN_LINK_CACHE_H