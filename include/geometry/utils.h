#ifndef GEOMETRY_UTILS_H
#define GEOMETRY_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

#import "types.h"

Emergence* emergence_create(void);
void emergence_destroy(Emergence* e);
void emergence_lock(Emergence* e);
void emergence_release(Emergence* e);
void emergence_resolve(Emergence* e);
void emergence_update(Emergence* e, Node* node, double activation, uint64_t global_step, uint64_t timestamp);


/* GuardianLinkNode definition provided by guardian_link_cache.h */
GuardianToken * guardian_create_pointer_token(TokenGuardian* g, void* ptr, NodeFeatureType type);


// Add missing function prototypes
TokenGuardian* find_token_authority(TokenGuardian* g);
TokenGuardian* guardian_initialize(TokenGuardian* parent, size_t num_threads);
GuardianToken* guardian_create_pointer_token(TokenGuardian* g, void* ptr, NodeFeatureType type);
GuardianToken* guardian_create_lock_token(TokenGuardian* g);
boolean guardian_lock_with_timeout(TokenGuardian* g, GuardianToken guardian_lock_token, int duration, boolean reentrant);
void* guardian_dereference_object(TokenGuardian* g, unsigned long pointer_token);

TokenGuardian* ___guardian_create_internal_(void);
void ___guardian_destroy_internal_(TokenGuardian* g);
unsigned long ___guardian_register_thread_internal_(TokenGuardian* g);
void ___guardian_unregister_thread_internal_(TokenGuardian* g, unsigned long token);
void* ___guardian_alloc_internal_(TokenGuardian* g, size_t size, unsigned long* token_out);
void ___guardian_free_internal_(TokenGuardian* g, unsigned long token);
void ___guardian_send_internal_(TokenGuardian* g, unsigned long from, unsigned long to, const void* data, size_t size);
size_t ___guardian_receive_internal_(TokenGuardian* g, unsigned long to, void* buffer, size_t max_size);
unsigned long ___guardian_create_object_internal_(TokenGuardian* g, int type, int count, GuardianPointerToken referrent, GuardianList params);
void ___guardian_destroy_object_internal_(TokenGuardian* g, unsigned long token);
unsigned long ___guardian_parse_nested_object_internal_(TokenGuardian* g, const char* request);

/* Linked list helpers */
GuardianLinkedList* guardian_linked_list_set_chain(GuardianLinkedList* list,
                                                   GuardianLinkNode** chain,
                                                   int type,
                                                   int chain_length);
GuardianLinkedList* guardian_create_linked_list(TokenGuardian* g,
                                                int initialized_length,
                                                int type,
                                                void** data);
GuardianList* guardian_create_list(TokenGuardian* g,
                                   int initialized_length,
                                   int type,
                                   void** data);

#ifdef __cplusplus
}
#endif

#endif // GEOMETRY_UTILS_H
