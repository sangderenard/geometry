#ifndef ASSEMBLY_BACKEND_THREAD_OPS_H
#define ASSEMBLY_BACKEND_THREAD_OPS_H

#include <stddef.h>
#include "geometry/guardian_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

void thread_ops_init(size_t capacity);
void thread_ops_register_mutex(long long token, mutex_t* mutex);
mutex_t* thread_ops_get_mutex(long long token);
void thread_ops_register_thread(long long token, guardian_thread_handle_t thread);
guardian_thread_handle_t thread_ops_get_thread(long long token);
void thread_ops_register_pointer(long long token, void* ptr);
void* thread_ops_get_pointer(long long token);

int guardian_try_lock(TokenGuardian* g, unsigned long lock_token);
void guardian_lock(TokenGuardian* g, unsigned long lock_token);
void guardian_unlock(TokenGuardian* g, unsigned long lock_token);
int guardian_is_locked(TokenGuardian* g, unsigned long lock_token);

#ifdef __cplusplus
}
#endif

#endif /* ASSEMBLY_BACKEND_THREAD_OPS_H */
