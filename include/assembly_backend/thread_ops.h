#ifndef ASSEMBLY_BACKEND_THREAD_OPS_H
#define ASSEMBLY_BACKEND_THREAD_OPS_H

#include <stddef.h>
#include "geometry/guardian_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GuardianThread GuardianThread;
typedef struct TokenGuardian TokenGuardian;

void thread_ops_init(size_t capacity);
void thread_ops_register_mutex(long long token, mutex_t* mutex);
mutex_t* thread_ops_get_mutex(long long token);
void thread_ops_register_thread(long long token, guardian_thread_handle_t thread);
guardian_thread_handle_t thread_ops_get_thread(long long token);
void thread_ops_register_pointer(long long token, void* ptr);
void* thread_ops_get_pointer(long long token);

int try_lock(TokenGuardian* g, unsigned long lock_token);
void lock(TokenGuardian* g, unsigned long lock_token);
void unlock(TokenGuardian* g, unsigned long lock_token);
boolean unlock_timeout(TokenGuardian* g, unsigned long lock_token, int duration);
int is_locked(TokenGuardian* g, unsigned long lock_token);

#ifdef __cplusplus
}
#endif

#endif /* ASSEMBLY_BACKEND_THREAD_OPS_H */
