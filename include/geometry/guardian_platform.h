#ifndef GUARDIAN_PLATFORM_H
#define GUARDIAN_PLATFORM_H

// ========================================
// 🛠️  Platform Detection
// ========================================
#include "geometry/guardian_platform_types.h"

// ========================================
// 🔗 Standard Includes
// ========================================
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

// ========================================
// 🧵 Mutex Interface (Header-Level Inline)
// ========================================
static inline void guardian_mutex_lock(mutex_t* mtx) {
#if GUARDIAN_PLATFORM_WINDOWS
  WaitForSingleObject(*mtx, INFINITE);
#else
  pthread_mutex_lock(*mtx);
#endif
}

static inline void guardian_mutex_unlock(mutex_t* mtx) {
#if GUARDIAN_PLATFORM_WINDOWS
  ReleaseMutex(*mtx);
#else
  pthread_mutex_unlock(*mtx);
#endif
}

static inline int guardian_mutex_trylock(mutex_t* mtx) {
#if GUARDIAN_PLATFORM_WINDOWS
  // Returns 1 on success, 0 on failure (timeout).
  return WaitForSingleObject(*mtx, 0) == WAIT_OBJECT_0;
#else
  // Returns 1 on success, 0 on failure.
  return pthread_mutex_trylock(*mtx) == 0;
#endif
}

// Full mutex lifecycle functions declared
mutex_t* guardian_mutex_init();
void guardian_mutex_destroy(mutex_t* mtx);

// ========================================
// ⏱️ Time Utilities (Prototype)
// ========================================
guardian_time_t guardian_now();
double guardian_seconds_since(guardian_time_t start);
void guardian_sleep_ms(unsigned int ms);

// ========================================
// 📁 File System
// ========================================
FILE* guardian_fopen(const char* path, const char* mode);
int guardian_file_exists(const char* path);
int guardian_mkdir(const char* path);

// ========================================
// 🔌 Dynamic Libraries
// ========================================
#if GUARDIAN_PLATFORM_WINDOWS
  #define guardian_lib_handle HMODULE
  #define guardian_load_library(path) LoadLibraryA(path)
  #define guardian_get_proc(handle, name) GetProcAddress(handle, name)
  #define guardian_close_library(handle) FreeLibrary(handle)
#else
  #define guardian_lib_handle void*
  #define guardian_load_library(path) dlopen(path, RTLD_LAZY)
  #define guardian_get_proc(handle, name) dlsym(handle, name)
  #define guardian_close_library(handle) dlclose(handle)
#endif

// ========================================
// 🧠 System Info
// ========================================
size_t guardian_get_cpu_count();
size_t guardian_get_page_size();

// ========================================
// 🧵 Threads
// ========================================
int guardian_thread_create(guardian_thread_handle_t* handle, void* (*start_routine)(void*), void* arg);
void guardian_thread_exit();
int guardian_thread_join(guardian_thread_handle_t handle);
int guardian_get_current_thread_id();
void guardian_yield();

#endif // GUARDIAN_PLATFORM_H
