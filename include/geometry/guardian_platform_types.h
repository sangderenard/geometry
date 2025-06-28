#ifndef GUARDIAN_PLATFORM_TYPES_H
#define GUARDIAN_PLATFORM_TYPES_H

#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
  #define GUARDIAN_PLATFORM_WINDOWS 1
#else
  #define GUARDIAN_PLATFORM_POSIX 1
#endif

#include "geometry/types.h"

#if GUARDIAN_PLATFORM_WINDOWS
  #include <windows.h>
  #include <io.h>
  #include <direct.h>
  #define PATH_SEPARATOR '\\'
  typedef HANDLE mutex_t;
  typedef HANDLE guardian_thread_handle_t;
  #define guardian_lib_handle HMODULE
#else
  #include <pthread.h>
  #include <unistd.h>
  #include <dlfcn.h>
  #include <sys/time.h>
  #include <sys/stat.h>
  #define PATH_SEPARATOR '/'
  typedef pthread_mutex_t* mutex_t;
  typedef pthread_t guardian_thread_handle_t;
  #define guardian_lib_handle void*
#endif

#endif /* GUARDIAN_PLATFORM_TYPES_H */
