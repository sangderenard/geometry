#ifndef ASSEMBLY_BACKEND_CORE_H
#define ASSEMBLY_BACKEND_CORE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void assembly_backend_core_init(void);

void* mg_alloc(size_t size);
void mg_free(void* ptr);

#ifdef __cplusplus
}
#endif

#endif /* ASSEMBLY_BACKEND_CORE_H */
