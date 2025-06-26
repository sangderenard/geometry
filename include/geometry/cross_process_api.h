#ifndef GEOMETRY_CROSS_PROCESS_API_H
#define GEOMETRY_CROSS_PROCESS_API_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int cp_send(int fd, const void* data, size_t size);
int cp_recv(int fd, void* data, size_t size);

#ifdef __cplusplus
}
#endif

#endif // GEOMETRY_CROSS_PROCESS_API_H
