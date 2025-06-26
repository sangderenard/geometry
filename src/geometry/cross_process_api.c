#include "geometry/cross_process_api.h"
#include <unistd.h>

int cp_send(int fd, const void* data, size_t size) {
    const char* buf = (const char*)data;
    size_t total = 0;
    while (total < size) {
        ssize_t n = write(fd, buf + total, size - total);
        if (n <= 0) return 0;
        total += (size_t)n;
    }
    return 1;
}

int cp_recv(int fd, void* data, size_t size) {
    char* buf = (char*)data;
    size_t total = 0;
    while (total < size) {
        ssize_t n = read(fd, buf + total, size - total);
        if (n <= 0) return 0;
        total += (size_t)n;
    }
    return 1;
}
