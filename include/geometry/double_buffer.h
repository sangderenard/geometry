#ifndef GEOMETRY_DOUBLE_BUFFER_H
#define GEOMETRY_DOUBLE_BUFFER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

typedef struct {
    void* buffers[2];
    size_t active;
} DoubleBuffer;

static inline void double_buffer_init(DoubleBuffer* db, void* buf_a, void* buf_b) {
    if (!db) return;
    db->buffers[0] = buf_a;
    db->buffers[1] = buf_b;
    db->active = 0;
}

static inline void* double_buffer_read(DoubleBuffer* db) {
    return db ? db->buffers[db->active] : NULL;
}

static inline void* double_buffer_write(DoubleBuffer* db) {
    return db ? db->buffers[1 - db->active] : NULL;
}

static inline void double_buffer_swap(DoubleBuffer* db) {
    if (db) db->active = 1 - db->active;
}

#ifdef __cplusplus
}
#endif

#endif /* GEOMETRY_DOUBLE_BUFFER_H */
