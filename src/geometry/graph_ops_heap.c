#include "geometry/graph_ops_heap.h"
#include "geometry/guardian_platform.h"
// Stub implementations for heap contiguous operations
static boolean heap_make_contiguous(void* container) {
    // TODO: lock, rebuild memory using memory_ops, optimize allocation
    return true;
}
static boolean heap_make_contiguous_no_wait(void* container) {
    return heap_make_contiguous(container);
}
static boolean heap_make_contiguous_wait(void* container) {
    return heap_make_contiguous(container);
}
static boolean heap_make_contiguous_wait_timeout(void* container) {
    return heap_make_contiguous(container);
}
static boolean heap_make_contiguous_force(void* container) {
    return heap_make_contiguous(container);
}

// Populate only the math_ops sub-struct; other ops remain zero
const OperationSuite graph_ops_heap = {
    .math_ops = {
        .make_contiguous              = heap_make_contiguous,
        .make_contiguous_no_wait      = heap_make_contiguous_no_wait,
        .make_contiguous_wait         = heap_make_contiguous_wait,
        .make_contiguous_wait_timeout = heap_make_contiguous_wait_timeout,
        .make_contiguous_force        = heap_make_contiguous_force
    }
};
