/// differentiator.h
/// High-performance multi-axis differentiation & integration engine
/// with DEC, metric tensors, autograd, and streaming support.

#ifndef DIFFERENTIATOR_H
#define DIFFERENTIATOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include "utils.h"    // boolean, sizing, platform
#include "guardian_platform.h"  // threading, mutex

// ========================================
// Error Codes
// ========================================
typedef enum {
    DIFF_OK = 0,
    DIFF_ERR_NULL_POINTER,
    DIFF_ERR_INVALID_SIZE,
    DIFF_ERR_BAD_FLAGS,
    DIFF_ERR_NOT_INITIALIZED,
    DIFF_ERR_ALLOCATION_FAILED,
    DIFF_ERR_GRAPH_FAILURE,
    DIFF_ERR_DEC_FAILURE,
    DIFF_ERR_METRIC_FAILURE,
    DIFF_ERR_INTEGRATION_FAILURE
} diff_error_t;

// ========================================
// Flags (32-bit)
// ========================================
// Diff modes
#define DIFF_FLAG_BYTEWISE        0x00000001u
#define DIFF_FLAG_TYPEWISE        0x00000002u
// Axis selection
#define DIFF_FLAG_AXIS_X          0x00000010u
#define DIFF_FLAG_AXIS_Y          0x00000020u
#define DIFF_FLAG_AXIS_Z          0x00000040u
#define DIFF_FLAG_AXIS_T          0x00000080u
#define DIFF_FLAG_MULTI_AXIS      (DIFF_FLAG_AXIS_X | DIFF_FLAG_AXIS_Y | \
                                 DIFF_FLAG_AXIS_Z | DIFF_FLAG_AXIS_T)
// ND & Tensor
#define DIFF_FLAG_METRIC_TENSOR   0x00000100u
#define DIFF_FLAG_USE_JACOBIAN    0x00000200u
#define DIFF_FLAG_ND_ENGINE       (DIFF_FLAG_METRIC_TENSOR | DIFF_FLAG_USE_JACOBIAN)
#define DIFF_FLAG_DEC_ENABLED     0x00000400u
// Graph / Set theory
#define DIFF_FLAG_GRAPH_MODE      0x00000800u
#define DIFF_FLAG_SET_THEORY      0x00001000u
// Automatic differentiation
#define DIFF_FLAG_AUTOGRAD        0x00002000u
#define DIFF_FLAG_BACKPROP_ENABLED 0x00004000u
// Streaming
#define DIFF_FLAG_STREAM_MODE     0x00008000u
// Kernel support
#define DIFF_FLAG_KERNEL_SUPPORT  0x00010000u
// Encoding scheme
#define DIFF_FLAG_ENCODING_MASK   0x00700000u
#define DIFF_FLAG_ENCODING_GRAY     0x00000000u
#define DIFF_FLAG_ENCODING_HAMMING  0x00100000u
#define DIFF_FLAG_ENCODING_TERNARY  0x00200000u
#define DIFF_FLAG_ENCODING_UNARY    0x00300000u
#define DIFF_FLAG_ENCODING_CUSTOM   0x00400000u
// Combined presets
#define DIFF_FLAG_DEFAULT           (DIFF_FLAG_BYTEWISE | DIFF_FLAG_MULTI_AXIS)
#define DIFF_FLAG_DEFAULT_ND       (DIFF_FLAG_ND_ENGINE)
#define DIFF_FLAG_FULL_FEATURE     (DIFF_FLAG_DEFAULT | DIFF_FLAG_DEFAULT_ND | \
                                   DIFF_FLAG_DEC_ENABLED | DIFF_FLAG_AUTOGRAD | \
                                   DIFF_FLAG_BACKPROP_ENABLED | DIFF_FLAG_STREAM_MODE | \
                                   DIFF_FLAG_KERNEL_SUPPORT)

// ========================================
// diff_result structures
// ========================================
typedef struct {
    size_t index;
    uint8_t old_val;
    uint8_t new_val;
    boolean changed;
} diff_result_byte_t;

typedef struct {
    size_t index;
    void const* old_val;
    void const* new_val;
    size_t byte_size;
    boolean changed;
} diff_result_type_t;

// ========================================
// Integration hooks
// ========================================
typedef struct {
    void (*on_diff)(void* user, size_t idx, void const* oldv, void const* newv, size_t bytes);
    void (*on_integrate)(void* user, size_t idx, void* outv, void const* delta, size_t bytes);
    void* user_data;
} integration_hook_t;

// ========================================
// Metric tensor
// ========================================
typedef struct {
    float* data;      // [n x n] row-major
    size_t dim;
} metric_tensor_t;

// ========================================
// DEC operators
// ========================================
typedef struct {
    diff_error_t (*d0)(void* out, void const* in, size_t bytes);
    diff_error_t (*d1)(void* out, void const* in, size_t bytes);
    diff_error_t (*d2)(void* out, void const* in, size_t bytes);
} dec_ops_t;

// ========================================
// Autograd graph
// ========================================
struct diff_node;
typedef float (*grad_fn_t)(float x, void* ctx);

typedef struct diff_node {
    struct diff_node** inputs;
    size_t input_count;
    grad_fn_t grad_fn;
    void* context;
    float value;
    float grad;
    boolean requires_grad;
} diff_node_t;

typedef struct {
    diff_node_t** nodes;
    size_t count, capacity;
    boolean track;
    mutex_t lock;
} diff_graph_t;

// ========================================
// Differentiator handle
// ========================================
typedef struct {
    uint8_t* curr;
    uint8_t* prev;
    size_t size;
    uint32_t flags;
    boolean initialized;
    integration_hook_t hooks;
    boolean stream;
    metric_tensor_t* metric;
    dec_ops_t dec;
    diff_graph_t* graph;
} differentiator_t;

// ========================================
// API
// ========================================

// Initialize differentiator; must call before use
diff_error_t differentiator_init(differentiator_t* d, size_t size, uint32_t flags, boolean enable_stream);

// Configure memory buffers (const correctness)
diff_error_t differentiator_set_buffers(differentiator_t* d, void* current, const void* previous);

// Install hooks, metric, DEC operators, graph
void differentiator_set_hooks(differentiator_t* d, integration_hook_t const* hooks);
void differentiator_attach_metric(differentiator_t* d, metric_tensor_t* metric);
void differentiator_set_dec_ops(differentiator_t* d, dec_ops_t const* ops);
void differentiator_set_graph(differentiator_t* d, diff_graph_t* graph);

// Perform diffing and integration
// Results are sent via hooks or collected in buffers
diff_error_t differentiator_diff(differentiator_t* d);
diff_error_t differentiator_integrate(differentiator_t* d);

// Autograd forward/backward
diff_error_t diff_graph_forward(diff_graph_t* g);
diff_error_t diff_graph_backward(diff_graph_t* g);

// Reset state
void differentiator_reset(differentiator_t* d);

#ifdef __cplusplus
}
#endif

#endif // DIFFERENTIATOR_H
