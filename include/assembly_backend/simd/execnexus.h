// execution_nexus.h
// this file is a speculative exploration of multi lane work, not yet used, not related to other code
#ifndef EXECUTION_NEXUS_H
#define EXECUTION_NEXUS_H

#include <stddef.h>   // for size_t
#include <stdint.h>   // for error codes, timeouts

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque type for the execution nexus
typedef struct ExecNexus ExecNexus;

/// Overall status and lane-failure flags for a pulse
/// Bits 0-7: status flags; bits 8+ encode failed lane indices
typedef uint64_t ExecNexusStatus;

// Status flag bits (0-7)
#define EXECNEXUS_OK             0ULL            ///< no errors
#define EXECNEXUS_FLAG_TIMEOUT   (1ULL << 0)     ///< one or more lanes timed out
#define EXECNEXUS_FLAG_THREAD    (1ULL << 1)     ///< a lane encountered a fatal error
#define EXECNEXUS_FLAG_PARTIAL   (1ULL << 2)     ///< some lanes succeeded, others failed

// Lane failure mask: set bit (lane_id + 8) to mark lane failures
#define EXECNEXUS_LANE_SHIFT     8
#define EXECNEXUS_LANE_MASK      (~((1ULL << EXECNEXUS_LANE_SHIFT) - 1))
#define EXECNEXUS_SET_LANE_FAIL(status, lane) \
    ((status) |= (1ULL << ((lane) + EXECNEXUS_LANE_SHIFT)))
#define EXECNEXUS_LANE_FAILED(status, lane) \
    (((status) >> (EXECNEXUS_LANE_SHIFT + (lane))) & 1ULL)

/// Per-lane status detail (paired by index after a pulse)
typedef struct {
    size_t          lane_id;      ///< 0-based lane index
    ExecNexusStatus status;       ///< execution result
    int32_t         attempts;     ///< number of retries performed
} ExecLaneStatus;

/// User-provided callback: the "black-box" work each lane executes
/// @param ctx  user context pointer
/// @return     0 on success, non-zero on failure
typedef int (*ExecWorkerFn)(void* ctx);

/// Error handler callback: invoked when a lane fails
/// @param nexus   the nexus instance
/// @param lane_id the lane index that failed
/// @param code    status code for the failure
/// @param ctx     user-provided context
typedef void (*ExecErrorHandlerFn)(ExecNexus* nexus,
                                  size_t lane_id,
                                  ExecNexusStatus code,
                                  void* ctx);

/// Options to configure the nexus behavior at creation
typedef struct {
    uint32_t             timeout_ms;       ///< per-lane timeout (0 = no timeout)
    uint8_t              max_retries;      ///< how many times to retry on failure
    ExecErrorHandlerFn   error_handler;    ///< callback for unrrecoverable errors
    void*                handler_ctx;      ///< passed to error_handler
} ExecNexusOptions;

/**
 * Create an execution nexus with `n_lanes` threads.
 * Each thread will invoke `worker(ctx)` on every clock pulse.
 * @param n_lanes   number of concurrent lanes (threads)
 * @param worker    black-box worker function; return 0 for success
 * @param ctx       user context forwarded to worker
 * @param opts      optional behavior flags (NULL for defaults)
 * @return          pointer to nexus, or NULL on failure
 */
ExecNexus*
execnexus_create(size_t n_lanes,
                 ExecWorkerFn worker,
                 void* ctx,
                 const ExecNexusOptions* opts);

/**
 * Issue one clock pulse: wakes all lanes in lock-step.
 * Blocks until all lanes have returned or timed out.
 * @param nexus        the nexus instance
 * @param out_status   caller-allocated array of length `n_lanes` to receive per-lane status
 * @return             overall ExecNexusStatus (OK, PARTIAL, TIMEOUT, etc.)
 */
ExecNexusStatus
execnexus_pulse(ExecNexus* nexus,
                ExecLaneStatus* out_status);

/**
 * Forcefully kill a specific lane and optionally retry it on the next pulse.
 * @param nexus    the nexus instance
 * @param lane_id  0-based lane index
 * @return         0 on success, non-zero if lane_id invalid
 */
int
execnexus_kill_lane(ExecNexus* nexus,
                     size_t lane_id);

/**
 * Reset a lane's internal retry counter so it will attempt again immediately.
 * @param nexus    the nexus instance
 * @param lane_id  0-based lane index
 * @return         0 on success, non-zero if lane_id invalid
 */
int
execnexus_retry_lane(ExecNexus* nexus,
                      size_t lane_id);

/**
 * Retrieve a simple ASCII trace of recent lane events for debugging.
 * @param nexus     the nexus instance
 * @param lane_id   0-based lane index
 * @param buf       buffer to fill with trace (null-terminated)
 * @param buf_len   size of buf in bytes
 * @return          number of bytes written (excluding null), or -1 on error
 */
int
execnexus_get_trace(ExecNexus* nexus,
                     size_t lane_id,
                     char* buf,
                     size_t buf_len);

/**
 * Destroy the nexus: stops all lanes (after current pulse), waits for exit, and frees resources.
 */
void
execnexus_destroy(ExecNexus* nexus);

#ifdef __cplusplus
}
#endif

#endif // EXECUTION_NEXUS_H
