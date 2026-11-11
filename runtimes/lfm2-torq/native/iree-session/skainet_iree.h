/*
 * skainet_iree: a small C API over the IREE runtime (Torq fork) for Kotlin/Native cinterop. Derived from
 * SKaiNET-embedded's IREE shim (MIT; ../upstream.json).
 *
 * Self-contained on purpose: only opaque handles and plain C types, so cinterop never parses IREE's headers.
 * One process holds one skainet_iree_runtime (one IREE instance); devices ("local-task", "torq", "vulkan") are created on
 * demand and shared by every session on them.
 *
 * Errors: functions return NULL / a negative value and set a thread-local message, read with skainet_iree_last_error().
 */
#ifndef SKAINET_IREE_H
#define SKAINET_IREE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct skainet_iree_runtime skainet_iree_runtime;
typedef struct skainet_iree_session skainet_iree_session;
typedef struct skainet_iree_view skainet_iree_view;

/* Message of the last failure on this thread ("" when none). Valid until the next call on this thread. */
const char* skainet_iree_last_error(void);

/* ---------- runtime ---------- */

/*
 * Creates the IREE instance with every driver compiled in. `flags` are IREE runtime flags parsed once per process,
 * e.g. "--torq_hw_type=astra_machina" or "--task_topology_max_group_count=2" (may be NULL / 0).
 */
skainet_iree_runtime* skainet_iree_runtime_create(const char* const* flags, int flag_count);
void skainet_iree_runtime_release(skainet_iree_runtime* rt);
/* Space-separated names of the HAL drivers compiled in (e.g. "local-sync local-task vulkan torq"). */
const char* skainet_iree_runtime_drivers(skainet_iree_runtime* rt);

/* ---------- generic sessions (one vmfb, optional parameter archive) ---------- */

/*
 * Opens `vmfb` on `device`. When `irpa` is not NULL its parameters are served under `scope` ("model" when NULL).
 */
skainet_iree_session* skainet_iree_session_create(skainet_iree_runtime* rt, const char* device, const char* vmfb, const char* irpa, const char* scope);
void skainet_iree_session_release(skainet_iree_session* s);

/* Element types of skainet_iree_view (values of IREE's iree_hal_element_type_t for these types). */
enum {
  SKAINET_IREE_F32 = 0x21000020,
  SKAINET_IREE_F16 = 0x21000010,
  SKAINET_IREE_BF16 = 0x22000010,
  SKAINET_IREE_I32 = 0x10000020,
  SKAINET_IREE_I64 = 0x10000040,
  SKAINET_IREE_I8 = 0x10000008,
};

/* Copies host data to a new device buffer of `session`'s device. `bytes` must equal the shape's dense size. */
skainet_iree_view* skainet_iree_view_from_host(skainet_iree_session* s, int32_t element_type, int rank, const int64_t* dims, const void* data, size_t bytes);
/* Copies a view back to host memory; `bytes` must equal skainet_iree_view_bytes(). Returns 0 or -1. */
int skainet_iree_view_to_host(skainet_iree_session* s, skainet_iree_view* v, void* dst, size_t bytes);
int skainet_iree_view_rank(skainet_iree_view* v);
int64_t skainet_iree_view_dim(skainet_iree_view* v, int axis);
int32_t skainet_iree_view_element_type(skainet_iree_view* v);
size_t skainet_iree_view_bytes(skainet_iree_view* v);
void skainet_iree_view_retain(skainet_iree_view* v);
void skainet_iree_view_release(skainet_iree_view* v);

/*
 * Invokes `function` ("module.main"). Inputs are borrowed; on success `outputs[0..output_count)` receive new views
 * the caller releases. Returns the number of outputs the function produced, or -1. Fails when `output_capacity`
 * is too small.
 */
int skainet_iree_call(skainet_iree_session* s, const char* function, skainet_iree_view* const* inputs, int input_count,
             skainet_iree_view** outputs, int output_capacity);

#ifdef __cplusplus
}
#endif

#endif
