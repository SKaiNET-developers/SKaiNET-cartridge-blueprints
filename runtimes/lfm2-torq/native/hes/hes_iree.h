/*
 * hes_iree: a small C API over the IREE runtime (Torq fork) for Kotlin/Native cinterop.
 *
 * Self-contained on purpose: only opaque handles and plain C types, so cinterop never parses IREE's headers.
 * One process holds one hes_runtime (one IREE instance); devices ("local-task", "torq", "vulkan") are created on
 * demand and shared by every session on them.
 *
 * Errors: functions return NULL / a negative value and set a thread-local message, read with hes_last_error().
 */
#ifndef HES_IREE_H
#define HES_IREE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hes_runtime hes_runtime;
typedef struct hes_session hes_session;
typedef struct hes_view hes_view;
typedef struct hes_kv hes_kv;
typedef struct hes_kv_snapshot hes_kv_snapshot;

/* Message of the last failure on this thread ("" when none). Valid until the next call on this thread. */
const char* hes_last_error(void);

/* ---------- runtime ---------- */

/*
 * Creates the IREE instance with every driver compiled in. `flags` are IREE runtime flags parsed once per process,
 * e.g. "--torq_hw_type=astra_machina" or "--task_topology_max_group_count=2" (may be NULL / 0).
 */
hes_runtime* hes_runtime_create(const char* const* flags, int flag_count);
void hes_runtime_release(hes_runtime* rt);
/* Space-separated names of the HAL drivers compiled in (e.g. "local-sync local-task vulkan torq"). */
const char* hes_runtime_drivers(hes_runtime* rt);

/* ---------- generic sessions (one vmfb, optional parameter archive) ---------- */

/*
 * Opens `vmfb` on `device`. When `irpa` is not NULL its parameters are served under `scope` ("model" when NULL).
 */
hes_session* hes_session_create(hes_runtime* rt, const char* device, const char* vmfb, const char* irpa, const char* scope);
void hes_session_release(hes_session* s);

/* Element types of hes_view (values of IREE's iree_hal_element_type_t for these types). */
enum {
  HES_F32 = 0x21000020,
  HES_F16 = 0x21000010,
  HES_BF16 = 0x22000010,
  HES_I32 = 0x10000020,
  HES_I64 = 0x10000040,
  HES_I8 = 0x10000008,
};

/* Copies host data to a new device buffer of `session`'s device. `bytes` must equal the shape's dense size. */
hes_view* hes_view_from_host(hes_session* s, int32_t element_type, int rank, const int64_t* dims, const void* data, size_t bytes);
/* Copies a view back to host memory; `bytes` must equal hes_view_bytes(). Returns 0 or -1. */
int hes_view_to_host(hes_session* s, hes_view* v, void* dst, size_t bytes);
int hes_view_rank(hes_view* v);
int64_t hes_view_dim(hes_view* v, int axis);
int32_t hes_view_element_type(hes_view* v);
size_t hes_view_bytes(hes_view* v);
void hes_view_retain(hes_view* v);
void hes_view_release(hes_view* v);

/*
 * Invokes `function` ("module.main"). Inputs are borrowed; on success `outputs[0..output_count)` receive new views
 * the caller releases. Returns the number of outputs the function produced, or -1. Fails when `output_capacity`
 * is too small.
 */
int hes_call(hes_session* s, const char* function, hes_view* const* inputs, int input_count,
             hes_view** outputs, int output_capacity);

/* ---------- KV-cache decoder (functiongemma-kv-v1 / qwen-kv-v1 host-gather contracts) ---------- */

typedef struct {
  int32_t n_layers, head_dim, n_kv_heads, n_heads, hidden_size, vocab_size;
  int32_t sliding_window, global_layer_period, chunk, mask_heads;  /* mask_heads: 0 = n_heads, or 1 */
  float sliding_rope_base, global_rope_base;
} hes_kv_spec;

/*
 * Three graphs (prefill once, chunk per utterance, step per token), each its own vmfb + irpa. Pass the same paths
 * for all three to share one session (merged-module contract). The prefill pair may be NULL. Function names
 * default to module.gemma_with_past / module.gemma_prefill_with_past / module.gemma_prefill_at when NULL.
 */
hes_kv* hes_kv_create(hes_runtime* rt, const char* device, const hes_kv_spec* spec,
                      const char* vmfb_with_past, const char* irpa_with_past, const char* fn_with_past,
                      const char* vmfb_chunk, const char* irpa_chunk, const char* fn_chunk,
                      const char* vmfb_prefill, const char* irpa_prefill, const char* fn_prefill);
void hes_kv_release(hes_kv* k);
/* Prefill `n` real tokens of `tokens[0..seq)` (zero-padded to the graph's sequence). Returns the next token or -1. */
int32_t hes_kv_prefill(hes_kv* k, const int32_t* tokens, int32_t seq, int32_t n);
/* Frees the prefill graph's session once the prefix snapshot exists. */
void hes_kv_release_prefill(hes_kv* k);
/* Runs `n` <= chunk tokens against the cache in one call. Returns the token after the last real one, or -1. */
int32_t hes_kv_chunk(hes_kv* k, const int32_t* tokens, int32_t n);
/* Appends `token`, returns the next one, or -1. */
int32_t hes_kv_step(hes_kv* k, int32_t token);
/* Absolute position of the next token. */
int32_t hes_kv_position(hes_kv* k);
/* Retains the current cache (zero-copy). */
hes_kv_snapshot* hes_kv_snapshot_take(hes_kv* k);
int hes_kv_restore(hes_kv* k, hes_kv_snapshot* s);
void hes_kv_snapshot_release(hes_kv_snapshot* s);

/* ---------- Moonshine streaming ASR (moonshine-v2-streaming-v1 contract) ---------- */

typedef struct hes_ms hes_ms;

/* Zero fields take the default (Moonshine v2 tiny / tiny-de). */
typedef struct {
  int32_t n_layers;       /* decoder layers: 6 */
  int32_t head_dim;       /* 40 */
  int32_t rope_pairs;     /* rotary pairs per head: 16 */
  int32_t dim;            /* decoder width: 320 (dec_embed.bin must hold at least vocab rows of it) */
  int32_t enc_dim;        /* frontend/encoder width; 0 = dim */
  int32_t lag_budget_ms;  /* drop partial decodes once this far behind real time; 0 = 250, negative = never */
  int32_t fast_finish;    /* 1 = keep the incremental decode as the final instead of re-decoding exactly */
  int32_t skip_warmup;    /* 1 = no zero-input pass through every graph at create */
  int32_t verbose;        /* 1 = per-window / per-hop timing lines on stderr */
  int32_t no_peek;        /* 1 = no early partial at ~0.72 s (saves one window + one decode per utterance) */
  int32_t partial_every;  /* decode partials on every Nth hop only (0/1 = every hop); the final is unaffected */
} hes_ms_spec;

/*
 * Five graphs, each on its stage's device (frontend, encoder, adapter; prefill and step share the decoder device),
 * the decoder parameter archive (NULL when the decoder graphs carry their weights inline, e.g. int8), the vocabulary
 * table and the decoder embedding rows. Runs one warm-up pass unless
 * the spec says otherwise. One utterance at a time; not thread-safe.
 */
hes_ms* hes_ms_create(hes_runtime* rt, const hes_ms_spec* spec,
                      const char* dev_frontend, const char* dev_encoder, const char* dev_adapter, const char* dev_decoder,
                      const char* vmfb_frontend, const char* vmfb_encoder, const char* vmfb_adapter,
                      const char* vmfb_prefill, const char* vmfb_step,
                      const char* irpa, const char* vocab_bin, const char* dec_embed_bin);
void hes_ms_release(hes_ms* e);
/* Feeds 16 kHz mono PCM in [-1, 1]. Returns the new cumulative partial, or NULL when it did not change. The string
 * is valid until the next call on `e`. */
const char* hes_ms_feed(hes_ms* e, const float* pcm, int32_t n);
/* End of utterance: the exact final ("" when nothing was recognized), then resets for the next utterance. NULL when
 * a stage failed and nothing was recognized (see hes_last_error()). Valid until the next call on `e`. */
const char* hes_ms_finish(hes_ms* e);
/* Aborts the current utterance. */
void hes_ms_reset(hes_ms* e);
/* The last finish()'s counters as comma-separated key=value pairs ("" before the first). */
const char* hes_ms_stats(hes_ms* e);
int32_t hes_ms_dim(hes_ms* e);
int32_t hes_ms_vocab(hes_ms* e);

#ifdef __cplusplus
}
#endif
#endif
