/*
 * Cartridge Runtime ABI — the call-level contract every cartridge exports.
 *
 * ABI version 0.1 (CTG_ABI_MAJOR.CTG_ABI_MINOR). Spec: docs/modules/ROOT/pages/abi.adoc
 * (this header is normative where the two disagree).
 *
 * Design rules (abi.adoc "Design rules"):
 *   - C ABI as the lingua franca; language facades wrap it, never replace it.
 *   - Pull-based streaming; the cartridge never calls the host.
 *   - Caller owns what it passes in (borrowed per call); the cartridge owns what it
 *     returns; ctg_result is the only handed-over object and is released with
 *     ctg_result_release().
 *   - One session, one thread; ctg_cancel() is the only cross-thread call.
 *   - Every failure is a ctg_status; no "success plus side-channel error flag".
 *   - Structs begin with a caller-set `size` so minor versions can append fields.
 *
 * The header is deliberately FFI-clean: fixed-width types, no function-like macros in
 * the API, no C++ types — consumable via Rust bindgen, Kotlin/Native cinterop, Python
 * cffi, cgo, JNI/Panama alike.
 */
#ifndef CARTRIDGE_ABI_H
#define CARTRIDGE_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CTG_ABI_MAJOR 0
#define CTG_ABI_MINOR 1

/* ------------------------------------------------------------------ status -- */

typedef enum ctg_status {
  CTG_OK = 0,
  CTG_E_INVALID_ARG   = -1,  /* NULL handle, malformed options, bad buffer          */
  CTG_E_UNSUPPORTED   = -2,  /* wrong io family for this cartridge (batch/streaming) */
  CTG_E_PREFLIGHT     = -3,  /* a runtime requirement is unmet (see report/last_error) */
  CTG_E_LOAD          = -4,  /* model/runtime artifact failed to load                */
  CTG_E_BACKEND       = -5,  /* inference backend failed mid-call                    */
  CTG_E_CANCELLED     = -6,  /* call aborted by ctg_cancel()                         */
  CTG_E_STATE         = -7,  /* call not valid in this session state (e.g. push after flush) */
  CTG_E_ABI_MISMATCH  = -8,  /* struct size / version negotiation failed             */
  CTG_E_INTERNAL      = -99
} ctg_status;

/* --------------------------------------------------------------- identity -- */

/* (CTG_ABI_MAJOR << 16) | CTG_ABI_MINOR. A host MUST refuse a cartridge whose
 * major differs from the one it was built against. */
uint32_t ctg_abi_version(void);

/* The cartridge's capability descriptor as embedded, canonical UTF-8 JSON
 * (cartridge-descriptor.schema.json). Static storage; never freed by the caller.
 * This is the same descriptor the manifest digests — a host MAY compare. */
const char* ctg_descriptor_json(void);

/* -------------------------------------------------------------- preflight -- */

typedef struct ctg_preflight_report {
  size_t   size;              /* caller sets sizeof(ctg_preflight_report)           */
  int32_t  ok;                /* 1 = all runtime requirements met                   */
  /* On failure: a NUL-terminated summary in cartridge-owned static storage,
   * stable until the next ctg_preflight call in this process. */
  const char* detail;
} ctg_preflight_report;

/* Check the requirements only a device can answer (requirements.driver presence and
 * version, accelerator presence, memory) WITHOUT loading models. Hosts MUST call this
 * before the first ctg_open on a device; build-time staging cannot see fleet state.
 * pack_dir: directory holding the cartridge package (artifacts + descriptor + manifest). */
ctg_status ctg_preflight(const char* pack_dir, ctg_preflight_report* report);

/* -------------------------------------------------------------- lifecycle -- */

typedef struct ctg_cartridge ctg_cartridge;   /* opaque */
typedef struct ctg_session   ctg_session;     /* opaque */

typedef struct ctg_open_options {
  size_t      size;           /* caller sets sizeof(ctg_open_options)               */
  /* Reserved for cartridge-generic options; cartridge-specific configuration comes
   * from the package itself (descriptor + side files), NOT from host code — a host
   * that must pass model-specific knobs is coupling to one cartridge. */
  uint32_t    flags;          /* none defined in 0.1; pass 0                        */
} ctg_open_options;

/* Load the cartridge from pack_dir (heavy: models, tokenizer, runtime init).
 * opts MAY be NULL (defaults). On failure returns a status; ctg_last_error(NULL)
 * carries the message until the next failing open in this process. */
ctg_status ctg_open(const char* pack_dir, const ctg_open_options* opts, ctg_cartridge** out);
ctg_status ctg_close(ctg_cartridge* cartridge);

typedef struct ctg_session_options {
  size_t      size;           /* caller sets sizeof(ctg_session_options)            */
  uint32_t    flags;          /* none defined in 0.1; pass 0                        */
} ctg_session_options;

/* Sessions are cheap relative to open; one utterance/inference stream = one session.
 * A session MUST be driven from one thread at a time (ctg_cancel excepted). */
ctg_status ctg_session_begin(ctg_cartridge* cartridge, const ctg_session_options* opts,
                             ctg_session** out);
ctg_status ctg_session_end(ctg_session* session);

/* ---------------------------------------------------------------- buffers -- */

/* A borrowed, caller-owned view of contiguous bytes. Encoding (dtype, shape or
 * chunk size, sample rate) is defined by the descriptor's io block for the port
 * this buffer feeds — the ABI moves bytes; the descriptor says what they mean. */
typedef struct ctg_buffer {
  size_t      size;           /* caller sets sizeof(ctg_buffer)                     */
  const void* data;
  size_t      len;            /* bytes                                              */
} ctg_buffer;

/* ---------------------------------------------------------------- results -- */

typedef enum ctg_result_kind {
  CTG_RESULT_PARTIAL  = 1,    /* incremental output; may be superseded              */
  CTG_RESULT_FINAL    = 2,    /* committed output for the current segment/call      */
  CTG_RESULT_ENDPOINT = 3     /* DATA event: signal-level end-of-speech marker.
                                 Only cartridges with attributes.endpointing=signal
                                 emit this; acting on it is host policy (ADR-003). */
} ctg_result_kind;

typedef struct ctg_result ctg_result;         /* opaque, cartridge-allocated */

ctg_result_kind ctg_result_get_kind(const ctg_result* result);

/* Payload as a view owned by the result; valid until ctg_result_release(result).
 * Encoding per the descriptor's io.output port. */
ctg_status ctg_result_get_payload(const ctg_result* result, ctg_buffer* out_view);

void ctg_result_release(ctg_result* result);

/* --------------------------------------------------------------- io: batch -- */

/* Batch cartridges (descriptor io.mode == "batch") implement this; streaming
 * cartridges return CTG_E_UNSUPPORTED. One whole input, one FINAL result. */
ctg_status ctg_infer(ctg_session* session, const ctg_buffer* input, ctg_result** out);

/* ----------------------------------------------------------- io: streaming -- */

/* Streaming cartridges (descriptor io.mode == "streaming") implement these; batch
 * cartridges return CTG_E_UNSUPPORTED for all three. Pull-based: the host pushes
 * chunks and polls for results; the cartridge never calls the host. */
ctg_status ctg_push(ctg_session* session, const ctg_buffer* chunk);

/* *out == NULL with CTG_OK means "nothing pending" — poll again after more input
 * (or after flush, until it stays NULL, which means the stream is drained). */
ctg_status ctg_pull(ctg_session* session, ctg_result** out);

/* Signal end of input. After flush: push is CTG_E_STATE; drain remaining results
 * with ctg_pull until *out stays NULL. */
ctg_status ctg_flush(ctg_session* session);

/* ---------------------------------------------------------------- control -- */

/* Cooperative cancel; the ONLY call legal from another thread. The in-flight call
 * on the session returns CTG_E_CANCELLED as soon as the cartridge can stop; the
 * session then only accepts ctg_session_end. */
ctg_status ctg_cancel(ctg_session* session);

/* ----------------------------------------------------------------- errors -- */

/* Human-readable message for the most recent failure on this handle (a
 * ctg_cartridge* or ctg_session*), or on process-level calls when passed NULL.
 * Cartridge-owned storage, per-handle (NOT a process-global), stable until the
 * next failing call on the same handle. Diagnostic only — never parse it. */
const char* ctg_last_error(const void* cartridge_or_session);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CARTRIDGE_ABI_H */
