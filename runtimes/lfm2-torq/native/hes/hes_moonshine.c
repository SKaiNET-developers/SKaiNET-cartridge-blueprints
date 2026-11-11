/*
 * hes_moonshine: streaming Moonshine v2 ASR over the shared runtime (see hes_iree.h, "Moonshine streaming").
 *
 * A port of SKaiNET-transformers' Android streaming session (llm-runtime/iree-android/native/moonshine_stream_jni.c,
 * 74ca315): the same five-graph contract (moonshine-v2-streaming-v1), the same streaming loop and the same decode
 * policy, so a transcript here matches the phone's for the same audio and graphs:
 *
 *   feed PCM -> 64-frame windows, hop 44 -> frontend + encoder + adapter -> finalized cross memory
 *     -> incremental greedy decode -> cumulative partial
 *   finish() -> flush the tail, one exact full re-decode over the final memory -> final text
 *
 * Differences from the original:
 *   - plain C entry points instead of JNI; devices come from the shared hes_runtime, one per stage, so the encoder
 *     side can move to the NPU without touching the decoder (a view crossing devices goes through the host);
 *   - the decoder geometry is a runtime spec instead of compile-time macros;
 *   - scratch buffers belong to the engine instead of being function statics;
 *   - dec_embed.bin and the parameter archive are memory-mapped, not copied to the heap;
 *   - per-stage timing lines go to stderr only when asked for (the TUI owns the terminal);
 *   - the window at frame 0 gets no frontend left context instead of zero audio (see process_window), which the
 *     original feeds, and which costs it whole transcripts on clips that start with speech after a pause;
 *   - an empty partial (leading spaces only) is not surfaced;
 *   - with frontend_step.vmfb next to frontend.vmfb (the original's incremental frontend, 35103361), the frontend
 *     runs on 16 + 4 frames as the audio arrives, on the frontend's device, and a window or the flush only computes
 *     the frames past the fed audio;
 *   - when the encoder module also exports moonshine_v2_cross_kv (`cross_kv_cache` targets: the adapter's rows to
 *     every layer's cross K and V, side by side), the finalized rows' K/V go into a cache once, prefill reads the
 *     cache instead of the memory, and a prefill that only refreshes the cross views runs no graph.
 */
#define _GNU_SOURCE  /* mmap */
#include "hes_iree_internal.h"

#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define BOS 1
#define EOS 2

#define SPF 320                          /* audio samples per feature frame (4 x 80) */
#define CHUNK 64                         /* encoder window, feature frames (1.28 s) */
#define WINDOW 16                        /* encoder left context frames */
#define LOOKAHEAD 4                      /* encoder lookahead frames */
#define HOP (CHUNK - WINDOW - LOOKAHEAD) /* 44 frames = 0.88 s */
#define FE_LC 4                          /* frontend causal left-context frames re-fed per window */
#define FE_IN ((CHUNK + FE_LC) * SPF)    /* 21760 samples */
#define FE_OUT (CHUNK + FE_LC)           /* 68 frames out */
#define MAXMEM 256                       /* compiled cross-memory pad (5.12 s of finalized speech) */
/* Incremental frontend (optional frontend_step.vmfb next to frontend.vmfb): the same frontend exported for FE_STEP +
 * FE_LC frames, run as the audio arrives, so a window only runs the encoder and the adapter on features that exist.
 * The frontend is causal: output frame f reads 200 Hz frames 4f-12..4f, i.e. 3 frames of left context, so FE_LC = 4
 * frames of real audio in front of a chunk make its kept frames exact. */
#define FE_STEP 16                       /* frames per incremental frontend call (0.32 s) */
#define FES_IN ((FE_STEP + FE_LC) * SPF) /* 6400 samples */
#define FES_OUT (FE_STEP + FE_LC)        /* 20 frames out */
#define MAXF (MAXMEM + CHUNK + FES_OUT)  /* feature cache, frames */
#define FE_RF 4                          /* frames past `produced` that still see audio (3 of context + a partly fed last one) */
#define MAXTOK 48
#define MAXLAYERS 16
#define FINISH_TOKENS 24                    /* hard ceiling for the final budget (see finish_budget) */
#define TOKENS_PER_SAMPLE (6.5f / 16000.0f) /* the model card's cap: max_new_tokens = samples * 6.5/16000 + 2 */
#define HOP_TOKENS 6                        /* max new tokens decoded per hop */
#define RESTART_HOPS 2                      /* first hops re-decode exactly instead of incrementally */
#define PEEK_FRAMES 36                      /* early peek: first partial at ~0.72 s instead of 1.28 s */
#define STEP_SLICE 2                        /* tokens decoded per feed call: partials trickle out mid-hop */
#define DEFAULT_LAG_BUDGET_MS 250
#define MAXEVT 24
#define EVT_WINDOW 0
#define EVT_PEEK 1
#define EVT_HOP 2
#define EVT_DROPPED 3
#define MAXPCM ((MAXMEM + CHUNK) * SPF) /* capacity, samples */
#define NEGMASK (-1.0e30f)

static const char* FN_FE = "module.moonshine_v2_frontend";
static const char* FN_ENC = "module.moonshine_v2_encoder";
static const char* FN_ADP = "module.moonshine_v2_adapter";
static const char* FN_XKV = "module.moonshine_v2_cross_kv";  /* optional, in the encoder module */
static const char* FN_PRE = "module.moonshine_v2_decoder_prefill";
static const char* FN_STEP = "module.moonshine_v2_decoder_with_past";

typedef struct { int n; char** t; } Vocab;

struct hes_ms {
  hes_runtime* rt;
  hes_session fe, enc, adp, pre, step;  /* each on its stage's device (borrowed from the runtime) */
  hes_session fes;                      /* incremental frontend on the frontend's device; session NULL = absent */
  int nl, hd, dim, enc_dim, rot;
  int lag_budget_ms, fast_finish, verbose, no_peek, partial_every;
  Vocab vocab;
  const float* embed; void* embed_map; size_t embed_len; /* [vocab.n][dim], mapped */
  float* rope_c, *rope_s;                                /* [MAXTOK + 4][hd] */

  /* scratch */
  float* fein;   /* FE_IN */
  float* feats;  /* FE_OUT x enc_dim */
  float* fesin;  /* FES_IN */
  float* fesout; /* FES_OUT x enc_dim */
  float* fe_silence; /* enc_dim: the frontend's output past the audio (every frame of zero padding beyond FE_RF) */
  float* mem64;  /* CHUNK x dim */
  float* mem256; /* MAXMEM x dim */
  float* mask;   /* MAXMEM */
  float* kvwin;  /* CHUNK x (2 nl x dim): the window rows' K/V, output j of a row at j x dim, head h at h x hd */
  float* row;    /* vocab.n */
  void* xfer; size_t xfer_len; /* host bounce buffer for views crossing devices */

  /* utterance state */
  float* pcm; int npcm;
  float* feat; int fe_done;  /* MAXF x enc_dim features of this utterance; [0, fe_done) exact, never recomputed */
  double fe_feed_ms; int fe_calls;  /* incremental frontend cost while the audio streams in */
  int win;
  float* mem; int nmem;  /* MAXMEM x dim finalized memory rows */
  int kv_cached;         /* the encoder module exports FN_XKV */
  float* kvc;            /* per layer K, V: heads x MAXMEM x hd, the finalized rows' cross K/V */
  iree_hal_buffer_view_t* sk[MAXLAYERS], *sv[MAXLAYERS], *ck[MAXLAYERS], *cv[MAXLAYERS];
  int has_self, has_cross;
  int toks[MAXTOK]; int ntoks; int pos;
  int peeked, ended_eos;
  size_t shown_len;
  int pending, halted;
  double t_first_feed;
  int dropped, windows, hops, errors;
  double prefill_ms, step_ms; int prefills, steps;  /* decoder cost split, per utterance */
  short evt_at[MAXEVT], evt_a[MAXEVT], evt_b[MAXEVT], evt_c[MAXEVT];
  unsigned char evt_kind[MAXEVT];
  int nevt;
  char text[4096];
  char out[4096];
  char stats[1400];
};

static void tlog(const hes_ms* e, const char* fmt, ...) {
  if (!e->verbose) return;
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
}

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void note_evt(hes_ms* e, int kind, double at, double a, double b, double c) {
  if (e->nevt >= MAXEVT) return;
  int i = e->nevt++;
  e->evt_kind[i] = (unsigned char)kind;
  e->evt_at[i] = (short)(at < 0 ? 0 : (at > 32000 ? 32000 : at));
  e->evt_a[i] = (short)(a > 32000 ? 32000 : a);
  e->evt_b[i] = (short)(b > 32000 ? 32000 : b);
  e->evt_c[i] = (short)(c > 32000 ? 32000 : c);
}

/* ---------- vocab (u32 count, then u16 len + utf8 piece per id; U+2581 = word boundary) ---------- */

static Vocab vocab_load(const char* p) {
  Vocab v = {0, NULL};
  FILE* f = fopen(p, "rb");
  if (!f) return v;
  uint32_t n = 0;
  if (fread(&n, 4, 1, f) != 1 || n == 0 || n > 1000000) { fclose(f); return v; }
  v.n = (int)n;
  v.t = calloc(n, sizeof(char*));
  for (uint32_t i = 0; i < n; ++i) {
    uint16_t l = 0;
    if (fread(&l, 2, 1, f) != 1) break;
    char* s = malloc((size_t)l + 1);
    if (l && fread(s, 1, l, f) != l) { free(s); break; }
    s[l] = 0;
    if (l >= 2 && s[0] == '<' && s[l - 1] == '>') s[0] = 0; /* specials decode to nothing */
    v.t[i] = s;
  }
  fclose(f);
  return v;
}

static void vocab_free(Vocab* v) {
  for (int i = 0; i < v->n; ++i) free(v->t[i]);
  free(v->t);
  v->t = NULL; v->n = 0;
}

static void detok_append(const Vocab* v, int id, char* out, size_t cap) {
  if (id < 0 || id >= v->n || !v->t[id] || !v->t[id][0]) return;
  const char* t = v->t[id];
  size_t ol = strlen(out);
  while (*t && ol + 4 < cap) {
    if ((unsigned char)t[0] == 0xE2 && (unsigned char)t[1] == 0x96 && (unsigned char)t[2] == 0x81) {
      out[ol++] = ' '; t += 3;
    } else {
      out[ol++] = *t++;
    }
  }
  out[ol] = 0;
}

/* ---------- IREE plumbing ---------- */

static iree_hal_buffer_view_t* mkview(hes_ms* e, hes_session* s, const iree_hal_dim_t* shape, int rank,
                                      iree_hal_element_type_t et, const void* data, size_t bytes) {
  iree_hal_buffer_view_t* bv = NULL;
  if (hes__check(hes__view_copy(s->device, et, rank, shape, data, bytes, &bv), "moonshine view")) { e->errors++; return NULL; }
  return bv;
}

static int pull(hes_ms* e, hes_session* s, iree_hal_buffer_view_t* bv, void* dst, size_t bytes) {
  if (iree_hal_buffer_view_byte_length(bv) < bytes) {
    hes__set_error("moonshine: view has %zu bytes, %zu wanted", (size_t)iree_hal_buffer_view_byte_length(bv), bytes);
    e->errors++;
    return -1;
  }
  if (hes__check(iree_hal_device_transfer_d2h(s->device, iree_hal_buffer_view_buffer(bv), 0, dst, bytes,
                                              IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()),
                 "moonshine d2h")) { e->errors++; return -1; }
  return 0;
}

/* `v` (from `from`'s device) as a view usable on `to`'s device: itself when they share one, a host bounce otherwise. */
static iree_hal_buffer_view_t* to_device(hes_ms* e, hes_session* from, hes_session* to, iree_hal_buffer_view_t* v) {
  if (from->device == to->device) { iree_hal_buffer_view_retain(v); return v; }
  size_t bytes = (size_t)iree_hal_buffer_view_byte_length(v);
  if (bytes > e->xfer_len) {
    void* p = realloc(e->xfer, bytes);
    if (!p) return NULL;
    e->xfer = p; e->xfer_len = bytes;
  }
  if (pull(e, from, v, e->xfer, bytes)) return NULL;
  iree_host_size_t rank = iree_hal_buffer_view_shape_rank(v);
  iree_hal_dim_t dims[8];
  for (iree_host_size_t i = 0; i < rank && i < 8; ++i) dims[i] = iree_hal_buffer_view_shape_dim(v, i);
  return mkview(e, to, dims, (int)rank, iree_hal_buffer_view_element_type(v), e->xfer, bytes);
}

/* Calls `fn` with `ins`; pops up to `cap` outputs into `outs`. Returns the output count, or -1. */
static int call(hes_ms* e, hes_session* s, const char* fn, iree_hal_buffer_view_t* const* ins, int n_in,
                iree_hal_buffer_view_t** outs, int cap) {
  iree_runtime_call_t c;
  if (hes__check(iree_runtime_call_initialize_by_name(s->session, iree_make_cstring_view(fn), &c), fn)) { e->errors++; return -1; }
  iree_status_t st = iree_ok_status();
  for (int i = 0; i < n_in && iree_status_is_ok(st); ++i) st = iree_runtime_call_inputs_push_back_buffer_view(&c, ins[i]);
  if (iree_status_is_ok(st)) st = iree_runtime_call_invoke(&c, 0);
  int n = 0;
  if (iree_status_is_ok(st)) {
    int produced = (int)iree_vm_list_size(iree_runtime_call_outputs(&c));
    if (produced > cap) produced = cap;
    for (; n < produced && iree_status_is_ok(st); ++n) st = iree_runtime_call_outputs_pop_front_buffer_view(&c, &outs[n]);
    if (!iree_status_is_ok(st)) { for (int i = 0; i < n; ++i) iree_hal_buffer_view_release(outs[i]); n = 0; }
  }
  iree_runtime_call_deinitialize(&c);
  if (hes__check(st, fn)) { e->errors++; return -1; }
  return n;
}

static iree_hal_buffer_view_t* run1(hes_ms* e, hes_session* s, const char* fn, iree_hal_buffer_view_t* in) {
  iree_hal_buffer_view_t* out = NULL;
  return call(e, s, fn, &in, 1, &out, 1) == 1 ? out : NULL;
}

static int argmax_logits(hes_ms* e, iree_hal_buffer_view_t* bv) {
  if (!bv || pull(e, &e->step, bv, e->row, (size_t)e->vocab.n * sizeof(float))) return EOS;
  int best = 0;
  float b = e->row[0];
  for (int i = 1; i < e->vocab.n; ++i) if (e->row[i] > b) { b = e->row[i]; best = i; }
  return best;
}

static void rope_init(hes_ms* e) {
  const int half = e->hd / 2;
  for (int p = 0; p < MAXTOK + 4; ++p) {
    float* c = e->rope_c + (size_t)p * e->hd;
    float* s = e->rope_s + (size_t)p * e->hd;
    for (int i = 0; i < half; ++i) {
      double cv = 1.0, sv = 0.0;
      if (i < e->rot) { double a = p * pow(10000.0, -(double)i / e->rot); cv = cos(a); sv = sin(a); }
      c[2 * i] = (float)cv; c[2 * i + 1] = (float)cv;
      s[2 * i] = (float)-sv; s[2 * i + 1] = (float)sv;
    }
  }
}

static void release_self(hes_ms* e) {
  if (!e->has_self) return;
  for (int l = 0; l < e->nl; ++l) { iree_hal_buffer_view_release(e->sk[l]); iree_hal_buffer_view_release(e->sv[l]); }
  e->has_self = 0;
}

static void release_cross(hes_ms* e) {
  if (!e->has_cross) return;
  for (int l = 0; l < e->nl; ++l) { iree_hal_buffer_view_release(e->ck[l]); iree_hal_buffer_view_release(e->cv[l]); }
  e->has_cross = 0;
}

/* The model card's output-length cap for `npcm` samples, floored at 4 and ceilinged at FINISH_TOKENS. */
static int finish_budget(int npcm) {
  int b = (int)(npcm * TOKENS_PER_SAMPLE) + 2;
  if (b < 4) b = 4;
  if (b > FINISH_TOKENS) b = FINISH_TOKENS;
  return b;
}

/* ---------- stages ---------- */

/* One incremental frontend call for the chunk starting at frame `s`, zero-padded past the fed audio, fed from sample 0
 * when s < FE_LC (the model's own padding is then the utterance start, see process_window). Writes frames s ..
 * s+n-1 into e->feat and returns n, or -1. */
static int fe_chunk(hes_ms* e, int s) {
  const int lc = s < FE_LC ? s : FE_LC;
  const int s0 = (s - lc) * SPF;
  for (int i = 0; i < FES_IN; ++i) {
    int si = s0 + i;
    e->fesin[i] = (si >= 0 && si < e->npcm) ? e->pcm[si] : 0.0f;
  }
  iree_hal_buffer_view_t* v = mkview(e, &e->fes, (iree_hal_dim_t[]){1, FES_IN}, 2, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                                     e->fesin, FES_IN * sizeof(float));
  iree_hal_buffer_view_t* o = v ? run1(e, &e->fes, FN_FE, v) : NULL;
  if (v) iree_hal_buffer_view_release(v);
  if (!o) return -1;
  int rc = pull(e, &e->fes, o, e->fesout, (size_t)FES_OUT * e->enc_dim * sizeof(float));
  iree_hal_buffer_view_release(o);
  if (rc) return -1;
  int n = FES_OUT - lc;
  if (s + n > MAXF) n = MAXF - s;
  memcpy(e->feat + (size_t)s * e->enc_dim, e->fesout + (size_t)lc * e->enc_dim, (size_t)n * e->enc_dim * sizeof(float));
  return n;
}

/* Extends the exact features as far as the fed audio allows, in whole FE_STEP chunks. */
static void fe_advance(hes_ms* e, int produced) {
  while (e->fes.session && e->fe_done + FE_STEP <= produced && e->fe_done + FES_OUT <= MAXF) {
    double t0 = now_ms();
    int n = fe_chunk(e, e->fe_done);
    if (n <= 0) break;
    e->fe_done += n < produced - e->fe_done ? n : produced - e->fe_done;
    e->fe_feed_ms += now_ms() - t0; e->fe_calls++;
  }
}

/* Frontend + encoder + adapter over the window at feature frame `start`; finalizes rows into e->mem. */
static int process_window(hes_ms* e, int start, int produced, int flush, int peek) {
  double t0 = now_ms();
  /* The frontend re-reads FE_LC frames before the window as its convolutions' left context. At the utterance start
   * there is none, and zero samples are not the same thing: after the first convolution's bias they are non-zero
   * features, where the model (and the graph's own padding) sees zeros. That corrupts the first rows of the memory
   * and, through attention, everything after them (measured against the reference model: an empty transcript where
   * it reads "Fail again, fail better."). So the window at frame 0 feeds the audio from sample 0 and takes the first
   * 64 output frames: the graph's internal zero padding is then exactly the reference start. */
  int lc = start == 0 ? 0 : FE_LC;
  const float* encsrc;
  int rc = 0;
  if (e->fes.session) {
    /* Features up to the fed audio exist already; only the part of the window past them (a peek's or the flush's
     * zero-padded tail) is computed here, and only as far as a frame can still see audio: the rest is padding, whose
     * feature is a constant (from the warm-up). The window at frame 0 keeps the graph throughout: there the
     * zero-padded start is the model's own and FE_RF does not describe it. */
    fe_advance(e, produced);
    int need = start == 0 ? start + CHUNK : produced + FE_RF;
    if (need > start + CHUNK) need = start + CHUNK;
    int s = e->fe_done;
    while (s < need) {
      int n = fe_chunk(e, s);
      if (n <= 0) return -1;
      s += n;
    }
    for (int f = s > need ? need : s; f < start + CHUNK; ++f)
      memcpy(e->feat + (size_t)f * e->enc_dim, e->fe_silence, (size_t)e->enc_dim * sizeof(float));
    if (produced > e->fe_done) e->fe_done = produced < s ? produced : s;
    encsrc = e->feat + (size_t)start * e->enc_dim;
  } else {
    int s0 = (start - lc) * SPF;
    for (int i = 0; i < FE_IN; ++i) {
      int si = s0 + i;
      e->fein[i] = (si >= 0 && si < e->npcm) ? e->pcm[si] : 0.0f;
    }
    iree_hal_buffer_view_t* fev = mkview(e, &e->fe, (iree_hal_dim_t[]){1, FE_IN}, 2, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                                         e->fein, FE_IN * sizeof(float));
    if (!fev) return -1;
    iree_hal_buffer_view_t* feo = run1(e, &e->fe, FN_FE, fev);
    iree_hal_buffer_view_release(fev);
    if (!feo) return -1;
    size_t feats_bytes = (size_t)FE_OUT * e->enc_dim * sizeof(float);
    rc = pull(e, &e->fe, feo, e->feats, feats_bytes);
    iree_hal_buffer_view_release(feo);
    if (rc) return -1;
    encsrc = e->feats + (size_t)lc * e->enc_dim;
  }
  double t1 = now_ms();

  /* encoder on the clean 64 frames (drop the FE_LC context rows) */
  iree_hal_buffer_view_t* encin = mkview(e, &e->enc, (iree_hal_dim_t[]){1, CHUNK, (iree_hal_dim_t)e->enc_dim}, 3,
                                         IREE_HAL_ELEMENT_TYPE_FLOAT_32, encsrc,
                                         (size_t)CHUNK * e->enc_dim * sizeof(float));
  iree_hal_buffer_view_t* enco = encin ? run1(e, &e->enc, FN_ENC, encin) : NULL;
  if (encin) iree_hal_buffer_view_release(encin);
  if (!enco) return -1;
  double t2 = now_ms();

  /* adapter: positions start..start+63 */
  int32_t posv[CHUNK];
  for (int i = 0; i < CHUNK; ++i) posv[i] = start + i;
  iree_hal_buffer_view_t* ins[2] = {
      mkview(e, &e->adp, (iree_hal_dim_t[]){1, CHUNK}, 2, IREE_HAL_ELEMENT_TYPE_INT_32, posv, sizeof posv),
      to_device(e, &e->enc, &e->adp, enco)};
  iree_hal_buffer_view_release(enco);
  iree_hal_buffer_view_t* adpo = NULL;
  int na = ins[0] && ins[1] ? call(e, &e->adp, FN_ADP, ins, 2, &adpo, 1) : -1;
  for (int i = 0; i < 2; ++i) if (ins[i]) iree_hal_buffer_view_release(ins[i]);
  if (na != 1) {
    if (na == 0) { hes__set_error("moonshine adapter: no output"); e->errors++; }
    return -1;
  }
  rc = pull(e, &e->adp, adpo, e->mem64, (size_t)CHUNK * e->dim * sizeof(float));
  /* the window rows' cross K/V, next to the encoder */
  const size_t kvrow = (size_t)2 * e->nl * e->dim;
  if (rc == 0 && e->kv_cached) {
    iree_hal_buffer_view_t* xin = to_device(e, &e->adp, &e->enc, adpo);
    iree_hal_buffer_view_t* xkv = xin ? run1(e, &e->enc, FN_XKV, xin) : NULL;
    if (xin) iree_hal_buffer_view_release(xin);
    rc = xkv ? pull(e, &e->enc, xkv, e->kvwin, CHUNK * kvrow * sizeof(float)) : -1;
    if (xkv) iree_hal_buffer_view_release(xkv);
  }
  iree_hal_buffer_view_release(adpo);

  /* finalize [nmem, new_final); at flush the lookahead holds nothing back */
  int limit = flush ? start + CHUNK : start + CHUNK - LOOKAHEAD;
  int new_final = limit < produced ? limit : produced;
  if (new_final > MAXMEM) new_final = MAXMEM;
  /* The real window 0 overwrites rows an early peek finalized with a zero-padded tail; regular windows only append
   * (their overlap rows were finalized with a fuller attention band by the previous window). */
  int first = (!peek && start == 0 && e->peeked) ? 0 : e->nmem;
  for (int f = first; rc == 0 && f < new_final; ++f)
    memcpy(e->mem + (size_t)f * e->dim, e->mem64 + (size_t)(f - start) * e->dim, (size_t)e->dim * sizeof(float));
  /* the same rows' cross K/V into [heads, MAXMEM, hd] per layer and side */
  const int heads = e->dim / e->hd;
  for (int f = first; rc == 0 && e->kv_cached && f < new_final; ++f)
    for (int j = 0; j < 2 * e->nl; ++j)
      for (int h = 0; h < heads; ++h)
        memcpy(e->kvc + (((size_t)j * heads + h) * MAXMEM + f) * e->hd, e->kvwin + (size_t)(f - start) * kvrow + (size_t)j * e->dim + (size_t)h * e->hd,
               (size_t)e->hd * sizeof(float));
  if (rc) return -1;
  if (new_final > e->nmem) e->nmem = new_final;
  if (!peek && start == 0) e->peeked = 0;
  double t3 = now_ms();
  e->windows++;
  note_evt(e, EVT_WINDOW, t0 - e->t_first_feed, t1 - t0, t2 - t1, t3 - t2);
  tlog(e, "win %d fe %.0fms enc %.0fms adp+fin %.0fms nmem %d\n", e->win, t1 - t0, t2 - t1, t3 - t2, e->nmem);
  return 0;
}

static void fill_mask(hes_ms* e) {
  for (int f = 0; f < MAXMEM; ++f) e->mask[f] = f < e->nmem ? 0.0f : NEGMASK;
}

/* Cross views over the K/V cache, and with `start_tokens` the prefill that reads them (see run_prefill). */
static int run_prefill_cached(hes_ms* e, int start_tokens, double tp) {
  size_t row = (size_t)e->dim;
  const int heads = e->dim / e->hd;
  size_t side = (size_t)heads * MAXMEM * e->hd;
  iree_hal_dim_t kvshape[4] = {1, (iree_hal_dim_t)heads, MAXMEM, (iree_hal_dim_t)e->hd};
  release_cross(e);
  int ok = 1;
  for (int l = 0; l < e->nl; ++l) {
    e->ck[l] = mkview(e, &e->pre, kvshape, 4, IREE_HAL_ELEMENT_TYPE_FLOAT_32, e->kvc + 2 * l * side, side * sizeof(float));
    e->cv[l] = mkview(e, &e->pre, kvshape, 4, IREE_HAL_ELEMENT_TYPE_FLOAT_32, e->kvc + (2 * l + 1) * side, side * sizeof(float));
    ok = ok && e->ck[l] && e->cv[l];
  }
  e->has_cross = 1;
  if (!ok) return -1;
  e->halted = 0; e->ended_eos = 0;
  if (!start_tokens) { e->prefill_ms += now_ms() - tp; return 0; }  /* counted as time, not as a prefill call */

  fill_mask(e);
  iree_hal_dim_t d = (iree_hal_dim_t)e->dim;
  /* inputs: BOS embedding, per layer cross K, cross V, then the mask; outputs: per layer self K, self V, then logits */
  iree_hal_buffer_view_t* ins[2 + 2 * MAXLAYERS] = {0};
  int n_in = 0;
  ins[n_in++] = mkview(e, &e->pre, (iree_hal_dim_t[]){1, 1, d}, 3, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                       e->embed + (size_t)BOS * row, row * sizeof(float));
  for (int l = 0; l < e->nl; ++l) { ins[n_in++] = e->ck[l]; ins[n_in++] = e->cv[l]; }
  iree_hal_buffer_view_t* maskv = mkview(e, &e->pre, (iree_hal_dim_t[]){1, 1, 1, MAXMEM}, 4, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                                         e->mask, MAXMEM * sizeof(float));
  ins[n_in++] = maskv;
  iree_hal_buffer_view_t* outs[2 * MAXLAYERS + 1] = {0};
  int want = 2 * e->nl + 1;
  int n = ins[0] && maskv ? call(e, &e->pre, FN_PRE, ins, n_in, outs, want) : -1;
  if (ins[0]) iree_hal_buffer_view_release(ins[0]);
  if (maskv) iree_hal_buffer_view_release(maskv);
  if (n != want) {
    for (int i = 0; i < n; ++i) iree_hal_buffer_view_release(outs[i]);
    if (n >= 0) { hes__set_error("moonshine prefill: %d outputs, %d expected", n, want); e->errors++; }
    return -1;
  }
  release_self(e);
  for (int l = 0; l < e->nl; ++l) { e->sk[l] = outs[2 * l]; e->sv[l] = outs[2 * l + 1]; }
  e->has_self = 1;
  int tok = argmax_logits(e, outs[2 * e->nl]);
  iree_hal_buffer_view_release(outs[2 * e->nl]);
  e->ntoks = 0; e->pos = 1; e->text[0] = 0;
  if (tok != EOS) { e->toks[e->ntoks++] = tok; detok_append(&e->vocab, tok, e->text, sizeof e->text); }
  e->prefill_ms += now_ms() - tp; e->prefills++;
  return 0;
}

/* Prefill over the finalized memory: refreshes cross-K/V and, with `start_tokens`, restarts the token sequence. */
static int run_prefill(hes_ms* e, int start_tokens) {
  double tp = now_ms();
  if (e->kv_cached) return run_prefill_cached(e, start_tokens, tp);
  size_t row = (size_t)e->dim;
  memcpy(e->mem256, e->mem, (size_t)e->nmem * row * sizeof(float));
  memset(e->mem256 + (size_t)e->nmem * row, 0, (size_t)(MAXMEM - e->nmem) * row * sizeof(float));
  fill_mask(e);

  iree_hal_dim_t d = (iree_hal_dim_t)e->dim;
  iree_hal_buffer_view_t* ins[3] = {
      mkview(e, &e->pre, (iree_hal_dim_t[]){1, 1, d}, 3, IREE_HAL_ELEMENT_TYPE_FLOAT_32, e->embed + (size_t)BOS * row,
             row * sizeof(float)),
      mkview(e, &e->pre, (iree_hal_dim_t[]){1, MAXMEM, d}, 3, IREE_HAL_ELEMENT_TYPE_FLOAT_32, e->mem256,
             (size_t)MAXMEM * row * sizeof(float)),
      mkview(e, &e->pre, (iree_hal_dim_t[]){1, 1, 1, MAXMEM}, 4, IREE_HAL_ELEMENT_TYPE_FLOAT_32, e->mask,
             MAXMEM * sizeof(float))};
  iree_hal_buffer_view_t* outs[4 * MAXLAYERS + 1] = {0};
  int want = 4 * e->nl + 1;
  int n = (ins[0] && ins[1] && ins[2]) ? call(e, &e->pre, FN_PRE, ins, 3, outs, want) : -1;
  for (int i = 0; i < 3; ++i) if (ins[i]) iree_hal_buffer_view_release(ins[i]);
  if (n != want) {
    for (int i = 0; i < n; ++i) iree_hal_buffer_view_release(outs[i]);
    if (n >= 0) { hes__set_error("moonshine prefill: %d outputs, %d expected", n, want); e->errors++; }
    return -1;
  }
  /* outputs per layer: self K, self V, cross K, cross V; then the logits */
  release_cross(e);
  for (int l = 0; l < e->nl; ++l) { e->ck[l] = outs[4 * l + 2]; e->cv[l] = outs[4 * l + 3]; }
  e->has_cross = 1;
  iree_hal_buffer_view_t* logits = outs[4 * e->nl];

  e->halted = 0; e->ended_eos = 0;
  if (start_tokens) {
    release_self(e);
    for (int l = 0; l < e->nl; ++l) { e->sk[l] = outs[4 * l]; e->sv[l] = outs[4 * l + 1]; }
    e->has_self = 1;
    int tok = argmax_logits(e, logits);
    e->ntoks = 0; e->pos = 1; e->text[0] = 0;
    if (tok != EOS) { e->toks[e->ntoks++] = tok; detok_append(&e->vocab, tok, e->text, sizeof e->text); }
  } else {
    for (int l = 0; l < e->nl; ++l) { iree_hal_buffer_view_release(outs[4 * l]); iree_hal_buffer_view_release(outs[4 * l + 1]); }
  }
  iree_hal_buffer_view_release(logits);
  e->prefill_ms += now_ms() - tp; e->prefills++;
  return 0;
}

static void rebuild_text(hes_ms* e) {
  e->text[0] = 0;
  for (int i = 0; i < e->ntoks; ++i) detok_append(&e->vocab, e->toks[i], e->text, sizeof e->text);
}

/* A proper suffix equal to a prefix of the tokens is a restart of the utterance on a silent tail: drop it, repeatedly.
 * Applied only to finals that did not end on EOS. */
static void drop_restart_suffix(hes_ms* e) {
  int again = 1;
  while (again && e->ntoks >= 2) {
    again = 0;
    for (int s2 = e->ntoks / 2; s2 >= 1; --s2) {
      if (memcmp(e->toks, e->toks + e->ntoks - s2, (size_t)s2 * sizeof(int)) == 0) { e->ntoks -= s2; again = 1; break; }
    }
  }
  rebuild_text(e);
}

/* Cycle guard: periods >= 3 trip on the first repeat, periods 1-2 only on a triple ("very very" survives). */
static int repeat_period(const int* t, int n) {
  for (int p = 1; p <= n / 2; ++p) {
    int need = p >= 3 ? 2 : 3;
    if (n < p * need) continue;
    int same = 1;
    for (int i = 0; i < p * (need - 1) && same; ++i)
      if (t[n - 1 - i] != t[n - 1 - i - p]) same = 0;
    if (same) return p;
  }
  return 0;
}

/* Greedy steps from the current state; stops on EOS / cycle / budget / MAXTOK. `trim` drops a detected cycle's
 * repeat from the tokens. Returns the number of new tokens. */
static int run_steps(hes_ms* e, int budget, int trim) {
  fill_mask(e);
  size_t row = (size_t)e->dim;
  iree_hal_dim_t d = (iree_hal_dim_t)e->dim, hd = (iree_hal_dim_t)e->hd;
  int added = 0;
  while (e->ntoks > 0 && e->ntoks < MAXTOK && added < budget && e->pos < MAXTOK) {
    double ts = now_ms();
    int last = e->toks[e->ntoks - 1];
    iree_hal_buffer_view_t* ev = mkview(e, &e->step, (iree_hal_dim_t[]){1, 1, d}, 3, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                                        e->embed + (size_t)last * row, row * sizeof(float));
    iree_hal_buffer_view_t* cq = mkview(e, &e->step, (iree_hal_dim_t[]){1, hd}, 2, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                                        e->rope_c + (size_t)e->pos * e->hd, (size_t)e->hd * sizeof(float));
    iree_hal_buffer_view_t* sq = mkview(e, &e->step, (iree_hal_dim_t[]){1, hd}, 2, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                                        e->rope_s + (size_t)e->pos * e->hd, (size_t)e->hd * sizeof(float));
    iree_hal_buffer_view_t* maskv = mkview(e, &e->step, (iree_hal_dim_t[]){1, 1, 1, MAXMEM}, 4,
                                           IREE_HAL_ELEMENT_TYPE_FLOAT_32, e->mask, MAXMEM * sizeof(float));
    int ok = ev && cq && sq && maskv;
    /* inputs: embedding, rope cos, rope sin, then per layer self K, self V, cross K, cross V (mask after layer 0) */
    iree_hal_buffer_view_t* ins[3 + 4 * MAXLAYERS + 1];
    int n_in = 0;
    ins[n_in++] = ev; ins[n_in++] = cq; ins[n_in++] = sq;
    for (int l = 0; l < e->nl; ++l) {
      ins[n_in++] = e->sk[l]; ins[n_in++] = e->sv[l]; ins[n_in++] = e->ck[l]; ins[n_in++] = e->cv[l];
      if (l == 0) ins[n_in++] = maskv;
    }
    iree_hal_buffer_view_t* outs[2 * MAXLAYERS + 1] = {0};
    int want = 2 * e->nl + 1;
    int n = ok ? call(e, &e->step, FN_STEP, ins, n_in, outs, want) : -1;
    if (ev) iree_hal_buffer_view_release(ev);
    if (cq) iree_hal_buffer_view_release(cq);
    if (sq) iree_hal_buffer_view_release(sq);
    if (maskv) iree_hal_buffer_view_release(maskv);
    if (n != want) {
      for (int i = 0; i < n; ++i) iree_hal_buffer_view_release(outs[i]);
      if (n >= 0) { hes__set_error("moonshine step: %d outputs, %d expected", n, want); e->errors++; }
      return added;
    }
    for (int l = 0; l < e->nl; ++l) {
      iree_hal_buffer_view_release(e->sk[l]); iree_hal_buffer_view_release(e->sv[l]);
      e->sk[l] = outs[2 * l]; e->sv[l] = outs[2 * l + 1];
    }
    int nxt = argmax_logits(e, outs[2 * e->nl]);
    iree_hal_buffer_view_release(outs[2 * e->nl]);
    e->step_ms += now_ms() - ts; e->steps++;
    e->pos++;
    if (nxt == EOS) { e->halted = 1; e->ended_eos = 1; break; } /* tentative while streaming */
    e->toks[e->ntoks++] = nxt;
    detok_append(&e->vocab, nxt, e->text, sizeof e->text);
    added++;
    int p = repeat_period(e->toks, e->ntoks);
    if (p > 0) {
      if (trim) { e->ntoks -= p >= 3 ? p : 2 * p; rebuild_text(e); } /* keep one instance of the looping p-gram */
      e->halted = 1;
      break;
    }
  }
  return added;
}

static void reset_utterance(hes_ms* e) {
  e->npcm = 0; e->win = 0; e->nmem = 0; e->fe_done = 0; e->fe_feed_ms = 0.0; e->fe_calls = 0;
  e->peeked = 0; e->pending = 0; e->halted = 0; e->shown_len = 0;
  e->t_first_feed = 0.0; e->dropped = 0; e->windows = 0; e->hops = 0; e->nevt = 0; e->errors = 0;
  e->prefill_ms = e->step_ms = 0.0; e->prefills = e->steps = 0;
  release_self(e); release_cross(e);
  e->ntoks = 0; e->pos = 0; e->text[0] = 0;
}

/* One zero-input pass through every graph, so lazy first-dispatch costs (pipeline compilation on a GPU, page-ins of
 * the mapped weights on a CPU) do not land in the first utterance. */
static int warmup(hes_ms* e) {
  double t0 = now_ms();
  e->npcm = 0;
  int rc = process_window(e, 0, CHUNK, 1, 0);
  /* the padding feature: a chunk at s >= FE_LC over zeros, any kept frame */
  if (rc == 0 && e->fes.session) rc = fe_chunk(e, FE_LC) > 0 ? 0 : -1;
  if (rc == 0 && e->fes.session) memcpy(e->fe_silence, e->feat + (size_t)FE_LC * e->enc_dim, (size_t)e->enc_dim * sizeof(float));
  if (rc == 0) {
    memset(e->mem, 0, (size_t)MAXMEM * e->dim * sizeof(float));
    e->nmem = CHUNK;
    rc = run_prefill(e, 1);
    if (rc == 0 && e->ntoks > 0) run_steps(e, 1, 0);
  }
  int errors = e->errors;
  reset_utterance(e);
  tlog(e, "warmup %.0fms\n", now_ms() - t0);
  return rc == 0 && errors == 0 ? 0 : -1;
}

/* ---------- public API ---------- */

void hes_ms_release(hes_ms* e) {
  if (!e) return;
  reset_utterance(e);
  hes__close_session(&e->fe); hes__close_session(&e->enc); hes__close_session(&e->adp);
  hes__close_session(&e->pre); hes__close_session(&e->step); hes__close_session(&e->fes);
  vocab_free(&e->vocab);
  if (e->embed_map) munmap(e->embed_map, e->embed_len);
  free(e->rope_c); free(e->rope_s);
  free(e->fein); free(e->feats); free(e->mem64); free(e->mem256); free(e->mask); free(e->kvwin); free(e->kvc);
  free(e->fesin); free(e->fesout); free(e->fe_silence); free(e->feat);
  free(e->row); free(e->xfer);
  free(e->pcm); free(e->mem);
  free(e);
}

static int open_stage(hes_ms* e, hes_session* s, const char* device, const char* vmfb, const char* irpa) {
  iree_hal_device_t* dev = hes__device_for(e->rt, device);
  if (!dev) return -1;
  char what[512];
  snprintf(what, sizeof what, "moonshine %s on %s", vmfb, device);
  return hes__check(hes__open_session(e->rt, dev, vmfb, irpa, NULL, s, NULL), what);
}

hes_ms* hes_ms_create(hes_runtime* rt, const hes_ms_spec* spec,
                      const char* dev_frontend, const char* dev_encoder, const char* dev_adapter, const char* dev_decoder,
                      const char* vmfb_frontend, const char* vmfb_encoder, const char* vmfb_adapter,
                      const char* vmfb_prefill, const char* vmfb_step,
                      const char* irpa, const char* vocab_bin, const char* dec_embed_bin) {
  hes__clear_error();
  hes_ms* e = calloc(1, sizeof *e);
  e->rt = rt;
  e->nl = spec && spec->n_layers > 0 ? spec->n_layers : 6;
  e->hd = spec && spec->head_dim > 0 ? spec->head_dim : 40;
  e->rot = spec && spec->rope_pairs > 0 ? spec->rope_pairs : 16;
  e->lag_budget_ms = !spec || spec->lag_budget_ms == 0 ? DEFAULT_LAG_BUDGET_MS : (spec->lag_budget_ms < 0 ? 0 : spec->lag_budget_ms);
  e->fast_finish = spec ? spec->fast_finish : 0;
  e->verbose = spec ? spec->verbose : 0;
  e->no_peek = spec ? spec->no_peek : 0;
  e->partial_every = spec && spec->partial_every > 0 ? spec->partial_every : 1;
  if (e->nl > MAXLAYERS) { hes__set_error("moonshine: %d layers, at most %d", e->nl, MAXLAYERS); goto fail; }

  e->vocab = vocab_load(vocab_bin);
  if (e->vocab.n <= 0) { hes__set_error("moonshine: cannot read vocab %s", vocab_bin); goto fail; }

  /* dec_embed.bin: [rows][dim] f32, mapped. The model's embedding is padded (EN: 32768 rows for a 32000-piece
   * table), so only the first vocab.n rows are used, as in the original. */
  e->dim = spec && spec->dim > 0 ? spec->dim : 320;
  {
    int fd = open(dec_embed_bin, O_RDONLY);
    off_t size = fd >= 0 ? lseek(fd, 0, SEEK_END) : -1;
    size_t per_row = (size_t)e->dim * sizeof(float);
    if (size <= 0 || (size_t)size % per_row || (size_t)size / per_row < (size_t)e->vocab.n) {
      if (fd >= 0) close(fd);
      hes__set_error("moonshine: %s (%lld bytes) is not [>= %d rows][%d] f32", dec_embed_bin, (long long)size,
                     e->vocab.n, e->dim);
      goto fail;
    }
    e->embed_len = (size_t)size;
    e->embed_map = mmap(NULL, e->embed_len, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (e->embed_map == MAP_FAILED) { e->embed_map = NULL; hes__set_error("moonshine: cannot map %s", dec_embed_bin); goto fail; }
    e->embed = (const float*)e->embed_map;
  }
  e->enc_dim = spec && spec->enc_dim > 0 ? spec->enc_dim : e->dim;

  {  /* the incremental frontend, by convention next to frontend.vmfb, on the frontend's device; with it the
      * whole-window frontend is never called, so it is not loaded */
    const char* slash = strrchr(vmfb_frontend, '/');
    char fesp[1024];
    snprintf(fesp, sizeof fesp, "%.*sfrontend_step.vmfb", slash ? (int)(slash - vmfb_frontend + 1) : 0, vmfb_frontend);
    if (access(fesp, R_OK) == 0 ? open_stage(e, &e->fes, dev_frontend, fesp, NULL)
                                : open_stage(e, &e->fe, dev_frontend, vmfb_frontend, NULL))
      goto fail;
  }
  if (open_stage(e, &e->enc, dev_encoder, vmfb_encoder, NULL) ||
      open_stage(e, &e->adp, dev_adapter, vmfb_adapter, NULL) ||
      open_stage(e, &e->pre, dev_decoder, vmfb_prefill, irpa) ||
      open_stage(e, &e->step, dev_decoder, vmfb_step, irpa))
    goto fail;
  {  /* the cross-K/V cache, when the encoder module computes the window rows' K/V */
    iree_vm_function_t fn;
    e->kv_cached = iree_status_consume_code(iree_runtime_session_lookup_function(e->enc.session, iree_make_cstring_view(FN_XKV), &fn)) == IREE_STATUS_OK;
  }

  e->rope_c = malloc((size_t)(MAXTOK + 4) * e->hd * sizeof(float));
  e->rope_s = malloc((size_t)(MAXTOK + 4) * e->hd * sizeof(float));
  e->fein = malloc(FE_IN * sizeof(float));
  e->feats = malloc((size_t)FE_OUT * e->enc_dim * sizeof(float));
  e->mem64 = malloc((size_t)CHUNK * e->dim * sizeof(float));
  e->mem256 = malloc((size_t)MAXMEM * e->dim * sizeof(float));
  e->mask = malloc(MAXMEM * sizeof(float));
  e->row = malloc((size_t)e->vocab.n * sizeof(float));
  e->pcm = malloc(MAXPCM * sizeof(float));
  e->mem = calloc((size_t)MAXMEM * e->dim, sizeof(float));
  e->kvwin = malloc((size_t)CHUNK * 2 * e->nl * e->dim * sizeof(float));        /* CHUNK x 2 x layers x dim */
  e->kvc = calloc((size_t)2 * e->nl * MAXMEM * e->dim, sizeof(float));          /* 2 x layers x heads x MAXMEM x hd */
  if (e->fes.session) {
    e->fesin = malloc(FES_IN * sizeof(float));
    e->fesout = malloc((size_t)FES_OUT * e->enc_dim * sizeof(float));
    e->fe_silence = calloc((size_t)e->enc_dim, sizeof(float));
    e->feat = malloc((size_t)MAXF * e->enc_dim * sizeof(float));
    if (!e->fesin || !e->fesout || !e->fe_silence || !e->feat) { hes__set_error("moonshine: out of memory"); goto fail; }
  }
  if (!e->rope_c || !e->rope_s || !e->fein || !e->feats || !e->mem64 || !e->mem256 || !e->mask || !e->row || !e->pcm || !e->mem ||
      !e->kvwin || !e->kvc) {
    hes__set_error("moonshine: out of memory");
    goto fail;
  }
  rope_init(e);
  if (!(spec && spec->skip_warmup) && warmup(e)) goto fail;  /* keeps the failing stage's message */
  return e;
fail:
  hes_ms_release(e);
  return NULL;
}

const char* hes_ms_feed(hes_ms* e, const float* pcm, int32_t n) {
  hes__clear_error();
  if (!e) return NULL;
  int take = n;
  if (e->npcm + take > MAXPCM) take = MAXPCM - e->npcm;
  if (take > 0) { memcpy(e->pcm + e->npcm, pcm, (size_t)take * sizeof(float)); e->npcm += take; }

  int produced = e->npcm / SPF;
  int changed = 0;
  fe_advance(e, produced);
  /* Behind the microphone by more than the lag budget: give up only the optional partial decodes; the windows still
   * run, so the final is unaffected (fast_finish keeps the incremental result, so there every hop counts). */
  if (e->t_first_feed == 0.0) e->t_first_feed = now_ms();
  int behind = 0;
  if (!e->fast_finish && e->lag_budget_ms > 0)
    behind = (now_ms() - e->t_first_feed) - (double)e->npcm * 1000.0 / 16000.0 > e->lag_budget_ms;

  /* Early peek: first words at ~0.72 s; window 0 recomputes the rows and the restart decode replaces the text. */
  if (!e->no_peek && !e->peeked && e->win == 0 && produced >= PEEK_FRAMES && produced < CHUNK) {
    double t0 = now_ms();
    int usable = produced - LOOKAHEAD > 0 ? produced - LOOKAHEAD : produced;
    if (process_window(e, 0, usable, 0, 1) == 0) {
      e->peeked = 1;
      release_self(e); e->ntoks = 0; e->pos = 0; e->text[0] = 0;
      if (run_prefill(e, 1) == 0) run_steps(e, STEP_SLICE, 1);
      e->pending = 2 * HOP_TOKENS - STEP_SLICE;
      if (e->ntoks > 0) changed = 1;
      note_evt(e, EVT_PEEK, t0 - e->t_first_feed, now_ms() - t0, 0, 0);
      tlog(e, "peek total %.0fms nmem %d toks %d \"%s\"\n", now_ms() - t0, e->nmem, e->ntoks, e->text);
    }
  }
  int hopped = 0;
  while (e->win * HOP + CHUNK <= produced && e->nmem < MAXMEM) {
    double t0 = now_ms();
    if (process_window(e, e->win * HOP, produced, 0, 0)) break;
    e->win++;
    int prev = e->ntoks;
    /* the first hops re-decode from scratch, later ones go incremental with the prefix locked */
    int restart = e->win <= RESTART_HOPS || e->ntoks == 0;
    /* Partials only every partial_every-th hop (the restart hops always run): the skipped hops cost a window,
     * not a decode. Like a dropped hop, this cannot change the final, which re-decodes from the memory. */
    int skip = !restart && e->partial_every > 1 && (e->win % e->partial_every) != 0 && !e->fast_finish;
    if (skip) {
      note_evt(e, EVT_DROPPED, t0 - e->t_first_feed, now_ms() - t0, 0, 0);
      hopped = 1;
      continue;
    }
    if (behind) {
      e->dropped++;
      note_evt(e, EVT_DROPPED, t0 - e->t_first_feed, now_ms() - t0, 0, 0);
      hopped = 1;
      tlog(e, "hop %d total %.0fms DROPPED decode (behind) nmem %d\n", e->win, now_ms() - t0, e->nmem);
      continue;
    }
    if (restart) { release_self(e); e->ntoks = 0; e->pos = 0; e->text[0] = 0; }
    if (run_prefill(e, restart) == 0) run_steps(e, STEP_SLICE, restart);
    e->pending = (restart ? 2 * HOP_TOKENS : HOP_TOKENS) - STEP_SLICE;
    e->hops++;
    note_evt(e, EVT_HOP, t0 - e->t_first_feed, now_ms() - t0, 0, 0);
    hopped = 1;
    if (e->ntoks != prev || e->win == 1) changed = 1;
    tlog(e, "hop %d total %.0fms toks %d \"%s\"\n", e->win, now_ms() - t0, e->ntoks, e->text);
  }
  if (!hopped && !behind && e->pending > 0 && !e->halted && e->ntoks > 0 && e->has_self && e->has_cross) {
    int budget = e->pending < STEP_SLICE ? e->pending : STEP_SLICE;
    int added = run_steps(e, budget, e->win <= RESTART_HOPS);
    e->pending -= added;
    if (added > 0) changed = 1;
  }
  /* surface a partial only once it is at least as long as the last one shown (restart hops rebuild the text) */
  size_t len = strlen(e->text);
  const char* p = e->text;
  while (*p == ' ') ++p;
  if (changed && *p && len >= e->shown_len) {
    e->shown_len = len;
    snprintf(e->out, sizeof e->out, "%s", p);
    return e->out;
  }
  return NULL;
}

const char* hes_ms_finish(hes_ms* e) {
  hes__clear_error();
  if (!e) return NULL;
  double t0 = now_ms();
  int produced = e->npcm / SPF;
  while (e->nmem < produced && e->nmem < MAXMEM) {  /* flush the remaining windows, lookahead released */
    if (process_window(e, e->win * HOP, produced, 1, 0)) break;
    e->win++;
  }
  double t_flush = now_ms();
  e->out[0] = 0;
  if (e->nmem > 0) {
    if (!e->fast_finish) {  /* exact: a fresh decode over the final memory */
      release_self(e); e->ntoks = 0; e->pos = 0; e->text[0] = 0;
      if (run_prefill(e, 1) == 0) run_steps(e, finish_budget(e->npcm), 1);
    }
    if (!e->ended_eos) drop_restart_suffix(e);
    /* command utterances are one sentence; on a silent tail the greedy decode restarts it instead of ending */
    for (char* q = e->text; *q; ++q) {
      if (*q == '.' || *q == '!' || *q == '?') { q[1] = 0; break; }
    }
    const char* p = e->text;
    while (*p == ' ') ++p;
    snprintf(e->out, sizeof e->out, "%s", p);
    tlog(e, "finish %.0fms (flush %.0fms decode %.0fms) nmem %d toks %d/%d dropped %d festep %d/%.0fms \"%s\"\n",
         now_ms() - t0, t_flush - t0, now_ms() - t_flush, e->nmem, e->ntoks, finish_budget(e->npcm), e->dropped,
         e->fe_calls, e->fe_feed_ms, e->out);
  }
  int errors = e->errors;
  int n = snprintf(e->stats, sizeof e->stats,
                   "flushMs=%.0f,decodeMs=%.0f,finishMs=%.0f,finishAtMs=%.0f,lagMs=%.0f,droppedDecodes=%d,"
                   "windows=%d,hops=%d,memFrames=%d,tokens=%d,tokenBudget=%d,audioMs=%.0f,fastFinish=%d,lagBudgetMs=%d,"
                   "errors=%d,prefills=%d,prefillMs=%.0f,steps=%d,stepMs=%.0f,feStepCalls=%d,feStepMs=%.0f",
                   t_flush - t0, now_ms() - t_flush, now_ms() - t0, e->t_first_feed > 0.0 ? t0 - e->t_first_feed : 0.0,
                   e->t_first_feed > 0.0 ? t0 - e->t_first_feed - (double)e->npcm * 1000.0 / 16000.0 : 0.0, e->dropped,
                   e->windows, e->hops, e->nmem, e->ntoks, finish_budget(e->npcm), (double)e->npcm * 1000.0 / 16000.0,
                   e->fast_finish, e->lag_budget_ms, errors, e->prefills, e->prefill_ms, e->steps, e->step_ms,
                   e->fe_calls, e->fe_feed_ms);
  /* timeline=<kind>:<atMs>:<a>:<b>:<c>|... (w window: frontend/encoder/adapter, p peek, h hop, d dropped hop) */
  if (e->nevt > 0 && n > 0 && (size_t)n < sizeof e->stats) {
    n += snprintf(e->stats + n, sizeof e->stats - n, ",timeline=");
    for (int i = 0; i < e->nevt && n > 0 && (size_t)n < sizeof e->stats; ++i) {
      const char k = e->evt_kind[i] == EVT_WINDOW ? 'w' : e->evt_kind[i] == EVT_PEEK ? 'p' : e->evt_kind[i] == EVT_HOP ? 'h' : 'd';
      n += snprintf(e->stats + n, sizeof e->stats - n, "%s%c:%d:%d:%d:%d", i ? "|" : "", k, e->evt_at[i], e->evt_a[i],
                    e->evt_b[i], e->evt_c[i]);
    }
  }
  reset_utterance(e);
  if (errors && !e->out[0]) return NULL;  /* nothing recognized because a stage failed: hes_last_error() says which */
  return e->out;
}

void hes_ms_reset(hes_ms* e) { if (e) reset_utterance(e); }

const char* hes_ms_stats(hes_ms* e) { return e ? e->stats : ""; }

int32_t hes_ms_dim(hes_ms* e) { return e ? e->dim : 0; }
int32_t hes_ms_vocab(hes_ms* e) { return e ? e->vocab.n : 0; }
