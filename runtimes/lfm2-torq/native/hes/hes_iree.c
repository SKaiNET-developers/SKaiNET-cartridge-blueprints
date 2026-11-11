/*
 * hes_iree: see hes_iree.h. Generic sessions plus the KV-cache decoder.
 *
 * The Moonshine streaming session lives in hes_moonshine.c. The KV part is a port of SKaiNET-transformers' Android session (llm-runtime/iree-android/native/iree_kv_jni.c):
 * same graph contract (functiongemma-kv-v1 / qwen-kv-v1, host-gather embedding), same cache handling, same
 * snapshots. Differences: plain C entry points instead of JNI, errors in a thread-local message, and devices
 * shared through one hes_runtime instead of an instance per session.
 *
 * Contract summary (details in the original):
 *   prefill  : *_prefill_at(tokens SEQ i32, emb SEQx{hidden} f32, select 1xSEQ f32) -> per-layer K,V ..., token
 *   chunk    : *_prefill_with_past(tokens C, emb, per-base cos/sin [C,headDim], per-layer K,V, masks, select 1xC)
 *   withPast : *_with_past(token 1, emb 1x{hidden}, per-base cos/sin [1,headDim], per-layer K,V)
 * Argument order is the tracer's first-use order; sliding layers only ever see their last `window` rows.
 */
#define _GNU_SOURCE  /* strdup, mmap, sysconf */
#include "hes_iree_internal.h"

#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "iree/base/internal/flags.h"
#include "iree/io/file_handle.h"
#include "iree/io/formats/irpa/irpa_parser.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/modules/io/parameters/module.h"
#include "iree/runtime/api.h"

_Static_assert(HES_F32 == IREE_HAL_ELEMENT_TYPE_FLOAT_32, "HES_F32");
_Static_assert(HES_F16 == IREE_HAL_ELEMENT_TYPE_FLOAT_16, "HES_F16");
_Static_assert(HES_BF16 == IREE_HAL_ELEMENT_TYPE_BFLOAT_16, "HES_BF16");
_Static_assert(HES_I32 == IREE_HAL_ELEMENT_TYPE_INT_32, "HES_I32");
_Static_assert(HES_I64 == IREE_HAL_ELEMENT_TYPE_INT_64, "HES_I64");
_Static_assert(HES_I8 == IREE_HAL_ELEMENT_TYPE_INT_8, "HES_I8");

#define DEFAULT_SCOPE "model"
#define MAX_CONCURRENT_PARAM_OPS 16
#define MASK_NEG (-1.0e30f)
#define MAX_DEVICES 4

/* ---------- errors ---------- */

static _Thread_local char g_error[1024];

const char* hes_last_error(void) { return g_error; }

void hes__clear_error(void) { g_error[0] = 0; }

void hes__set_error(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(g_error, sizeof g_error, fmt, ap);
  va_end(ap);
}

/* Records `st` (consumed) under `what`; returns 0 when ok, -1 otherwise. */
int hes__check(iree_status_t st, const char* what) {
  if (iree_status_is_ok(st)) return 0;
  char* buf = NULL;
  iree_host_size_t len = 0;
  iree_allocator_t alloc = iree_allocator_system();
  if (iree_status_to_string(st, &alloc, &buf, &len) && buf) {
    hes__set_error("%s: %.*s", what, (int)(len < 900 ? len : 900), buf);
    iree_allocator_free(alloc, buf);
  } else {
    hes__set_error("%s: status code %d", what, (int)iree_status_code(st));
  }
  iree_status_ignore(st);
  return -1;
}

/* ---------- runtime ---------- */

struct hes_runtime {
  iree_runtime_instance_t* instance;
  struct { char* uri; iree_hal_device_t* device; } devices[MAX_DEVICES];
  int device_count;
  char drivers[256];
};

static int g_flags_parsed = 0;

hes_runtime* hes_runtime_create(const char* const* flags, int flag_count) {
  g_error[0] = 0;
  if (!g_flags_parsed && flags && flag_count > 0) {
    /* IREE flags are process-global and parsed from an argv; argv[0] is skipped. String flags keep a pointer into
     * argv (e.g. FLAG_torq_hw_type), so the strings are copied and intentionally never freed. */
    int argc = flag_count + 1;
    char** argv = calloc((size_t)argc + 1, sizeof(char*));
    argv[0] = "hes";
    for (int i = 0; i < flag_count; ++i) argv[i + 1] = strdup(flags[i]);
    char** parse_argv = argv;
    iree_status_t st = iree_flags_parse(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &parse_argv);
    /* argv itself (the pointer array) may go; the strings it pointed to stay alive. */
    free(argv);
    if (hes__check(st, "flags")) return NULL;
    g_flags_parsed = 1;
  }
  hes_runtime* rt = calloc(1, sizeof *rt);
  iree_runtime_instance_options_t options;
  iree_runtime_instance_options_initialize(&options);
  iree_runtime_instance_options_use_all_available_drivers(&options);
  if (hes__check(iree_runtime_instance_create(&options, iree_allocator_system(), &rt->instance), "instance")) {
    free(rt);
    return NULL;
  }
  iree_hal_driver_registry_t* registry = iree_runtime_instance_driver_registry(rt->instance);
  iree_host_size_t count = 0;
  iree_hal_driver_info_t* infos = NULL;
  if (iree_status_is_ok(iree_hal_driver_registry_enumerate(registry, iree_allocator_system(), &count, &infos))) {
    size_t used = 0;
    for (iree_host_size_t i = 0; i < count && used + 1 < sizeof rt->drivers; ++i) {
      int w = snprintf(rt->drivers + used, sizeof rt->drivers - used, "%s%.*s", i ? " " : "",
                       (int)infos[i].driver_name.size, infos[i].driver_name.data);
      if (w > 0) used += (size_t)w;
    }
    iree_allocator_free(iree_allocator_system(), infos);
  }
  return rt;
}

const char* hes_runtime_drivers(hes_runtime* rt) { return rt ? rt->drivers : ""; }

void hes_runtime_release(hes_runtime* rt) {
  if (!rt) return;
  for (int i = 0; i < rt->device_count; ++i) {
    iree_hal_device_release(rt->devices[i].device);
    free(rt->devices[i].uri);
  }
  iree_runtime_instance_release(rt->instance);
  free(rt);
}

/* The shared device for `uri`, created on first use. Borrowed: the runtime owns it. */
iree_hal_device_t* hes__device_for(hes_runtime* rt, const char* uri) {
  for (int i = 0; i < rt->device_count; ++i)
    if (strcmp(rt->devices[i].uri, uri) == 0) return rt->devices[i].device;
  if (rt->device_count == MAX_DEVICES) { hes__set_error("device %s: too many devices", uri); return NULL; }
  iree_hal_device_t* device = NULL;
  char what[128];
  snprintf(what, sizeof what, "device %s", uri);
  if (hes__check(iree_runtime_instance_try_create_default_device(rt->instance, iree_make_cstring_view(uri), &device), what))
    return NULL;
  rt->devices[rt->device_count].uri = strdup(uri);
  rt->devices[rt->device_count].device = device;
  rt->device_count++;
  return device;
}

/* ---------- sessions ---------- */

static void unmap_file(void* user_data, iree_io_file_handle_primitive_t primitive) {
  (void)user_data;
  munmap(primitive.value.host_allocation.data, primitive.value.host_allocation.data_length);
}

/*
 * Opens `path` as a read-only memory mapping wrapped in a host-allocation file handle. Parameters in such a file
 * are imported by the CPU device without a copy, so the weights stay clean, evictable page cache instead of
 * anonymous heap: a 0.5 GB archive per graph would otherwise exhaust a 2 GB board.
 */
iree_status_t hes__open_mapped(const char* path, iree_io_file_handle_t** out) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) return iree_make_status(IREE_STATUS_NOT_FOUND, "cannot open %s", path);
  off_t size = lseek(fd, 0, SEEK_END);
  void* map = size > 0 ? mmap(NULL, (size_t)size, PROT_READ, MAP_PRIVATE, fd, 0) : MAP_FAILED;
  close(fd);
  if (map == MAP_FAILED) return iree_make_status(IREE_STATUS_INTERNAL, "cannot map %s", path);
  iree_io_file_handle_release_callback_t release = {.fn = unmap_file, .user_data = NULL};
  iree_status_t st = iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ, iree_make_byte_span(map, (iree_host_size_t)size), release, iree_allocator_system(), out);
  if (!iree_status_is_ok(st)) munmap(map, (size_t)size);
  return st;
}

/*
 * New IREE session on `device`; with `irpa` its parameters are appended first (scope `scope`). On success
 * `*out_index` (when not NULL) receives the parameter index, retained, for callers that read the archive.
 */
iree_status_t hes__open_session(hes_runtime* rt, iree_hal_device_t* device, const char* vmfb, const char* irpa,
                                  const char* scope, hes_session* s, iree_io_parameter_index_t** out_index) {
  iree_allocator_t alloc = iree_allocator_system();
  s->device = device;
  iree_runtime_session_options_t options;
  iree_runtime_session_options_initialize(&options);
  IREE_RETURN_IF_ERROR(iree_runtime_session_create_with_device(
      rt->instance, &options, device, iree_runtime_instance_host_allocator(rt->instance), &s->session));
  if (irpa) {
    iree_io_file_handle_t* file = NULL;
    IREE_RETURN_IF_ERROR(hes__open_mapped(irpa, &file));
    iree_io_parameter_index_t* index = NULL;
    iree_status_t st = iree_io_parameter_index_create(alloc, &index);
    if (iree_status_is_ok(st)) st = iree_io_parse_irpa_index(file, index, alloc);
    iree_io_file_handle_release(file);
    if (iree_status_is_ok(st))
      st = iree_io_parameter_index_provider_create(iree_make_cstring_view(scope ? scope : DEFAULT_SCOPE), index,
                                                   MAX_CONCURRENT_PARAM_OPS, alloc, &s->provider);
    iree_vm_module_t* module = NULL;
    if (iree_status_is_ok(st))
      st = iree_io_parameters_module_create(iree_runtime_instance_vm_instance(rt->instance), 1, &s->provider, alloc, &module);
    if (iree_status_is_ok(st)) st = iree_runtime_session_append_module(s->session, module);
    if (module) iree_vm_module_release(module);
    if (iree_status_is_ok(st) && out_index) { *out_index = index; index = NULL; }
    if (index) iree_io_parameter_index_release(index);
    IREE_RETURN_IF_ERROR(st);
  }
  return iree_runtime_session_append_bytecode_module_from_file(s->session, vmfb);
}

void hes__close_session(hes_session* s) {
  if (s->session) iree_runtime_session_release(s->session);
  if (s->provider) iree_io_parameter_provider_release(s->provider);
  memset(s, 0, sizeof *s);
}

hes_session* hes_session_create(hes_runtime* rt, const char* device, const char* vmfb, const char* irpa, const char* scope) {
  g_error[0] = 0;
  iree_hal_device_t* dev = hes__device_for(rt, device);
  if (!dev) return NULL;
  hes_session* s = calloc(1, sizeof *s);
  char what[512];
  snprintf(what, sizeof what, "session %s on %s", vmfb, device);
  if (hes__check(hes__open_session(rt, dev, vmfb, irpa, scope, s, NULL), what)) {
    hes__close_session(s);
    free(s);
    return NULL;
  }
  return s;
}

void hes_session_release(hes_session* s) {
  if (!s) return;
  hes__close_session(s);
  free(s);
}

/* ---------- views ---------- */

/* hes_view is IREE's buffer view; the alias keeps the header free of IREE types. */
#define VIEW(v) ((iree_hal_buffer_view_t*)(v))

iree_hal_buffer_params_t hes__device_params(void) {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
  return params;
}

iree_status_t hes__view_copy(iree_hal_device_t* device, iree_hal_element_type_t type, int rank,
                               const iree_hal_dim_t* dims, const void* data, size_t bytes, iree_hal_buffer_view_t** out) {
  return iree_hal_buffer_view_allocate_buffer_copy(device, iree_hal_device_allocator(device), (iree_host_size_t)rank,
                                                    dims, type, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, hes__device_params(),
                                                    iree_make_const_byte_span(data, bytes), out);
}

hes_view* hes_view_from_host(hes_session* s, int32_t element_type, int rank, const int64_t* dims, const void* data, size_t bytes) {
  g_error[0] = 0;
  if (rank < 0 || rank > 8) { hes__set_error("view: rank %d out of range", rank); return NULL; }
  iree_hal_dim_t d[8];
  size_t count = 1;
  for (int i = 0; i < rank; ++i) { d[i] = (iree_hal_dim_t)dims[i]; count *= (size_t)dims[i]; }
  size_t want = count * iree_hal_element_dense_byte_count((iree_hal_element_type_t)element_type);
  if (want != bytes) { hes__set_error("view: %zu bytes given, shape needs %zu", bytes, want); return NULL; }
  iree_hal_buffer_view_t* v = NULL;
  if (hes__check(hes__view_copy(s->device, (iree_hal_element_type_t)element_type, rank, d, data, bytes, &v), "view")) return NULL;
  return (hes_view*)v;
}

int hes_view_to_host(hes_session* s, hes_view* v, void* dst, size_t bytes) {
  g_error[0] = 0;
  size_t have = (size_t)iree_hal_buffer_view_byte_length(VIEW(v));
  if (have != bytes) { hes__set_error("to_host: view has %zu bytes, %zu requested", have, bytes); return -1; }
  return hes__check(iree_hal_device_transfer_d2h(s->device, iree_hal_buffer_view_buffer(VIEW(v)), 0, dst, bytes,
                                            IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()), "to_host");
}

int hes_view_rank(hes_view* v) { return (int)iree_hal_buffer_view_shape_rank(VIEW(v)); }
int64_t hes_view_dim(hes_view* v, int axis) { return (int64_t)iree_hal_buffer_view_shape_dim(VIEW(v), (iree_host_size_t)axis); }
int32_t hes_view_element_type(hes_view* v) { return (int32_t)iree_hal_buffer_view_element_type(VIEW(v)); }
size_t hes_view_bytes(hes_view* v) { return (size_t)iree_hal_buffer_view_byte_length(VIEW(v)); }
void hes_view_retain(hes_view* v) { if (v) iree_hal_buffer_view_retain(VIEW(v)); }
void hes_view_release(hes_view* v) { if (v) iree_hal_buffer_view_release(VIEW(v)); }

/* ---------- calls ---------- */

static iree_status_t invoke(iree_runtime_session_t* session, const char* fn, iree_hal_buffer_view_t* const* ins, int n_in,
                            iree_hal_buffer_view_t** outs, int out_capacity, int* out_count) {
  iree_runtime_call_t call;
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(session, iree_make_cstring_view(fn), &call));
  iree_status_t st = iree_ok_status();
  for (int i = 0; i < n_in && iree_status_is_ok(st); ++i) st = iree_runtime_call_inputs_push_back_buffer_view(&call, ins[i]);
  if (iree_status_is_ok(st)) st = iree_runtime_call_invoke(&call, 0);
  int n = 0;
  if (iree_status_is_ok(st)) {
    iree_vm_list_t* results = iree_runtime_call_outputs(&call);
    iree_host_size_t produced = iree_vm_list_size(results);
    if ((int)produced > out_capacity)
      st = iree_make_status(IREE_STATUS_OUT_OF_RANGE, "%s produced %d outputs, capacity %d", fn, (int)produced, out_capacity);
    for (; iree_status_is_ok(st) && n < (int)produced; ++n) st = iree_runtime_call_outputs_pop_front_buffer_view(&call, &outs[n]);
    if (!iree_status_is_ok(st)) for (int i = 0; i < n; ++i) { iree_hal_buffer_view_release(outs[i]); outs[i] = NULL; }
  }
  iree_runtime_call_deinitialize(&call);
  if (out_count) *out_count = iree_status_is_ok(st) ? n : 0;
  return st;
}

int hes_call(hes_session* s, const char* function, hes_view* const* inputs, int input_count, hes_view** outputs, int output_capacity) {
  g_error[0] = 0;
  int n = 0;
  char what[256];
  snprintf(what, sizeof what, "call %s", function);
  if (hes__check(invoke(s->session, function, (iree_hal_buffer_view_t* const*)inputs, input_count,
                   (iree_hal_buffer_view_t**)outputs, output_capacity, &n), what))
    return -1;
  return n;
}

/* ---------- KV-cache decoder ---------- */

typedef struct {
  hes_session s;     /* s.session NULL = absent */
  int alias;         /* 1 = s.session is retained from another graph and s.provider is not owned */
  char* fn;
} Graph;

struct hes_kv {
  hes_runtime* rt;
  iree_hal_device_t* dev;
  Graph withPast, chunk, prefill;
  int hasPrefill;
  int nLayers, headDim, nKV, nHeads, hidden, vocab, window, period, chunkC, maskH;
  float baseS, baseG;
  iree_hal_buffer_view_t** kv; /* 2*nLayers, K then V: [1, nKV, len[l], headDim] f32 */
  int* len;
  int pos;
  uint8_t* embMap; size_t embMapLen; size_t embOff;
  uint8_t* sclMap; size_t sclMapLen; size_t sclOff; /* int8 embedding: its f32 row scales; NULL for bf16 */
};

struct hes_kv_snapshot {
  iree_hal_buffer_view_t** kv;
  int* len;
  int pos;
  int n;
};

static void graph_close(Graph* g) {
  if (g->alias) { if (g->s.session) iree_runtime_session_release(g->s.session); memset(&g->s, 0, sizeof g->s); }
  else hes__close_session(&g->s);
  free(g->fn);
  memset(g, 0, sizeof *g);
}

static void graph_alias(Graph* dst, const Graph* src, const char* fn) {
  dst->s = src->s;
  dst->s.provider = NULL;
  iree_runtime_session_retain(dst->s.session);
  dst->alias = 1;
  dst->fn = strdup(fn);
}

/* mmaps one file-backed entry of the archive, page-aligned; *out points at its first byte. */
static int map_entry(const char* irpa, const iree_io_parameter_index_entry_t* e, uint8_t** map, size_t* mapLen, size_t* off) {
  if (e->type != IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE) return 0;
  size_t at = (size_t)e->storage.file.offset;
  int fd = open(irpa, O_RDONLY);
  if (fd < 0) return 0;
  long pageSizeL = sysconf(_SC_PAGESIZE);
  size_t pageSize = pageSizeL > 0 ? (size_t)pageSizeL : 4096;
  size_t alignedStart = (at / pageSize) * pageSize;
  size_t alignedLen = (at - alignedStart) + (size_t)e->length;
  alignedLen = ((alignedLen + pageSize - 1) / pageSize) * pageSize;
  void* m = mmap(NULL, alignedLen, PROT_READ, MAP_PRIVATE, fd, (off_t)alignedStart);
  close(fd);
  if (m == MAP_FAILED) return 0;
  *map = m; *mapLen = alignedLen; *off = at - alignedStart;
  return 1;
}

/*
 * mmaps the embedding entry of the archive (host gather): vocab x hidden bf16, or vocab x hidden int8 `<key>_q`
 * with its vocab f32 row scales `<key>_s` (archives of int8 graphs).
 */
static int locate_embedding(hes_kv* k, const char* irpa, iree_io_parameter_index_t* index) {
  uint64_t rows = (uint64_t)k->vocab * (uint64_t)k->hidden;
  iree_host_size_t n = iree_io_parameter_index_count(index);
  for (iree_host_size_t i = 0; i < n; ++i) {
    const iree_io_parameter_index_entry_t* e = NULL;
    if (!iree_status_is_ok(iree_io_parameter_index_get(index, i, &e)) || !e) continue;
    if (e->length == rows * 2u) return map_entry(irpa, e, &k->embMap, &k->embMapLen, &k->embOff);
  }
  for (iree_host_size_t i = 0; i < n; ++i) {
    const iree_io_parameter_index_entry_t* e = NULL;
    if (!iree_status_is_ok(iree_io_parameter_index_get(index, i, &e)) || !e) continue;
    if (e->length != rows || e->key.size < 3 || memcmp(e->key.data + e->key.size - 2, "_q", 2) != 0) continue;
    char key[256];
    if (e->key.size >= sizeof key) continue;
    memcpy(key, e->key.data, e->key.size); key[e->key.size] = 0;
    key[e->key.size - 1] = 's';
    const iree_io_parameter_index_entry_t* sc = NULL;
    if (!iree_status_is_ok(iree_io_parameter_index_lookup(index, iree_make_cstring_view(key), &sc)) || !sc) continue;
    if (sc->length != (uint64_t)k->vocab * 4u) continue;
    return map_entry(irpa, e, &k->embMap, &k->embMapLen, &k->embOff) &&
           map_entry(irpa, sc, &k->sclMap, &k->sclMapLen, &k->sclOff);
  }
  return 0;
}

static void embed_rows(const hes_kv* k, const int32_t* toks, int n, int rows, float* out) {
  memset(out, 0, sizeof(float) * (size_t)rows * (size_t)k->hidden);
  for (int i = 0; i < n; ++i) {
    int32_t t = toks[i];
    if (t < 0 || t >= k->vocab) t = 0;
    float* dst = out + (size_t)i * (size_t)k->hidden;
    if (k->sclMap) {
      const int8_t* src = (const int8_t*)(k->embMap + k->embOff + (size_t)t * (size_t)k->hidden);
      float s;
      memcpy(&s, k->sclMap + k->sclOff + (size_t)t * 4u, 4);
      for (int j = 0; j < k->hidden; ++j) dst[j] = (float)src[j] * s;
    } else {
      const uint16_t* src = (const uint16_t*)(k->embMap + k->embOff + (size_t)t * (size_t)k->hidden * 2u);
      for (int j = 0; j < k->hidden; ++j) { uint32_t b = ((uint32_t)src[j]) << 16; memcpy(&dst[j], &b, 4); }
    }
  }
}

static iree_status_t view_f32(hes_kv* k, const float* data, int rank, const iree_hal_dim_t* dims, iree_hal_buffer_view_t** out) {
  size_t count = 1;
  for (int i = 0; i < rank; ++i) count *= (size_t)dims[i];
  return hes__view_copy(k->dev, IREE_HAL_ELEMENT_TYPE_FLOAT_32, rank, dims, data, count * 4, out);
}
static iree_status_t view_i32(hes_kv* k, const int32_t* data, int rank, const iree_hal_dim_t* dims, iree_hal_buffer_view_t** out) {
  size_t count = 1;
  for (int i = 0; i < rank; ++i) count *= (size_t)dims[i];
  return hes__view_copy(k->dev, IREE_HAL_ELEMENT_TYPE_INT_32, rank, dims, data, count * 4, out);
}

/* [1, nKV, rows, headDim] view over rows [rowStart, rowStart+rows): a zero-copy subspan for nKV == 1, a gather otherwise. */
static iree_status_t sub_rows(hes_kv* k, iree_hal_buffer_view_t* v, int rowStart, int rows, iree_hal_buffer_view_t** out) {
  if (k->nKV == 1) {
    iree_device_size_t rowBytes = (iree_device_size_t)k->headDim * 4;
    iree_hal_buffer_t* sub = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_subspan(iree_hal_buffer_view_buffer(v), (iree_device_size_t)rowStart * rowBytes,
                                                 (iree_device_size_t)rows * rowBytes, iree_allocator_system(), &sub));
    iree_hal_dim_t dims[4] = {1, 1, (iree_hal_dim_t)rows, (iree_hal_dim_t)k->headDim};
    iree_status_t st = iree_hal_buffer_view_create(sub, 4, dims, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                                                   IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, iree_allocator_system(), out);
    iree_hal_buffer_release(sub);
    return st;
  }
  iree_hal_dim_t srcRows = iree_hal_buffer_view_shape_dim(v, 2);
  iree_device_size_t planeBytes = (iree_device_size_t)k->headDim * 4;
  iree_device_size_t rowBytesOut = (iree_device_size_t)rows * planeBytes;
  iree_hal_buffer_t* dst = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(k->dev), hes__device_params(),
                                                          (iree_device_size_t)k->nKV * rowBytesOut, &dst));
  iree_hal_buffer_t* src = iree_hal_buffer_view_buffer(v);
  iree_status_t st = iree_ok_status();
  for (int h = 0; h < k->nKV && iree_status_is_ok(st); ++h) {
    iree_device_size_t srcOff = ((iree_device_size_t)h * (iree_device_size_t)srcRows + (iree_device_size_t)rowStart) * planeBytes;
    st = iree_hal_device_transfer_d2d(k->dev, src, srcOff, dst, (iree_device_size_t)h * rowBytesOut, rowBytesOut,
                                      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }
  if (!iree_status_is_ok(st)) { iree_hal_buffer_release(dst); return st; }
  iree_hal_dim_t dims[4] = {1, (iree_hal_dim_t)k->nKV, (iree_hal_dim_t)rows, (iree_hal_dim_t)k->headDim};
  st = iree_hal_buffer_view_create(dst, 4, dims, IREE_HAL_ELEMENT_TYPE_FLOAT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                   iree_allocator_system(), out);
  iree_hal_buffer_release(dst);
  return st;
}

static iree_status_t read_i32(hes_kv* k, iree_hal_buffer_view_t* v, int32_t* out) {
  return iree_hal_device_transfer_d2h(k->dev, iree_hal_buffer_view_buffer(v), 0, out, 4,
                                      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
}

/* Split-half RoPE tables, sign folded into the first half. */
static void rope_rows(const hes_kv* k, int pos0, int rows, float base, float* cos_out, float* sin_out) {
  int hd = k->headDim, half = hd / 2;
  for (int r = 0; r < rows; ++r) {
    double p = (double)(pos0 + r);
    for (int i = 0; i < half; ++i) {
      double a = p * (1.0 / pow((double)base, (2.0 * i) / hd));
      float c = (float)cos(a), s = (float)sin(a);
      cos_out[r * hd + i] = c; cos_out[r * hd + half + i] = c;
      sin_out[r * hd + i] = -s; sin_out[r * hd + half + i] = s;
    }
  }
}

/* Additive mask [1, maskH, C, past+C]; rows never depend on the head. */
static void chunk_mask(const hes_kv* k, int past, int nReal, int window, float* out) {
  int C = k->chunkC, K = past + C;
  for (int h = 0; h < k->maskH; ++h) {
    for (int i = 0; i < C; ++i) {
      float* row = out + ((size_t)h * C + i) * K;
      int a = k->pos + i;
      for (int j = 0; j < K; ++j) {
        int absKey = (j < past) ? (k->pos - past + j) : (k->pos + (j - past));
        int ok = (i < nReal) ? (absKey <= a && (window <= 0 || absKey >= a - window + 1)) : (j == past + i);
        row[j] = ok ? 0.0f : MASK_NEG;
      }
    }
  }
}

static int is_global(const hes_kv* k, int l) { return (l % k->period) == k->period - 1; }

static void release_views(iree_hal_buffer_view_t** v, int n) {
  for (int i = 0; i < n; ++i) if (v[i]) { iree_hal_buffer_view_release(v[i]); v[i] = NULL; }
}

static iree_status_t cache_inputs(hes_kv* k, iree_hal_buffer_view_t** outK, iree_hal_buffer_view_t** outV, int* pastS, int* pastG) {
  *pastS = 0; *pastG = 0;
  for (int l = 0; l < k->nLayers; ++l) {
    int len = k->len[l];
    int keep = is_global(k, l) ? len : (len > k->window ? k->window : len);
    if (is_global(k, l)) *pastG = keep; else *pastS = keep;
    if (keep == len) {
      outK[l] = k->kv[2 * l]; iree_hal_buffer_view_retain(outK[l]);
      outV[l] = k->kv[2 * l + 1]; iree_hal_buffer_view_retain(outV[l]);
    } else {
      IREE_RETURN_IF_ERROR(sub_rows(k, k->kv[2 * l], len - keep, keep, &outK[l]));
      IREE_RETURN_IF_ERROR(sub_rows(k, k->kv[2 * l + 1], len - keep, keep, &outV[l]));
    }
  }
  return iree_ok_status();
}

static iree_status_t adopt_outputs(hes_kv* k, iree_hal_buffer_view_t** outs, int addedRows, int realRows, const int* pastPerLayer) {
  for (int l = 0; l < k->nLayers; ++l) {
    for (int kv = 0; kv < 2; ++kv) {
      iree_hal_buffer_view_t* full = outs[2 * l + kv];
      iree_hal_buffer_view_t* keep = NULL;
      if (realRows == addedRows) { keep = full; iree_hal_buffer_view_retain(keep); }
      else { IREE_RETURN_IF_ERROR(sub_rows(k, full, 0, pastPerLayer[l] + realRows, &keep)); }
      if (k->kv[2 * l + kv]) iree_hal_buffer_view_release(k->kv[2 * l + kv]);
      k->kv[2 * l + kv] = keep;
    }
    k->len[l] = pastPerLayer[l] + realRows;
  }
  return iree_ok_status();
}

static iree_status_t invoke_graph(Graph* g, iree_hal_buffer_view_t** ins, int nIn, iree_hal_buffer_view_t** outs, int nOut) {
  int produced = 0;
  IREE_RETURN_IF_ERROR(invoke(g->s.session, g->fn, ins, nIn, outs, nOut, &produced));
  if (produced != nOut) return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "%s produced %d outputs, expected %d", g->fn, produced, nOut);
  return iree_ok_status();
}

void hes_kv_release(hes_kv* k) {
  if (!k) return;
  if (k->kv) release_views(k->kv, 2 * k->nLayers);
  graph_close(&k->prefill); graph_close(&k->chunk); graph_close(&k->withPast);
  if (k->embMap) munmap(k->embMap, k->embMapLen);
  if (k->sclMap) munmap(k->sclMap, k->sclMapLen);
  free(k->kv); free(k->len); free(k);
}

hes_kv* hes_kv_create(hes_runtime* rt, const char* device, const hes_kv_spec* spec,
                      const char* vWP, const char* iWP, const char* fWP,
                      const char* vCH, const char* iCH, const char* fCH,
                      const char* vPF, const char* iPF, const char* fPF) {
  g_error[0] = 0;
  if (!fWP) fWP = "module.gemma_with_past";
  if (!fCH) fCH = "module.gemma_prefill_with_past";
  if (!fPF) fPF = "module.gemma_prefill_at";
  if (spec->n_layers <= 0 || spec->head_dim <= 0 || spec->n_kv_heads <= 0 || spec->n_heads <= 0 ||
      spec->n_heads % spec->n_kv_heads != 0 || spec->hidden_size <= 0 || spec->vocab_size <= 0 ||
      spec->global_layer_period <= 0 || spec->chunk <= 0 || spec->head_dim > 1024) {
    hes__set_error("kv: bad spec (sizes > 0, n_heads a multiple of n_kv_heads, head_dim <= 1024)");
    return NULL;
  }
  int maskH = spec->mask_heads <= 0 ? spec->n_heads : spec->mask_heads;
  if (maskH != 1 && maskH != spec->n_heads) { hes__set_error("kv: bad spec (mask_heads must be 0, 1 or n_heads)"); return NULL; }
  iree_hal_device_t* dev = hes__device_for(rt, device);
  if (!dev) return NULL;

  hes_kv* k = calloc(1, sizeof *k);
  k->rt = rt; k->dev = dev;
  k->nLayers = spec->n_layers; k->headDim = spec->head_dim; k->nKV = spec->n_kv_heads; k->nHeads = spec->n_heads;
  k->hidden = spec->hidden_size; k->vocab = spec->vocab_size; k->window = spec->sliding_window;
  k->period = spec->global_layer_period; k->chunkC = spec->chunk; k->maskH = maskH;
  k->baseS = spec->sliding_rope_base; k->baseG = spec->global_rope_base;
  k->kv = calloc((size_t)2 * k->nLayers, sizeof(void*));
  k->len = calloc((size_t)k->nLayers, sizeof(int));

  int shared = vPF && iPF && strcmp(vWP, vCH) == 0 && strcmp(iWP, iCH) == 0 && strcmp(vWP, vPF) == 0 && strcmp(iWP, iPF) == 0;
  iree_io_parameter_index_t* wpIndex = NULL;
  iree_status_t st = hes__open_session(rt, dev, vWP, iWP, DEFAULT_SCOPE, &k->withPast.s, &wpIndex);
  if (iree_status_is_ok(st)) k->withPast.fn = strdup(fWP);
  if (iree_status_is_ok(st) && !locate_embedding(k, iWP, wpIndex))
    st = iree_make_status(IREE_STATUS_NOT_FOUND, "no vocab x hidden embedding entry (bf16, or int8 + row scales) in the with-past archive");
  if (wpIndex) iree_io_parameter_index_release(wpIndex);
  if (shared) {
    if (iree_status_is_ok(st)) { graph_alias(&k->chunk, &k->withPast, fCH); graph_alias(&k->prefill, &k->withPast, fPF); k->hasPrefill = 1; }
  } else {
    if (iree_status_is_ok(st)) { st = hes__open_session(rt, dev, vCH, iCH, DEFAULT_SCOPE, &k->chunk.s, NULL); if (iree_status_is_ok(st)) k->chunk.fn = strdup(fCH); }
    if (iree_status_is_ok(st) && vPF && iPF) {
      st = hes__open_session(rt, dev, vPF, iPF, DEFAULT_SCOPE, &k->prefill.s, NULL);
      if (iree_status_is_ok(st)) { k->prefill.fn = strdup(fPF); k->hasPrefill = 1; }
    }
  }
  if (hes__check(st, "kv create")) { hes_kv_release(k); return NULL; }
  return k;
}

void hes_kv_release_prefill(hes_kv* k) {
  if (!k || !k->hasPrefill) return;
  graph_close(&k->prefill);
  k->hasPrefill = 0;
}

int32_t hes_kv_position(hes_kv* k) { return k ? k->pos : -1; }

int32_t hes_kv_prefill(hes_kv* k, const int32_t* toks, int32_t seq, int32_t n) {
  g_error[0] = 0;
  if (!k->hasPrefill) { hes__set_error("prefill: no prefill graph in this session (or already released)"); return -1; }
  if (n <= 0 || n > seq) { hes__set_error("prefill: n out of range"); return -1; }
  float* emb = malloc((size_t)seq * k->hidden * 4);
  embed_rows(k, toks, n, seq, emb);
  float* sel = calloc((size_t)seq, 4);
  sel[n - 1] = 1.0f;
  iree_hal_buffer_view_t* ins[3] = {0};
  int nOut = 2 * k->nLayers + 1;
  iree_hal_buffer_view_t** outs = calloc((size_t)nOut, sizeof(void*));
  iree_hal_dim_t dT[1] = {(iree_hal_dim_t)seq}, dE[2] = {(iree_hal_dim_t)seq, (iree_hal_dim_t)k->hidden}, dS[2] = {1, (iree_hal_dim_t)seq};
  iree_status_t st = view_i32(k, toks, 1, dT, &ins[0]);
  if (iree_status_is_ok(st)) st = view_f32(k, emb, 2, dE, &ins[1]);
  if (iree_status_is_ok(st)) st = view_f32(k, sel, 2, dS, &ins[2]);
  if (iree_status_is_ok(st)) st = invoke_graph(&k->prefill, ins, 3, outs, nOut);
  int32_t tok = -1;
  if (iree_status_is_ok(st)) st = read_i32(k, outs[nOut - 1], &tok);
  if (iree_status_is_ok(st)) {
    int* past = calloc((size_t)k->nLayers, sizeof(int));
    release_views(k->kv, 2 * k->nLayers);
    st = adopt_outputs(k, outs, seq, n, past);
    free(past);
    k->pos = n;
  }
  release_views(ins, 3); release_views(outs, nOut); free(outs); free(emb); free(sel);
  return hes__check(st, "prefill") ? -1 : tok;
}

int32_t hes_kv_chunk(hes_kv* k, const int32_t* tokensIn, int32_t n) {
  g_error[0] = 0;
  int C = k->chunkC;
  if (n <= 0 || n > C) { hes__set_error("chunk: n must be in 1..%d", C); return -1; }
  if (k->len[0] == 0) { hes__set_error("chunk: empty cache, prefill first"); return -1; }
  int hasSliding = k->period > 1;
  int32_t* toks = calloc((size_t)C, 4);
  memcpy(toks, tokensIn, (size_t)n * 4);
  int hd = k->headDim;
  float* emb = malloc((size_t)C * k->hidden * 4);
  embed_rows(k, toks, n, C, emb);
  float *cosS = NULL, *sinS = NULL;
  float* cosG = malloc((size_t)C * hd * 4);
  float* sinG = malloc((size_t)C * hd * 4);
  if (hasSliding) { cosS = malloc((size_t)C * hd * 4); sinS = malloc((size_t)C * hd * 4); rope_rows(k, k->pos, C, k->baseS, cosS, sinS); }
  rope_rows(k, k->pos, C, k->baseG, cosG, sinG);
  float* sel = calloc((size_t)C, 4);
  sel[n - 1] = 1.0f;

  int nIn = 2 + (hasSliding ? 4 : 2) + 2 * k->nLayers + (hasSliding ? 2 : 1) + 1, nOut = 2 * k->nLayers + 1;
  iree_hal_buffer_view_t** ins = calloc((size_t)nIn, sizeof(void*));
  iree_hal_buffer_view_t** outs = calloc((size_t)nOut, sizeof(void*));
  iree_hal_buffer_view_t** cK = calloc((size_t)k->nLayers, sizeof(void*));
  iree_hal_buffer_view_t** cV = calloc((size_t)k->nLayers, sizeof(void*));
  int pastS = 0, pastG = 0;
  iree_status_t st = cache_inputs(k, cK, cV, &pastS, &pastG);
  float *maskS = NULL, *maskG = NULL;
  iree_hal_dim_t dT[1] = {(iree_hal_dim_t)C}, dE[2] = {(iree_hal_dim_t)C, (iree_hal_dim_t)k->hidden};
  iree_hal_dim_t dR[2] = {(iree_hal_dim_t)C, (iree_hal_dim_t)hd}, dSel[2] = {1, (iree_hal_dim_t)C};
  iree_hal_buffer_view_t *vCosS = NULL, *vSinS = NULL, *vCosG = NULL, *vSinG = NULL, *vMaskS = NULL, *vMaskG = NULL;
  if (iree_status_is_ok(st)) {
    if (hasSliding) { maskS = malloc((size_t)k->maskH * C * (pastS + C) * 4); chunk_mask(k, pastS, n, k->window, maskS); }
    maskG = malloc((size_t)k->maskH * C * (pastG + C) * 4);
    chunk_mask(k, pastG, n, 0, maskG);
    iree_hal_dim_t dMS[4] = {1, (iree_hal_dim_t)k->maskH, (iree_hal_dim_t)C, (iree_hal_dim_t)(pastS + C)};
    iree_hal_dim_t dMG[4] = {1, (iree_hal_dim_t)k->maskH, (iree_hal_dim_t)C, (iree_hal_dim_t)(pastG + C)};
    int i = 0;
    st = view_i32(k, toks, 1, dT, &ins[i++]);
    if (iree_status_is_ok(st)) st = view_f32(k, emb, 2, dE, &ins[i++]);
    if (iree_status_is_ok(st) && hasSliding) st = view_f32(k, cosS, 2, dR, &vCosS);
    if (iree_status_is_ok(st) && hasSliding) st = view_f32(k, sinS, 2, dR, &vSinS);
    if (iree_status_is_ok(st)) st = view_f32(k, cosG, 2, dR, &vCosG);
    if (iree_status_is_ok(st)) st = view_f32(k, sinG, 2, dR, &vSinG);
    if (iree_status_is_ok(st) && hasSliding) st = view_f32(k, maskS, 4, dMS, &vMaskS);
    if (iree_status_is_ok(st)) st = view_f32(k, maskG, 4, dMG, &vMaskG);
    if (iree_status_is_ok(st)) {
      int introS = 0, introG = 0;
      for (int l = 0; l < k->nLayers; ++l) {
        int g = is_global(k, l);
        if (g && !introG) { ins[i++] = vCosG; ins[i++] = vSinG; }
        if (hasSliding && !g && !introS) { ins[i++] = vCosS; ins[i++] = vSinS; }
        ins[i++] = cK[l]; ins[i++] = cV[l];
        if (g && !introG) { ins[i++] = vMaskG; introG = 1; }
        if (hasSliding && !g && !introS) { ins[i++] = vMaskS; introS = 1; }
      }
      st = view_f32(k, sel, 2, dSel, &ins[i++]);
      if (iree_status_is_ok(st) && i != nIn) st = iree_make_status(IREE_STATUS_INTERNAL, "chunk: assembled %d inputs, expected %d", i, nIn);
    }
  }
  if (iree_status_is_ok(st)) st = invoke_graph(&k->chunk, ins, nIn, outs, nOut);
  int32_t tok = -1;
  if (iree_status_is_ok(st)) st = read_i32(k, outs[nOut - 1], &tok);
  if (iree_status_is_ok(st)) {
    int* past = calloc((size_t)k->nLayers, sizeof(int));
    for (int l = 0; l < k->nLayers; ++l) past[l] = is_global(k, l) ? pastG : pastS;
    st = adopt_outputs(k, outs, C, n, past);
    free(past);
    k->pos += n;
  }
  /* owned here: tokens, emb, select, the rope/mask views; cache views are retained copies */
  if (ins[0]) iree_hal_buffer_view_release(ins[0]);
  if (ins[1]) iree_hal_buffer_view_release(ins[1]);
  if (ins[nIn - 1]) iree_hal_buffer_view_release(ins[nIn - 1]);
  iree_hal_buffer_view_t* owned[6] = {vCosS, vSinS, vCosG, vSinG, vMaskS, vMaskG};
  release_views(owned, 6);
  release_views(cK, k->nLayers); release_views(cV, k->nLayers); release_views(outs, nOut);
  free(ins); free(outs); free(cK); free(cV); free(toks); free(emb);
  free(cosS); free(sinS); free(cosG); free(sinG); free(sel); free(maskS); free(maskG);
  return hes__check(st, "chunk") ? -1 : tok;
}

int32_t hes_kv_step(hes_kv* k, int32_t token) {
  g_error[0] = 0;
  if (k->len[0] == 0) { hes__set_error("step: empty cache, prefill first"); return -1; }
  int hd = k->headDim;
  int32_t t = token;
  float* emb = malloc((size_t)k->hidden * 4);
  embed_rows(k, &t, 1, 1, emb);
  float cosS[1024], sinS[1024], cosG[1024], sinG[1024];
  int hasSliding = k->period > 1;
  if (hasSliding) rope_rows(k, k->pos, 1, k->baseS, cosS, sinS);
  rope_rows(k, k->pos, 1, k->baseG, cosG, sinG);

  int nIn = 2 + (hasSliding ? 4 : 2) + 2 * k->nLayers, nOut = 2 * k->nLayers + 1;
  iree_hal_buffer_view_t** ins = calloc((size_t)nIn, sizeof(void*));
  iree_hal_buffer_view_t** outs = calloc((size_t)nOut, sizeof(void*));
  iree_hal_buffer_view_t** cK = calloc((size_t)k->nLayers, sizeof(void*));
  iree_hal_buffer_view_t** cV = calloc((size_t)k->nLayers, sizeof(void*));
  int pastS = 0, pastG = 0;
  iree_status_t st = cache_inputs(k, cK, cV, &pastS, &pastG);
  iree_hal_dim_t dT[1] = {1}, dE[2] = {1, (iree_hal_dim_t)k->hidden}, dR[2] = {1, (iree_hal_dim_t)hd};
  iree_hal_buffer_view_t *vCosS = NULL, *vSinS = NULL, *vCosG = NULL, *vSinG = NULL;
  if (iree_status_is_ok(st)) {
    int i = 0;
    st = view_i32(k, &t, 1, dT, &ins[i++]);
    if (iree_status_is_ok(st)) st = view_f32(k, emb, 2, dE, &ins[i++]);
    if (iree_status_is_ok(st) && hasSliding) st = view_f32(k, cosS, 2, dR, &vCosS);
    if (iree_status_is_ok(st) && hasSliding) st = view_f32(k, sinS, 2, dR, &vSinS);
    if (iree_status_is_ok(st)) st = view_f32(k, cosG, 2, dR, &vCosG);
    if (iree_status_is_ok(st)) st = view_f32(k, sinG, 2, dR, &vSinG);
    if (iree_status_is_ok(st)) {
      int introS = 0, introG = 0;
      for (int l = 0; l < k->nLayers; ++l) {
        int g = is_global(k, l);
        if (g && !introG) { ins[i++] = vCosG; ins[i++] = vSinG; introG = 1; }
        if (hasSliding && !g && !introS) { ins[i++] = vCosS; ins[i++] = vSinS; introS = 1; }
        ins[i++] = cK[l]; ins[i++] = cV[l];
      }
      if (i != nIn) st = iree_make_status(IREE_STATUS_INTERNAL, "step: assembled %d inputs, expected %d", i, nIn);
    }
  }
  if (iree_status_is_ok(st)) st = invoke_graph(&k->withPast, ins, nIn, outs, nOut);
  int32_t tok = -1;
  if (iree_status_is_ok(st)) st = read_i32(k, outs[nOut - 1], &tok);
  if (iree_status_is_ok(st)) {
    int* past = calloc((size_t)k->nLayers, sizeof(int));
    for (int l = 0; l < k->nLayers; ++l) past[l] = is_global(k, l) ? pastG : pastS;
    st = adopt_outputs(k, outs, 1, 1, past);
    free(past);
    k->pos += 1;
  }
  if (ins[0]) iree_hal_buffer_view_release(ins[0]);
  if (ins[1]) iree_hal_buffer_view_release(ins[1]);
  iree_hal_buffer_view_t* owned[4] = {vCosS, vSinS, vCosG, vSinG};
  release_views(owned, 4);
  release_views(cK, k->nLayers); release_views(cV, k->nLayers); release_views(outs, nOut);
  free(ins); free(outs); free(cK); free(cV); free(emb);
  return hes__check(st, "step") ? -1 : tok;
}

hes_kv_snapshot* hes_kv_snapshot_take(hes_kv* k) {
  hes_kv_snapshot* s = calloc(1, sizeof *s);
  s->n = 2 * k->nLayers;
  s->pos = k->pos;
  s->kv = calloc((size_t)s->n, sizeof(void*));
  s->len = calloc((size_t)k->nLayers, sizeof(int));
  for (int i = 0; i < s->n; ++i) { s->kv[i] = k->kv[i]; if (s->kv[i]) iree_hal_buffer_view_retain(s->kv[i]); }
  memcpy(s->len, k->len, sizeof(int) * (size_t)k->nLayers);
  return s;
}

int hes_kv_restore(hes_kv* k, hes_kv_snapshot* s) {
  g_error[0] = 0;
  if (s->n != 2 * k->nLayers) { hes__set_error("restore: snapshot from another session"); return -1; }
  for (int i = 0; i < s->n; ++i) {
    if (s->kv[i]) iree_hal_buffer_view_retain(s->kv[i]);
    if (k->kv[i]) iree_hal_buffer_view_release(k->kv[i]);
    k->kv[i] = s->kv[i];
  }
  memcpy(k->len, s->len, sizeof(int) * (size_t)k->nLayers);
  k->pos = s->pos;
  return 0;
}

void hes_kv_snapshot_release(hes_kv_snapshot* s) {
  if (!s) return;
  release_views(s->kv, s->n);
  free(s->kv); free(s->len); free(s);
}
