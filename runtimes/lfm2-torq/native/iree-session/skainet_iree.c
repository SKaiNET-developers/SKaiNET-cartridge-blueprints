/*
 * skainet_iree: see skainet_iree.h. Generic IREE sessions over the Torq-fork runtime: one instance per process,
 * devices shared by the sessions on them, host <-> device tensors, function calls.
 *
 * Derived from the generic-session part of SKaiNET-embedded's IREE shim (MIT; commit and path in
 * ../upstream.json), with an application-neutral name and without that shim's KV-cache decoder and streaming-ASR
 * session, which this runtime does not use.
 */
#define _GNU_SOURCE  /* strdup, mmap, sysconf */
#include "skainet_iree.h"

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

struct skainet_iree_session {
  iree_runtime_session_t* session;
  iree_io_parameter_provider_t* provider;
  iree_hal_device_t* device;  /* borrowed from the runtime */
};

_Static_assert(SKAINET_IREE_F32 == IREE_HAL_ELEMENT_TYPE_FLOAT_32, "SKAINET_IREE_F32");
_Static_assert(SKAINET_IREE_F16 == IREE_HAL_ELEMENT_TYPE_FLOAT_16, "SKAINET_IREE_F16");
_Static_assert(SKAINET_IREE_BF16 == IREE_HAL_ELEMENT_TYPE_BFLOAT_16, "SKAINET_IREE_BF16");
_Static_assert(SKAINET_IREE_I32 == IREE_HAL_ELEMENT_TYPE_INT_32, "SKAINET_IREE_I32");
_Static_assert(SKAINET_IREE_I64 == IREE_HAL_ELEMENT_TYPE_INT_64, "SKAINET_IREE_I64");
_Static_assert(SKAINET_IREE_I8 == IREE_HAL_ELEMENT_TYPE_INT_8, "SKAINET_IREE_I8");

#define DEFAULT_SCOPE "model"
#define MAX_CONCURRENT_PARAM_OPS 16
#define MAX_DEVICES 4

/* ---------- errors ---------- */

static _Thread_local char g_error[1024];

const char* skainet_iree_last_error(void) { return g_error; }

static void set_error(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(g_error, sizeof g_error, fmt, ap);
  va_end(ap);
}

/* Records `st` (consumed) under `what`; returns 0 when ok, -1 otherwise. */
static int check_status(iree_status_t st, const char* what) {
  if (iree_status_is_ok(st)) return 0;
  char* buf = NULL;
  iree_host_size_t len = 0;
  iree_allocator_t alloc = iree_allocator_system();
  if (iree_status_to_string(st, &alloc, &buf, &len) && buf) {
    set_error("%s: %.*s", what, (int)(len < 900 ? len : 900), buf);
    iree_allocator_free(alloc, buf);
  } else {
    set_error("%s: status code %d", what, (int)iree_status_code(st));
  }
  iree_status_ignore(st);
  return -1;
}

/* ---------- runtime ---------- */

struct skainet_iree_runtime {
  iree_runtime_instance_t* instance;
  struct { char* uri; iree_hal_device_t* device; } devices[MAX_DEVICES];
  int device_count;
  char drivers[256];
};

static int g_flags_parsed = 0;

skainet_iree_runtime* skainet_iree_runtime_create(const char* const* flags, int flag_count) {
  g_error[0] = 0;
  if (!g_flags_parsed && flags && flag_count > 0) {
    /* IREE flags are process-global and parsed from an argv; argv[0] is skipped. String flags keep a pointer into
     * argv (e.g. FLAG_torq_hw_type), so the strings are copied and intentionally never freed. */
    int argc = flag_count + 1;
    char** argv = calloc((size_t)argc + 1, sizeof(char*));
    argv[0] = "skainet-iree";
    for (int i = 0; i < flag_count; ++i) argv[i + 1] = strdup(flags[i]);
    char** parse_argv = argv;
    iree_status_t st = iree_flags_parse(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &parse_argv);
    /* argv itself (the pointer array) may go; the strings it pointed to stay alive. */
    free(argv);
    if (check_status(st, "flags")) return NULL;
    g_flags_parsed = 1;
  }
  skainet_iree_runtime* rt = calloc(1, sizeof *rt);
  iree_runtime_instance_options_t options;
  iree_runtime_instance_options_initialize(&options);
  iree_runtime_instance_options_use_all_available_drivers(&options);
  if (check_status(iree_runtime_instance_create(&options, iree_allocator_system(), &rt->instance), "instance")) {
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

const char* skainet_iree_runtime_drivers(skainet_iree_runtime* rt) { return rt ? rt->drivers : ""; }

void skainet_iree_runtime_release(skainet_iree_runtime* rt) {
  if (!rt) return;
  for (int i = 0; i < rt->device_count; ++i) {
    iree_hal_device_release(rt->devices[i].device);
    free(rt->devices[i].uri);
  }
  iree_runtime_instance_release(rt->instance);
  free(rt);
}

/* The shared device for `uri`, created on first use. Borrowed: the runtime owns it. */
static iree_hal_device_t* device_for(skainet_iree_runtime* rt, const char* uri) {
  for (int i = 0; i < rt->device_count; ++i)
    if (strcmp(rt->devices[i].uri, uri) == 0) return rt->devices[i].device;
  if (rt->device_count == MAX_DEVICES) { set_error("device %s: too many devices", uri); return NULL; }
  iree_hal_device_t* device = NULL;
  char what[128];
  snprintf(what, sizeof what, "device %s", uri);
  if (check_status(iree_runtime_instance_try_create_default_device(rt->instance, iree_make_cstring_view(uri), &device), what))
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
static iree_status_t open_mapped(const char* path, iree_io_file_handle_t** out) {
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
 * New IREE session on `device`; with `irpa` its parameters are appended first (scope `scope`).
 */
static iree_status_t open_session(skainet_iree_runtime* rt, iree_hal_device_t* device, const char* vmfb, const char* irpa,
                                  const char* scope, skainet_iree_session* s) {
  iree_allocator_t alloc = iree_allocator_system();
  s->device = device;
  iree_runtime_session_options_t options;
  iree_runtime_session_options_initialize(&options);
  IREE_RETURN_IF_ERROR(iree_runtime_session_create_with_device(
      rt->instance, &options, device, iree_runtime_instance_host_allocator(rt->instance), &s->session));
  if (irpa) {
    iree_io_file_handle_t* file = NULL;
    IREE_RETURN_IF_ERROR(open_mapped(irpa, &file));
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
    if (index) iree_io_parameter_index_release(index);
    IREE_RETURN_IF_ERROR(st);
  }
  return iree_runtime_session_append_bytecode_module_from_file(s->session, vmfb);
}

static void close_session(skainet_iree_session* s) {
  if (s->session) iree_runtime_session_release(s->session);
  if (s->provider) iree_io_parameter_provider_release(s->provider);
  memset(s, 0, sizeof *s);
}

skainet_iree_session* skainet_iree_session_create(skainet_iree_runtime* rt, const char* device, const char* vmfb, const char* irpa, const char* scope) {
  g_error[0] = 0;
  iree_hal_device_t* dev = device_for(rt, device);
  if (!dev) return NULL;
  skainet_iree_session* s = calloc(1, sizeof *s);
  char what[512];
  snprintf(what, sizeof what, "session %s on %s", vmfb, device);
  if (check_status(open_session(rt, dev, vmfb, irpa, scope, s), what)) {
    close_session(s);
    free(s);
    return NULL;
  }
  return s;
}

void skainet_iree_session_release(skainet_iree_session* s) {
  if (!s) return;
  close_session(s);
  free(s);
}

/* ---------- views ---------- */

/* skainet_iree_view is IREE's buffer view; the alias keeps the header free of IREE types. */
#define VIEW(v) ((iree_hal_buffer_view_t*)(v))

static iree_hal_buffer_params_t device_params(void) {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
  return params;
}

static iree_status_t view_copy(iree_hal_device_t* device, iree_hal_element_type_t type, int rank,
                               const iree_hal_dim_t* dims, const void* data, size_t bytes, iree_hal_buffer_view_t** out) {
  return iree_hal_buffer_view_allocate_buffer_copy(device, iree_hal_device_allocator(device), (iree_host_size_t)rank,
                                                    dims, type, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, device_params(),
                                                    iree_make_const_byte_span(data, bytes), out);
}

skainet_iree_view* skainet_iree_view_from_host(skainet_iree_session* s, int32_t element_type, int rank, const int64_t* dims, const void* data, size_t bytes) {
  g_error[0] = 0;
  if (rank < 0 || rank > 8) { set_error("view: rank %d out of range", rank); return NULL; }
  iree_hal_dim_t d[8];
  size_t count = 1;
  for (int i = 0; i < rank; ++i) { d[i] = (iree_hal_dim_t)dims[i]; count *= (size_t)dims[i]; }
  size_t want = count * iree_hal_element_dense_byte_count((iree_hal_element_type_t)element_type);
  if (want != bytes) { set_error("view: %zu bytes given, shape needs %zu", bytes, want); return NULL; }
  iree_hal_buffer_view_t* v = NULL;
  if (check_status(view_copy(s->device, (iree_hal_element_type_t)element_type, rank, d, data, bytes, &v), "view")) return NULL;
  return (skainet_iree_view*)v;
}

int skainet_iree_view_to_host(skainet_iree_session* s, skainet_iree_view* v, void* dst, size_t bytes) {
  g_error[0] = 0;
  size_t have = (size_t)iree_hal_buffer_view_byte_length(VIEW(v));
  if (have != bytes) { set_error("to_host: view has %zu bytes, %zu requested", have, bytes); return -1; }
  return check_status(iree_hal_device_transfer_d2h(s->device, iree_hal_buffer_view_buffer(VIEW(v)), 0, dst, bytes,
                                            IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()), "to_host");
}

int skainet_iree_view_rank(skainet_iree_view* v) { return (int)iree_hal_buffer_view_shape_rank(VIEW(v)); }
int64_t skainet_iree_view_dim(skainet_iree_view* v, int axis) { return (int64_t)iree_hal_buffer_view_shape_dim(VIEW(v), (iree_host_size_t)axis); }
int32_t skainet_iree_view_element_type(skainet_iree_view* v) { return (int32_t)iree_hal_buffer_view_element_type(VIEW(v)); }
size_t skainet_iree_view_bytes(skainet_iree_view* v) { return (size_t)iree_hal_buffer_view_byte_length(VIEW(v)); }
void skainet_iree_view_retain(skainet_iree_view* v) { if (v) iree_hal_buffer_view_retain(VIEW(v)); }
void skainet_iree_view_release(skainet_iree_view* v) { if (v) iree_hal_buffer_view_release(VIEW(v)); }

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

int skainet_iree_call(skainet_iree_session* s, const char* function, skainet_iree_view* const* inputs, int input_count, skainet_iree_view** outputs, int output_capacity) {
  g_error[0] = 0;
  int n = 0;
  char what[256];
  snprintf(what, sizeof what, "call %s", function);
  if (check_status(invoke(s->session, function, (iree_hal_buffer_view_t* const*)inputs, input_count,
                   (iree_hal_buffer_view_t**)outputs, output_capacity, &n), what))
    return -1;
  return n;
}
