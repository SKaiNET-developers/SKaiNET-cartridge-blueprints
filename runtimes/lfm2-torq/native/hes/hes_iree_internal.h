/*
 * Shared between the shim's translation units (hes_iree.c, hes_moonshine.c); not part of the cinterop surface.
 */
#ifndef HES_IREE_INTERNAL_H
#define HES_IREE_INTERNAL_H

#include "hes_iree.h"

#include "iree/io/file_handle.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/runtime/api.h"

struct hes_session {
  iree_runtime_session_t* session;
  iree_io_parameter_provider_t* provider;
  iree_hal_device_t* device;  /* borrowed from the runtime */
};

/* Clears this thread's error message (start of every public entry point). */
void hes__clear_error(void);
void hes__set_error(const char* fmt, ...);
/* Records `st` (consumed) under `what`; returns 0 when ok, -1 otherwise. */
int hes__check(iree_status_t st, const char* what);
/* The shared device for `uri`, created on first use. Borrowed: the runtime owns it. */
iree_hal_device_t* hes__device_for(hes_runtime* rt, const char* uri);
/* Read-only mapping of `path` as a host-allocation file handle (zero-copy parameter import). */
iree_status_t hes__open_mapped(const char* path, iree_io_file_handle_t** out);
iree_status_t hes__open_session(hes_runtime* rt, iree_hal_device_t* device, const char* vmfb, const char* irpa,
                                const char* scope, hes_session* s, iree_io_parameter_index_t** out_index);
void hes__close_session(hes_session* s);
iree_hal_buffer_params_t hes__device_params(void);
iree_status_t hes__view_copy(iree_hal_device_t* device, iree_hal_element_type_t type, int rank,
                             const iree_hal_dim_t* dims, const void* data, size_t bytes, iree_hal_buffer_view_t** out);

#endif
