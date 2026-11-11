/*
 * ctg_smoke: drives a materialized pack through its runtime library the way a host does, with dlopen and the
 * Cartridge Runtime ABI only (no link-time dependency on the library).
 *
 *   ctg_smoke <pack_dir> <prompt token ids, comma-separated> [max new tokens]
 *
 * Prints the ABI version, preflight result, open time, every PARTIAL token with its pull time, the FINAL tokens and
 * the step p50. Exit status 0 when the stream ends with FINAL. Build (in skainet/iree-runtime-torq):
 *   $KONAN_LLVM/bin/clang --target=aarch64-unknown-linux-gnu --sysroot=$KONAN_GCC/aarch64-unknown-linux-gnu/sysroot \
 *     --gcc-toolchain=$KONAN_GCC -O2 -I../include ctg_smoke.c -ldl -o ctg_smoke
 */
#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cartridge_abi.h"

static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static int cmp(const void* a, const void* b) {
  double x = *(const double*)a, y = *(const double*)b;
  return x < y ? -1 : x > y;
}

#define SYM(name) typeof(&name) p_##name = (typeof(&name))dlsym(lib, #name); \
  if (!p_##name) { fprintf(stderr, "missing symbol %s\n", #name); return 2; }

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: ctg_smoke <pack_dir> <ids,comma-separated> [max new tokens]\n");
    return 2;
  }
  const char* pack = argv[1];
  int max_new = argc > 3 ? atoi(argv[3]) : 1 << 30;
  char lib_path[4096];
  snprintf(lib_path, sizeof lib_path, "%s/artifacts/runtime/liblfm2_torq_cartridge.so", pack);
  double t0 = now_ms();
  void* lib = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);
  if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
  SYM(ctg_abi_version) SYM(ctg_descriptor_json) SYM(ctg_preflight) SYM(ctg_open) SYM(ctg_close)
  SYM(ctg_session_begin) SYM(ctg_session_end) SYM(ctg_push) SYM(ctg_flush) SYM(ctg_pull) SYM(ctg_infer)
  SYM(ctg_result_get_kind) SYM(ctg_result_get_payload) SYM(ctg_result_release) SYM(ctg_last_error) SYM(ctg_cancel)
  printf("dlopen %.1f ms; abi 0x%08x; descriptor %zu bytes\n", now_ms() - t0, p_ctg_abi_version(), strlen(p_ctg_descriptor_json()));

  ctg_preflight_report report = { .size = sizeof report };
  ctg_status st = p_ctg_preflight(pack, &report);
  printf("preflight %d ok=%d%s%s\n", st, report.ok, report.detail ? " " : "", report.detail ? report.detail : "");
  if (st != CTG_OK) return 1;

  ctg_cartridge* c = NULL;
  t0 = now_ms();
  st = p_ctg_open(pack, NULL, &c);
  if (st != CTG_OK) { fprintf(stderr, "open %d: %s\n", st, p_ctg_last_error(NULL)); return 1; }
  printf("open %.1f ms\n", now_ms() - t0);

  ctg_session* s = NULL;
  ctg_session_options so = { .size = sizeof so };
  if ((st = p_ctg_session_begin(c, &so, &s)) != CTG_OK) { fprintf(stderr, "session %d: %s\n", st, p_ctg_last_error(c)); return 1; }
  ctg_buffer dummy = { .size = sizeof dummy };
  ctg_result* r = NULL;
  printf("infer -> %d (expected %d)\n", p_ctg_infer(s, &dummy, &r), CTG_E_UNSUPPORTED);

  int32_t ids[4096]; size_t n = 0;
  for (char* tok = strtok(argv[2], ","); tok && n < 4096; tok = strtok(NULL, ",")) ids[n++] = atoi(tok);
  ctg_buffer in = { .size = sizeof in, .data = ids, .len = n * sizeof(int32_t) };
  if ((st = p_ctg_push(s, &in)) != CTG_OK || (st = p_ctg_flush(s)) != CTG_OK) {
    fprintf(stderr, "push/flush %d: %s\n", st, p_ctg_last_error(s)); return 1;
  }
  printf("push after flush -> %d (expected %d)\n", p_ctg_push(s, &in), CTG_E_STATE);

  double times[4096]; int steps = 0, got_final = 0, partials = 0;
  for (;;) {
    t0 = now_ms();
    st = p_ctg_pull(s, &r);
    double dt = now_ms() - t0;
    if (st != CTG_OK) { fprintf(stderr, "pull %d: %s\n", st, p_ctg_last_error(s)); return 1; }
    if (!r) break;
    ctg_buffer view = { .size = sizeof view };
    p_ctg_result_get_payload(r, &view);
    const int32_t* t = (const int32_t*)view.data;
    size_t count = view.len / sizeof(int32_t);
    if (p_ctg_result_get_kind(r) == CTG_RESULT_PARTIAL) {
      printf("partial %d  (%.1f ms)\n", t[0], dt);
      if (partials++ > 0 && steps < 4096) times[steps++] = dt;   /* the first pull also runs the prompt */
    } else {
      printf("final");
      for (size_t i = 0; i < count; i++) printf("%s%d", i ? "," : " ", t[i]);
      printf("\n");
      got_final = 1;
    }
    p_ctg_result_release(r);
    if (partials >= max_new && !got_final) { printf("stopping after %d tokens\n", partials); break; }
  }
  if (steps) {
    qsort(times, steps, sizeof times[0], cmp);
    printf("decode pull p50 %.1f ms over %d steps\n", times[steps / 2], steps);
  }
  p_ctg_session_end(s);
  printf("close %d\n", p_ctg_close(c));
  return got_final || partials >= max_new ? 0 : 1;
}
