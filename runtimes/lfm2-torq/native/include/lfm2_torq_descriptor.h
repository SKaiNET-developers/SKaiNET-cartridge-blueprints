/*
 * The descriptor embedded in liblfm2_torq_cartridge.so. build-native.sh generates the definitions from the
 * descriptor that materializeCartridge writes into the pack (descriptor.json), so ctg_descriptor_json() and the
 * pack's descriptor cannot drift apart.
 */
#ifndef LFM2_TORQ_DESCRIPTOR_H
#define LFM2_TORQ_DESCRIPTOR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Canonical JSON (sorted keys, compact, UTF-8, NUL-terminated): what ctg_descriptor_json() returns. */
const char* lfm2_torq_descriptor_canonical(void);

/* The exact bytes of the pack's descriptor.json (not NUL-terminated; *length receives the size). */
const unsigned char* lfm2_torq_descriptor_pack(size_t* length);

#ifdef __cplusplus
}
#endif

#endif
