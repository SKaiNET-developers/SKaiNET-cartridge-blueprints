#!/usr/bin/env bash
# Runs inside skainet/iree-runtime-torq (SKaiNET-embedded-vendors synaptics-torq/docker-runtime), which carries the
# Torq-fork IREE runtime built from source and the Kotlin/Native clang + aarch64 sysroot it was built with.
#
#   /src          this directory (read-only)
#   /descriptors  descriptor.json and descriptor.canonical.json (read-only), resolved by the blueprint
#   /out          an empty directory; receives
#                   dist/lib/libiree_torq_runtime.a, dist/include/…, dist/licenses/…, dist/BUILD-INFO  (the image's runtime)
#                   lib/libhes_iree.a          the vendored hes_iree shim (hes/, MIT)
#                   lib/liblfm2_torq_descriptor.a  the descriptor bytes (include/lfm2_torq_descriptor.h)
#
# The Kotlin/Native link (runtimes/lfm2-torq, linkReleaseSharedLinuxArm64) takes the three archives from here.
set -euo pipefail
OUT=/out
: "${KONAN_LLVM:?not the skainet/iree-runtime-torq image}" "${KONAN_GCC:?}" "${TORQ_RUNTIME_DIST:?}"
mkdir -p "$OUT/dist" "$OUT/lib" "$OUT/obj"
cp -r "$TORQ_RUNTIME_DIST"/. "$OUT/dist/"

CC=("$KONAN_LLVM/bin/clang" --target=aarch64-unknown-linux-gnu --sysroot="$KONAN_GCC/aarch64-unknown-linux-gnu/sysroot"
    --gcc-toolchain="$KONAN_GCC" -mcpu=cortex-a55 -O2 -fPIC -std=c11 -Wall -Werror=implicit-function-declaration)

# the shim, as SKaiNET-embedded's build-dist.sh compiles it
for f in hes_iree hes_moonshine; do
  "${CC[@]}" -DIREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl -I"$OUT/dist/include" -c "/src/hes/$f.c" -o "$OUT/obj/$f.o"
done
"$KONAN_LLVM/bin/llvm-ar" rcs "$OUT/lib/libhes_iree.a" "$OUT/obj/hes_iree.o" "$OUT/obj/hes_moonshine.o"

# the descriptor, as byte arrays (no escaping questions)
python3 - /descriptors/descriptor.canonical.json /descriptors/descriptor.json "$OUT/obj/lfm2_torq_descriptor.c" <<'EOF'
import sys
canonical, pack, out = sys.argv[1:]
c = open(canonical, 'rb').read()
p = open(pack, 'rb').read()
def arr(name, data):
    body = ','.join(str(b) for b in data)
    return f"static const unsigned char {name}[{len(data) + 1}] = {{{body}{',' if data else ''}0}};\n"
with open(out, 'w') as f:
    f.write('#include "lfm2_torq_descriptor.h"\n')
    f.write(arr('canonical', c))
    f.write(arr('pack', p))
    f.write('const char* lfm2_torq_descriptor_canonical(void) { return (const char*)canonical; }\n')
    f.write(f'const unsigned char* lfm2_torq_descriptor_pack(size_t* length) {{ *length = {len(p)}; return pack; }}\n')
EOF
"${CC[@]}" -I/src/include -c "$OUT/obj/lfm2_torq_descriptor.c" -o "$OUT/obj/lfm2_torq_descriptor.o"
"$KONAN_LLVM/bin/llvm-ar" rcs "$OUT/lib/liblfm2_torq_descriptor.a" "$OUT/obj/lfm2_torq_descriptor.o"

rm -rf "$OUT/obj"
chown -R "$(stat -c %u:%g "$OUT")" "$OUT"
echo "built: $(cat "$OUT/dist/BUILD-INFO" | tr '\n' ' ')"
