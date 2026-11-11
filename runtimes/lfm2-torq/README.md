# lfm2-torq: the runtime of `generator-lfm25-350m-torq`

`liblfm2_torq_cartridge.so` is the artifact with role `runtime` in a `generator-lfm25-350m-torq` pack. It is a Kotlin/Native shared library for linux-aarch64 that exports the Cartridge Runtime ABI 0.1 (`native/include/cartridge_abi.h`). It runs the blueprint's own greedy decoder (`GreedyDecoder`, the blueprint's Kotlin API) over the decode step on the Synaptics Torq NPU, inside the host's process. The Torq-fork IREE runtime and the `hes_iree` shim are linked in statically, so the board needs no separate runtime library or Java.

## ABI behaviour

* `io.mode` is `streaming`, with int32 token ids (little-endian) in and out. The library has no tokenizer: the pack's `artifacts/tokenizer/` files are there for the host.
* `ctg_push` appends prompt ids. The prompt plus the generation must fit the compiled capacity (`io.txt`, 64 for the `torq-sl26xx` target).
* `ctg_flush` ends the prompt.
* Each `ctg_pull` produces one new greedy token as a `PARTIAL` result, running the decode step once. The first pull also runs one step per prompt token.
* When `generation_config.json`'s `eos_token_id` comes out or the capacity is reached, the next pull returns `FINAL` with every generated token, the stop token included. After that the stream is drained (`NULL`).
* To generate fewer tokens, stop pulling and call `ctg_session_end`. `ctg_cancel` stops at the next step boundary.
* `ctg_infer` returns `CTG_E_UNSUPPORTED`.
* Sessions are independent; each has its own conv histories and K/V. The step calls of all sessions are serialized on the one module loaded by `ctg_open`.
* `ctg_preflight` checks the pack's files. It also checks that `descriptor.json` is byte-identical to the descriptor embedded in the library: `ctg_descriptor_json` returns its canonical form, and the build embeds exactly what `materializeCartridge` writes. The torq device is opened by `ctg_open`.
* Configuration comes from the pack only: `artifacts/model/{decode_step.vmfb,io.txt,export.json}`, `artifacts/weights/embed.bf16.bin` and `artifacts/tokenizer/generation_config.json`.

## Build

The blueprint builds the library as part of `materializeCartridge`. Neither `build` nor `assemble` runs Docker or the link.

1. **`buildTorqNative`** runs `native/build-native.sh` in `skainet/iree-runtime-torq:2.3.0`. It produces:
   * the source-built Torq-fork IREE runtime archive with its headers and license set;
   * the vendored shim (`native/hes/`, MIT, pinned in `native/upstream.json`), compiled with the image's Kotlin/Native clang and sysroot;
   * the descriptor bytes (`lfm2_torq_descriptor.h`).
2. **`linkCartridgeReleaseSharedLinuxArm64`** links those three archives with the Kotlin code.

The image is not published. Build it from SKaiNET-embedded-vendors with `synaptics-torq/docker-runtime/build.sh`. `-PtorqRuntimeImage=<tag>` selects another one.

## Check on a device

`native/tests/ctg_smoke.c` drives a pack the way a host does, through `dlopen` and the ABI alone. Build it as its header comment describes, then run:

```sh
./ctg_smoke <pack_dir> 1,6,6423,708,... [max new tokens]
```
