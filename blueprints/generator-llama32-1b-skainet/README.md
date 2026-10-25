# generator-llama32-1b-skainet

The second `task: "generation"` / **llm/v1** blueprint (after
[`generator-smollm2-skainet`](../generator-smollm2-skainet)), and the first on a **quantized
GGUF** checkpoint: Meta's
[Llama-3.2-1B-Instruct](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF) (Unsloth's
Q4_K_M GGUF, ~0.8 GB) running entirely on SKaiNET-transformers' commonMain llama decoder path
in **eager mode** — packed Q4_K blocks served zero-copy (`MAPPED`) through the engine's
streaming GGUF loader into the quantized matmul kernels (the SKaiNET#1436 arc; engine ≥ 0.62.1
required), byte-level BPE tokenize → 16-layer llama (2048 hidden, 32 heads / 8 KV heads GQA,
tied embeddings, RoPE base 500 000) → stop-token-aware decode via `generateUntilStop` with
batched prefill.

Born as the cartridge behind
[SKaiNET-EdgeTranslator](https://github.com/michalharakal/SKaiNET-EdgeTranslator)'s cartridge
engine: the app already ships this exact checkpoint on its legacy path, so the cartridge is the
same model with the packaging story — pinned revision, SHA-256 digests, license gate, descriptor,
signature — and an llm/v1 API in front.

## The template stays with the weights

`llm/v1` passes **messages as data**; this cartridge renders the checkpoint's own Llama 3 turn
structure (`<|begin_of_text|>`, `<|start_header_id|>…<|end_header_id|>`, `<|eot_id|>`)
internally, byte-exact with the family template's no-tools path. Stop set is the checkpoint's
`generation_config.json` trio: 128001, 128008, 128009.

```kotlin
val engine: Llama32Engine = Llama32GeneratorCartridge.open()   // suspend; GGUF maps in ~1 s
val session = engine.session(LlmSessionConfig(temperature = 0f, maxOutputTokens = 256))
session.prompt(listOf(LlmMessage(LlmRole.USER, "Translate to German: good morning!")))
session.events.collect { event ->                              // Ready → Delta* → Final
    if (event is LlmEvent.Delta) print(event.text)
}
```

Sessions are one-shot (prompt once, collect once); each decode `reset()`s the runtime.

## Weights

The pinned snapshot (revision `b69aef1…`, sha256 in `blueprint.json`) lives in the shared
SKaiNET model cache `~/.cache/skainet/models/unsloth_Llama-3.2-1B-Instruct-GGUF`, or wherever
`LLAMA32_MODEL_DIR` points. Weights are **never in git** and never on Maven; the source is
`redistribution: "acceptance-required"` — the **Llama 3.2 Community License** is yours to
accept (the materializer stops before any download until the profile records that acceptance).
Tests skip loudly when the checkpoint is missing.

## Verification

`Llama32GenerationSmokeTest` (jvmTest) runs a real greedy decode against the pinned GGUF and
asserts the llm/v1 event contract (Ready → Delta* → Final, deltas concatenate to the final
text), a STOP finish, and token-for-token determinism across two runs. A committed golden
transcript in the SmolLM2 style is the natural next step once the output is reviewed.

## Building

```bash
./gradlew :blueprints:generator-llama32-1b-skainet:blueprintValidate
./gradlew :blueprints:generator-llama32-1b-skainet:jvmTest
./gradlew :blueprints:generator-llama32-1b-skainet:publishToMavenLocal   # hosts consume from ~/.m2
```
