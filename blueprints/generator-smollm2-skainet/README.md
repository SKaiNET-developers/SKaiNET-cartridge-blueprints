# generator-smollm2-skainet

The first `task: "generation"` blueprint, and the first **llm/v1** (ADR-011) implementation:
Hugging Face's [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
(Apache-2.0, 135 M params) running entirely on SKaiNET-transformers' commonMain llama decoder
path in **eager mode** — byte-level BPE tokenize → 30-layer llama (576 hidden, 9 heads / 3 KV
heads GQA, tied embeddings, RoPE base 100 000) → stop-token-aware decode via
`generateUntilStop` with batched prefill.

ADR-011 asked for a first consumer; this is it. The profile's event vocabulary
(`Ready → Delta* → Final`) maps onto the ABI's `PARTIAL`/`FINAL` pull exactly as ADR-007 laid
out, and the sampling defaults ride the descriptor's open `attributes` block
(`temperature 0.2`, `top_p 0.9`, the model card's recommendation).

## Full Apple coverage

Unlike `embedding-leaf-skainet` (blocked on upstream iOS klibs for inference-bert), the llama
path publishes the **whole target matrix**, so this is the first neural cartridge compiling for
`jvm`, `iosArm64`, `iosSimulatorArm64` and `macosArm64` — eager mode on the SKaiNET backend,
no dlopen anywhere — the iOS-clean tier of the cartridge story.

## The template stays with the weights

`llm/v1` passes **messages as data**; this cartridge renders the checkpoint's own ChatML
template (including its default system prompt) internally. That follows ADR-011's stated
decision — "a host that concatenates `<start_of_turn>` strings is in the wrong layer" — and
deliberately resolves the tension with the ADR sketch's `prompt(rendered: String)` signature,
which predates that sentence.

```kotlin
val engine: SmolLm2Engine = SmolLm2GeneratorCartridge.open()   // suspend; ~2 s cold
val session = engine.session(LlmSessionConfig(temperature = 0f, maxOutputTokens = 64))
session.prompt(listOf(LlmMessage(LlmRole.USER, "What is the capital of France?")))
session.events.collect { event ->                              // Ready → Delta* → Final
    if (event is LlmEvent.Delta) print(event.text)
}
```

Sessions are one-shot (prompt once, collect once): multi-turn chat replays history as
messages into a fresh session, which is the honest shape for a KV-cache-per-engine decoder.
Each decode `reset()`s the runtime — the cache would otherwise leak one turn into the next.

## Weights

The pinned Hugging Face snapshot (revision `12fd25f…`, sha256 per file in `blueprint.json`,
~260 MB bf16 widened to f32 on load) lives in the shared SKaiNET model cache
`~/.cache/skainet/models/HuggingFaceTB_SmolLM2-135M-Instruct`, or wherever `SMOLLM2_MODEL_DIR`
points. The tokenizer ships *inside* the cartridge (`artifacts/tokenizer.json`, role
`tokenizer`). Weights are **never in git**; tests skip loudly when the snapshot is missing.

## Verification

`samples/golden-transcript.json` pins the **greedy decode of one fixed conversation** — token
ids and text. commonTest asserts it token-for-token on every compiled target: greedy argmax
admits no tolerance budget, so the JVM (`jdk.incubator.vector`) and macosArm64 (Accelerate)
backends provably run one decode (a stricter claim than the embedding cartridges' 2e-3/dim).
The golden answer is the 135M model's honest greedy output — small-model quality is what hosts
gate on, never something a cartridge hides. Regenerate deliberately with:

```bash
./gradlew :blueprints:generator-smollm2-skainet:jvmTest -PregenerateGoldens=true --tests "*GoldenRegenerator*"
```

## Building

```bash
./gradlew :blueprints:generator-smollm2-skainet:blueprintValidate
./gradlew :blueprints:generator-smollm2-skainet:jvmTest \
    :blueprints:generator-smollm2-skainet:macosArm64Test
./gradlew :blueprints:generator-smollm2-skainet:publishToMavenLocal   # hosts consume from ~/.m2
```

Tests need ~4 GB heap (set in the build file): the f32-widened weights alone are ~540 MB.
