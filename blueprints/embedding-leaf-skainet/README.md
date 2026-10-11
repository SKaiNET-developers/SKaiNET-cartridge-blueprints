# embedding-leaf-skainet

The first **neural** `task: "embedding"` blueprint: MongoDB's
[mdbr-leaf-ir](https://huggingface.co/MongoDB/mdbr-leaf-ir) retrieval encoder (Apache-2.0,
~23 M params, 768-d output) running entirely on SKaiNET-transformers' commonMain BERT path in
**eager mode** — WordPiece tokenize → 6-layer BERT (384 hidden) → mean pooling →
sentence-transformers `2_Dense` projection 384→768 → L2 normalization. That is exactly the
checkpoint's `modules.json` stack, and the BERT path it runs on is the one upstream verifies
against sentence-transformers.

It pins the descriptor conventions for `task: "embedding"` (`io.output` float32 `[D]`,
`space_fingerprint`, prefixes, goldens-as-space-definition) and proves them on a real
model. LEAF is a *symmetric* retrieval model: both prefixes are empty, matching upstream's
`EmbeddingModelProfiles`.

## One engine, two contracts

The `embedding/v1` contract lives in [`profiles/embedding-v1`](../../profiles/embedding-v1),
the task profile every embedding cartridge implements. The
`LeafEmbeddingEngine` implements that profile *and* SKaiNET-transformers' `llm-api`
`EmbeddingModel` SPI with a single class — they were designed to converge
(`embed(text): FloatArray`, `dimensions`, `AutoCloseable`), and `spiAndProfileAgree` in
commonTest holds them to it byte-for-byte.

```kotlin
val engine: LeafEmbeddingEngine = LeafEmbeddingCartridge.open()   // suspend; ~100 ms warm
val vector = engine.embed("which JEP introduced virtual threads") // FloatArray(768), unit norm

// The same object is an llm-api EmbeddingModel:
val model: EmbeddingModel = engine
model.embedQuery("...")        // == embed(...): LEAF is symmetric
```

## Weights

This cartridge has real artifacts: the pinned Hugging Face
snapshot (revision `4262131b…`, sha256 per file in `blueprint.json`), fetched into the shared
SKaiNET model cache `~/.cache/skainet/models/MongoDB_mdbr-leaf-ir` — the same directory
`BertEmbeddingModel.fromHuggingFace` populates on JVM. The tokenizer ships *inside* the
cartridge (`artifacts/vocab.txt`, role `tokenizer`), per the spec's self-containment rule.
Weights are **never in git**; tests skip loudly when the snapshot is missing.

## Verification

`samples/golden-vectors.json` pins 8-dim prefixes of three fixed inputs plus their cosine
structure (query↔doc 0.809, query↔unrelated 0.674 — ordering is part of the space).
commonTest asserts them on every compiled target within a **2e-3 per-dimension cross-backend
budget**: the JVM (`jdk.incubator.vector`) and macosArm64 (Accelerate) backends provably share
one `mdbr-leaf-ir-v1` space. Regenerate deliberately with:

```bash
./gradlew :blueprints:embedding-leaf-skainet:jvmTest -PregenerateGoldens=true --tests "*GoldenRegenerator*"
```

## Building

```bash
./gradlew -PuseMavenLocal=true :blueprints:embedding-leaf-skainet:blueprintValidate
./gradlew -PuseMavenLocal=true :blueprints:embedding-leaf-skainet:jvmTest \
    :blueprints:embedding-leaf-skainet:macosArm64Test
```

Targets: `jvm` (Android consumes the jvm artifact; D8 dexes it) and `macosArm64`. **No iOS
yet** — not for any architectural reason (eager mode is the iOS-clean tier, no dlopen
anywhere), but because upstream publishes no iosArm64 klibs for
`skainet-transformers-inference-bert`. The day those appear, adding `iosArm64()` to
`build.gradle.kts` is the entire port.
