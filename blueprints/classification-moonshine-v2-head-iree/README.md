# Blueprint: `classification-moonshine-v2-head-iree`

A small classifier on top of a speech encoder you already run. The **Moonshine v2** streaming encoder stays
frozen; this blueprint **trains** a classification head on its rows from *your* label set and *your* recordings,
calibrates when to trust it, and compiles it with IREE for a Vulkan GPU or the CPU.

The typical use is spoken intents: an utterance goes in, and the head says *this is `play_music`, act on it*, *not
sure, use the transcript*, or *not one of mine*. The label set is yours; the code only knows classes.

This directory is a [cartridge blueprint](https://skainet-developers.github.io/SKaiNET-cartridge/skainet-cartridge/blueprints.html):
the model in the SKaiNET DSL, the Kotlin API and the complete build recipe — **no weights, no data, no native
library**. It is the first blueprint whose recipe fits weights (`train` and `calibrate` steps, ADR-018), and its
cartridges follow the `classification/v1` task profile (ADR-019).

> **Status.** Every step runs on the host and is tested. A cartridge was materialized end to end from the German
> example below; the compiled module matches the eager head on test rows (see *Verification*). It has not been run
> on a device yet, and there is no Android binding in this module yet: a host feeds the pack's module through
> SKaiNET-transformers' `IreeModuleRunner` itself.

| | |
|---|---|
| Recipe | [`blueprint.json`](blueprint.json) |
| Encoder (frozen) | `moonshine-ai/moonshine-streaming-tiny-de` @ `928bfb92…` (flavor `de`) or `moonshine-ai/moonshine-streaming-tiny` @ `f8e9dfd8…` (flavor `en`), pinned by SHA-256, MIT |
| Head | Masked attention pooling, one hidden layer of 256, logits. About 170 000 parameters, a 0.4 MB module |
| Your inputs | `label-set` and `dataset` (see *Three files*). The dataset's license enters the cartridge's effective license |
| Code license | MIT |
| Kotlin API | Kotlin Multiplatform (`jvm` today): the head, the pack locator and the decision are common code |
| Targets | `vulkan-armv7`, `cpu-armv7`, `vulkan-arm64`, `cpu-arm64` (Android) |
| Toolchain | SKaiNET 0.63.0, SKaiNET-transformers 0.64.2, IREE tools 3.11.0 (all from Maven Central, plus the IREE tools image) |

## Three files

You bring three files. Everything else is the recipe.

**1. A label set**, like [`samples/label-set.example.json`](samples/label-set.example.json)
([schema](schema/label-set.schema.json)):

```json
{
  "labels": ["play_music", "audio_volume_up", "audio_volume_down", "other"],
  "reject_label": "other",
  "held_out": ["qa_factoid"]
}
```

- `labels` is the output order of the head.
- `reject_label` (optional) is the class for everything else. Every recording whose label is not listed trains
  that class, so the head learns what it should refuse. It is never an accepted result.
- `held_out` (optional) names dataset labels kept away from training and calibration, so the test split can
  measure how the head treats classes it has never heard.

**2. A dataset manifest** ([schema](schema/dataset-manifest.schema.json)) next to your recordings:

```json
{
  "name": "my recordings, 2026-11",
  "entries": [
    { "audio": "wav/0001.wav", "label": "play_music", "split": "train", "speaker": "s01" },
    { "audio": "wav/0002.wav", "label": "qa_factoid", "split": "test",  "speaker": "s07" }
  ]
}
```

WAV files, 16 kHz mono, 16-bit PCM or 32-bit float. `train` fits the head, `dev` picks the epoch and the
calibration, `test` is measured once at the end. Keep a speaker in one split.

**3. A profile**, from [`profiles/example.vulkan-armv7.de.json`](profiles/example.vulkan-armv7.de.json): the
target, the encoder flavor (`de` or `en`, exactly one), and the two inputs with **their licenses**.

```bash
./gradlew :blueprints:classification-moonshine-v2-head-iree:measure              -Pprofile=<yours>.json
./gradlew :blueprints:classification-moonshine-v2-head-iree:materializeCartridge -Pprofile=<yours>.json
```

`measure` writes `build/blueprint/quality-report.<flavor>.json`. Copy its `quality` object into your profile's
`measurements.quality` and add `measurements.performance` from a run on your device; the materializer never
copies numbers for you.

Prerequisites: JDK 21+, Git, and Docker for the IREE steps.

## One definition, three executions

The head is written once, in the SKaiNET DSL
([`model/ClassificationHead.kt`](src/commonMain/kotlin/sk/ainet/cartridge/classification/moonshine/head/model/ClassificationHead.kt)).
A custom module does the pooling; the rest is the layer DSL:

```kotlin
override fun onForward(input: Tensor<FP32, Float>, ctx: ExecutionContext): Tensor<FP32, Float> {
    val (batch, rows) = input.shape.dimensions
    val mask = input.narrow(dim = 2, start = dim, length = 1)                       // [b, t, 1]
    val scores = score.forward(input, ctx) + (mask - 1f) * MASKED                   // [b, t, 1]
    val weights = scores.reshape(Shape(batch, rows)).softmax(dim = 1)               // over the rows
    …
}

MLP(
    MaskedAttentionPool(config.dim, ctx, random),
    sequential<FP32, Float>(ctx) {
        input(config.dim)
        dense(config.hidden, id = "hidden") { weights { randn(std = 0.05f, random = random) }; bias { zeros() } }
        activation { it.relu() }
    },
    Dropout(p = config.dropout, name = "dropout", random = random),
    sequential<FP32, Float>(ctx) { input(config.hidden); dense(config.classes, id = "logits") { … } },
)
```

The same object is then used three ways. Nothing in the definition knows which.

| Execution | Context | Where |
|---|---|---|
| **Trained** | `DefaultGraphExecutionContext(phase = Phase.TRAIN, …)`: dropout on, the tape records for autograd | `TrainCli`, with the training DSL below |
| **Run eagerly** | `DirectCpuExecutionContext()` | dev accuracy, calibration, the quality report, the parity reference |
| **Traced** | `DefaultGraphExecutionContext.tape(VoidTensorOps())` → `StableHloConverterFactory` | `ExportCli`: one static StableHLO function, weights folded in |

Training is the SKaiNET training DSL:

```kotlin
val runner = training<FP32, Float> {
    model { model }
    loss { CrossEntropyLoss() }
    optimizer { adam(lr = 0.001).apply { model.trainableParameters().forEach { addParameter(it) } } }
}
val loss = runner.step(ctx, rows, targets)
```

There is no Python anywhere in the recipe, and no second description of the model to keep in step.

## The recipe

| Step | Kind | What it does |
|---|---|---|
| `fetch` | fetch | The selected encoder's `config.json` and `model.safetensors`, digest-verified |
| `features` | export | Every recording through the frozen encoder, windowed exactly as a streaming recognizer windows it; rows cached in safetensors shards. Resumable |
| `train` | **train** | Fits the head on `train`; class-balanced sampling; keeps the epoch with the best `dev` top-1. Seeded |
| `calibrate` | **calibrate** | On `dev`: the temperature, then the accept and reject thresholds |
| `measure` | measure | The `test` split, once: the quality report |
| `export` | export | The fitted head, traced to StableHLO |
| `compile` | compile | IREE, for the profile's target |
| `build-runtime` | build-runtime | The generic module runner for the target ABI, from the released runtime |
| `pack`, `sign` | | The plugin's |

`train` and `calibrate` follow ADR-018: their data is declared, their seed is in `blueprint.json`, their outputs
are `derived_from` the dataset, and since trained weights are not byte-reproducible, the blueprint's
`verification` states a metric tolerance instead.

`features` is the slow step: about 20 times real time on a 12-core desktop with `-PfeatureWorkers=6`, so ten
hours of audio take about half an hour. Training takes minutes to about half an hour.

## What the cartridge is

```
artifacts/model/head.vmfb                   classify(rows[1, 256, 321]) -> logits[1, classes]
artifacts/weights/head.safetensors          the same weights, to run the head eagerly with SKaiNET
artifacts/other/label-set.json              labels in output order, reject_label
artifacts/calibration/calibration.json      temperature, score, thresholds.accept / .reject
artifacts/runtime/libskainet_iree_module.so the generic single-shot module runner
```

**Input.** The head takes what the encoder gives, described in the descriptor's `attributes.features`:

- rows of the named encoder at its pinned revision, taken *before* the recognizer's adapter (no positional
  table), 50 per second, 320 wide;
- computed in streaming windows of 64 frames that advance by 44, with 4 frames of lookahead;
- at most 256 rows (5.12 s); a longer utterance keeps its first 256;
- one extra column, the mask: 1 for a real row, 0 for padding. Padded rows are ignored exactly.

A recognizer that already runs this encoder can hand its rows to the head; the head costs one small extra call.
`PackDir.features.input(rows, count)` builds the input tensor.

**Output.** Raw logits. The decision is host arithmetic, the same on every backend:

```kotlin
val pack = PackDir("/path/to/pack")
when (val outcome = pack.decision().decide(logits)) {
    is Outcome.Accepted  -> act(outcome.label)        // score at or below thresholds.accept
    is Outcome.Undecided -> useAnotherPath()          // between the thresholds
    is Outcome.Rejected  -> refuse()                  // above thresholds.reject, or the reject class won
}
```

The score is the temperature-scaled energy, `−T · log Σ exp(z / T)` over the classes other than the reject class
(lower is more confident). `thresholds.accept` is the loosest score at which, on `dev`, at most `target_risk`
(1 %) of the accepted inputs were wrong: a wrong label, or an out-of-set input accepted. `thresholds.reject` keeps
95 % of the in-set inputs.

## Verification

- `./gradlew :blueprints:classification-moonshine-v2-head-iree:check` runs the unit tests: the head trains on
  masked rows and ignores padding, survives its weight file, exports to one static function; the decision, the
  metrics and the feature cache; and the blueprint's own rules (no data or weights in the tree, declared data and
  seed on the fitted steps, trained outputs derived from the dataset).
- `parityHost -Pprofile=…` compiles the head for the host and compares `iree-run-module` with the eager head on
  test rows. Measured: 8 test inputs, maximum absolute difference 8.6e-6, same top class on all.
- `test-top1` (`blueprint.json`): a re-materialization from the same inputs and seed reaches the reported test
  top-1 within 0.02.

## The German example: Speech-MASSIVE

[Speech-MASSIVE](https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE) has spoken German utterances with intent
labels. It is licensed **CC BY-NC-SA 4.0**: non-commercial, share-alike. A cartridge trained on it carries that
license (`MIT AND CC-BY-NC-SA-4.0`), so the example is for trying the recipe and for research, not for shipping.

The blueprint does not contain or pin the data as a source. A helper task, not a recipe step, turns the published
parquet files into a manifest and 16 kHz WAVs. It downloads nothing until you name the license:

```bash
./gradlew :blueprints:classification-moonshine-v2-head-iree:prepareSpeechMassive \
    -PdatasetDir=<dir> -PacceptDatasetLicense=CC-BY-NC-SA-4.0
```

It needs about 4.5 GB of downloads and 1.6 GB of WAVs (`-PdatasetOnly='train-00000|validation'` takes a tenth).
The files are pinned by revision and SHA-256 in [`samples/speech-massive-de.source.json`](samples/speech-massive-de.source.json).
The published `train` files become `train` and `dev` (every tenth speaker, whole, goes to `dev`); the published
`validation` files become `test`. Then point your profile's `dataset` input at `<dir>/dataset.json` with
`"license": "CC-BY-NC-SA-4.0"`.

With the example label set (18 classes plus `other`, three classes held out), flavor `de`, seed 20261113:

| | |
|---|---|
| Data | 8 676 train, 1 512 dev, 2 033 test recordings (after holding three classes out of train and dev) |
| Test top-1 (19 classes) | **0.794** |
| Macro F1 | 0.700 |
| Calibration error (ECE) | 0.032 |
| In-set vs out-of-set, AUROC | 0.838 (0.822 against the three classes never heard) |
| Out-of-set passing at 95 % in-set kept | 0.767 |
| Accepted | 44 of 846 in-set inputs (5.2 %), 3 of 1 187 out-of-set inputs |
| Wrong among accepted | 4 of 47 (8.5 %), against a target of 1 % set on dev |
| Rejected | 1 006 of 1 187 out-of-set inputs, 148 of 846 in-set inputs |
| Time | features 39 min (10 workers, 23 times real time), training 16 epochs in about half an hour |

Read these plainly. The classifier is decent for a head this small on a frozen recognizer encoder. The
separation of in-set from out-of-set by the energy score is weak: at a 1 % risk the head accepts only one input
in twenty, and on the test split the accepted ones were wrong more often than the dev-set target. The thresholds
do what they are meant to (little is accepted when the score cannot be trusted), and better separation is where
work on the head and on the out-of-set data should go.

These are what the authors measured on the host, as an expectation. Your cartridge reports your own numbers.

## Limits

- The head decides a class. It does not fill slots; open values (titles, names) need the transcript.
- 256 rows: 5.12 s of speech. Longer utterances are cut.
- The manifest's digest pins the list of recordings, not the audio bytes.
- The feature contract is strict: rows from another encoder, another revision, or taken after the adapter are
  not what the head was trained on.
- Quality depends on your data, above all on the out-of-set examples. A head that has heard few of them accepts
  little: the accept threshold is set by the risk you allow, not by a wish for coverage.

## Licensing

- This blueprint's code: MIT. The encoders: MIT.
- Your dataset and label set carry the licenses you state in the profile; the materializer computes the
  cartridge's effective license from them and records it in the manifest, per artifact.
- The example data adapter uses DuckDB (MIT) to read parquet. It is a separate compilation and never part of a
  cartridge or of the recipe.
