# SKaiNET cartridge blueprints

Open-source **blueprints** for [SKaiNET cartridges](https://github.com/SKaiNET-developers/SKaiNET-cartridge),
and the reference **materializer** that turns a blueprint into a cartridge.

A *cartridge* is a self-contained package that runs one model on one target: native runtime, model
artifacts, capability descriptor, signed manifest. A **blueprint** is everything needed to build one —
API and runtime *source*, a descriptor template and a machine-readable recipe — **without** the model
weights and without compiled binaries. Turning a blueprint into a cartridge is *materialization*:

```
blueprint ─► fetch ─► convert ─► quantize ─► export StableHLO ─► compile (IREE tools) ─► pack + sign ─► cartridge
   +          weights   Kotlin     Kotlin      Kotlin              StableHLO → IREE        descriptor + manifest
 profile      from their publisher, pinned by revision + SHA-256, under terms YOU accept
```

Normative text: [Blueprints](https://skainet-developers.github.io/SKaiNET-cartridge/skainet-cartridge/blueprints.html)
(cartridge spec v0.4, SKEEP-007). This repository never contains weights, `.vmfb`, `.irpa` or native libraries.

## What is here

| Path | Content | Status |
|---|---|---|
| [`blueprint-gradle-plugin/`](blueprint-gradle-plugin/) | Gradle plugin `sk.ainet.cartridge.blueprint` — the materializer | core working, see below |
| [`blueprints/nlu-functiongemma-270m-iree/`](blueprints/nlu-functiongemma-270m-iree/) | Function-calling NLU, FunctionGemma 270M — Kotlin Multiplatform API (runtime binding: Android), recipe, step-by-step guide (weights: Gemma Terms of Use, acceptance required) | recipe verified step by step; end-to-end materialization needs your license acceptance |
| `blueprints/asr-moonshine-v2-streaming-iree/` | Streaming ASR, Moonshine v2 tiny, en + de (weights: MIT) | planned |
| `docs/` | Model-specific guides: download, convert, quantize, compile | planned |

## The materializer

```kotlin
plugins { id("sk.ainet.cartridge.blueprint") }

blueprint {
    // What the model-specific build tasks produced, under the names blueprint.json uses.
    product("graphs-vmfb", compileGraphs.flatMap { it.outputDir })
    product("runtime-so", buildRuntime.flatMap { it.library })
}
```

```
./gradlew blueprintValidate                                   # blueprint.json (+ profile) against the spec schemas
./gradlew blueprintFetch       -Pprofile=profiles/mine.json   # license gate, then download + verify every digest
./gradlew materializeCartridge -Pprofile=profiles/mine.json   # descriptor, effective license, pack_dir, manifest, signature
```

What it does today, each covered by tests:

- validates `blueprint.json` and the profile against the spec's JSON Schemas (bundled at a recorded spec commit);
- **stops before any download** when a source is `acceptance-required` and the profile records no acceptance — it
  never accepts a license for you;
- fetches `hf://`, `https://` and `file:` sources (or a profile's mirror) **with SKaiNET's own
  [`skainet-data-source`](https://github.com/SKaiNET-developers/SKaiNET/tree/develop/skainet-data/skainet-data-source)**:
  SHA-256 verified while streaming and before anything enters the cache, one shared cache per machine
  (`~/.cache/skainet/data`), `./gradlew --offline` served from that cache, and `HF_TOKEN` /
  `HUGGING_FACE_HUB_TOKEN` attached to hub requests only — not forwarded along the hub's CDN redirect
  (pinned by a test). Plain `http://` is refused;
- resolves the descriptor from the template — `performance`/`quality` must come from the profile's *measurements*,
  never from a blueprint's reference numbers;
- computes the effective license from what each artifact is `derived_from`;
- writes the `pack_dir` (`descriptor.json`, `manifest.json`, `artifacts/<role>/…`) with `provenance.blueprint` and
  `provenance.materialization`;
- signs with Ed25519 exactly as the spec's `sign_manifest.py` does — a manifest produced here verifies with the
  spec's `verify_manifest.py`.

The steps in between — convert, quantize, export, compile, build-runtime, measure — are model- and
backend-specific. They are ordinary Gradle tasks in each blueprint module; the plugin sees their results as
named *products*.

Reusable step tasks the plugin offers to blueprint modules: `HostGatherTask` (move a decoder's token-embedding
lookup to the host; a text rewrite of the exported StableHLO, byte-identical to the rewrite it replaces),
`IreeCompileTask` and `IreeConvertParametersTask` (IREE tools, StableHLO → IREE, in the pinned container image),
`blueprint.sourceFile(source, path)` (a fetched, verified file as a task input) and `blueprint.target` (the
profile's target and its params).

The signing key is read from the environment variable the profile's `signing.key_ref` names, or from
`-PsigningKeyFile=<pem>` (unencrypted PKCS#8, `openssl genpkey -algorithm ed25519`). It is never in a profile.

## Build

```
./gradlew build        # builds the plugin, runs unit tests and TestKit functional tests
```

JDK 21 or newer. `BLUEPRINT_NETWORK_TESTS=1 ./gradlew build` additionally runs real `hf://` downloads of two small,
MIT-licensed, pinned files (off by default, so the build works offline).

## License

Code and recipes: MIT, see [`LICENSE`](LICENSE). Model weights are not part of this repository and keep their
publishers' licenses — each blueprint states them per source, with a link.
