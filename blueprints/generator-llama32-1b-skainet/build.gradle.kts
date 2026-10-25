// Blueprint: Llama-3.2-1B-Instruct text generation on SKaiNET-transformers' llama decoder
// path, eager mode over the QUANTIZED GGUF checkpoint (Q4_K_M, packed-block weights served
// zero-copy through the engine's streaming GGUF loader) — the second task:"generation" / llm/v1
// blueprint after generator-smollm2-skainet, and the first one that loads GGUF instead of
// SafeTensors, exercising the engine's quantized matmul kernels (the SKaiNET#1436 arc).
//
// Same two-things-in-one-module shape as the siblings: (1) the cartridge's Kotlin API — an
// llm/v1 engine over skainet-transformers-core's DecoderGgufWeightLoader + inference-llama's
// LlamaNetworkLoader (byte-level BPE tokenize → 16-layer llama 2048h GQA → greedy/sampled
// decode via generateUntilStop), with the checkpoint's own Llama 3 turn structure rendered
// inside the cartridge; (2) the recipe: fetch the pinned Hugging Face GGUF, verify, pack, sign.
//
// Tests need the model on disk (~0.8 GB, never in git):
//   ~/.cache/skainet/models/unsloth_Llama-3.2-1B-Instruct-GGUF (the shared SKaiNET model
//   cache), or point LLAMA32_MODEL_DIR somewhere else (e.g. EdgeTranslator's download at
//   ~/.litert-lm/models/skainet-llama). Without it the generation tests skip loudly.
plugins {
    alias(libs.plugins.kotlinMultiplatform)
    id("sk.ainet.cartridge.blueprint")
    // Hosts consume the cartridge API from ~/.m2, never as a nested source clone.
    `maven-publish`
}

kotlin {
    explicitApi()

    jvm { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_21) } }
    jvmToolchain(21)
    iosArm64()
    iosSimulatorArm64()
    macosArm64()

    sourceSets {
        commonMain.dependencies {
            // The llm/v1 contract — see profiles/llm-v1.
            api(project(":profiles:llm-v1"))

            // api() so the version constraints reach consumers' metadata (the LEAF lesson:
            // an implementation-scoped platform leaves api-scoped deps versionless).
            api(project.dependencies.platform("sk.ainet.transformers:skainet-transformers-bom:0.63.1"))
            implementation("sk.ainet.transformers:skainet-transformers-core")
            implementation("sk.ainet.transformers:skainet-transformers-inference-llama")

            // Engine deliberately ahead of the transformers BOM's 0.62.0 pin: 0.62.1 fixed
            // SKaiNET#1436 (multi-row Q8_0/Q4_0/Q5_x native kernels zeroed every output row
            // but the last) — this blueprint's whole point is the packed-quant GGUF path.
            implementation(project.dependencies.platform("sk.ainet:skainet-bom:0.62.3"))
            implementation("sk.ainet.core:skainet-lang-core")
            implementation("sk.ainet.core:skainet-backend-api")
            implementation("sk.ainet.core:skainet-backend-cpu")
            implementation("sk.ainet.core:skainet-io-core")
            implementation("sk.ainet.core:skainet-io-gguf")

            implementation(libs.kotlinx.coroutines.core)
        }
        jvmMain.dependencies {
            // The FFM row-major kernel pack (FfmRowMajorKernelPack): packed Q4_K weights with
            // MAPPED residency fall to the decoding reference kernel without it — correct but
            // ~1000× slower per matmul (measured: a 64-token decode took hours, not seconds).
            implementation("sk.ainet.core:skainet-backend-native-cpu")
        }
        commonTest.dependencies {
            implementation(libs.kotlin.test)
            implementation(libs.kotlinx.coroutines.test)
        }
    }
}

tasks.withType<org.jetbrains.kotlin.gradle.targets.jvm.tasks.KotlinJvmTest>().configureEach {
    // SKaiNET picks SIMD kernels where the JVM offers them.
    jvmArgs("--add-modules", "jdk.incubator.vector")
    // Packed Q4_K_M weights are mapped, but the KV cache and activations are heap tensors.
    maxHeapSize = "4g"
    environment("LLAMA32_MODEL_DIR", providers.gradleProperty("llama32ModelDir").orNull ?: "")
}
