// Blueprint: SmolLM2-135M-Instruct text generation on SKaiNET-transformers' llama decoder
// path, eager mode — the first task:"generation" blueprint, and the first llm/v1 implementation
// (ADR-011 gets its consumer here, exactly as the ADR asked for).
//
// Same two-things-in-one-module shape as the siblings: (1) the cartridge's Kotlin API — an
// `llm/v1` engine over `skainet-transformers-inference-llama`'s commonMain decoder (byte-level
// BPE tokenize → 30-layer llama 576h GQA → greedy/sampled decode via generateUntilStop), with
// the checkpoint's own ChatML template rendered inside the cartridge; (2) the recipe: fetch the
// pinned Hugging Face snapshot, verify the golden transcript, pack, sign.
//
// Unlike embedding-leaf, upstream publishes the WHOLE target matrix for the llama path, so this
// is the first neural cartridge with full Apple coverage: jvm + iosArm64 + iosSimulatorArm64 +
// macosArm64 (Android consumes the jvm artifact).
//
// Tests need the model snapshot on disk (~260 MB, never in git):
//   ~/.cache/skainet/models/HuggingFaceTB_SmolLM2-135M-Instruct (the shared SKaiNET model
//   cache), or point SMOLLM2_MODEL_DIR somewhere else. Without it the golden tests skip loudly
//   instead of failing.
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
            api(project.dependencies.platform("sk.ainet.transformers:skainet-transformers-bom:0.57.1"))
            implementation("sk.ainet.transformers:skainet-transformers-core")
            implementation("sk.ainet.transformers:skainet-transformers-inference-llama")

            implementation(project.dependencies.platform("sk.ainet:skainet-bom:0.57.0"))
            implementation("sk.ainet.core:skainet-lang-core")
            implementation("sk.ainet.core:skainet-backend-cpu")
            implementation("sk.ainet.core:skainet-io-core")
            implementation("sk.ainet.core:skainet-io-safetensors")

            implementation(libs.kotlinx.coroutines.core)
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
    // The f32-widened weights alone are ~540 MB; the default 512 MB test heap OOMs in the
    // bf16 dequant (upstream's llama module runs its tests at 6g for the same reason).
    maxHeapSize = "4g"
    environment("SMOLLM2_MODEL_DIR", providers.gradleProperty("smollm2ModelDir").orNull ?: "")
    environment("REGENERATE_GOLDENS", providers.gradleProperty("regenerateGoldens").orNull ?: "")
}
