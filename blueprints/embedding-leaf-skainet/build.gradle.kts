// Blueprint: MongoDB mdbr-leaf-ir sentence embeddings on SKaiNET-transformers' BERT path,
// eager mode — the first *neural* task:"embedding" blueprint.
//
// Same two-things-in-one-module shape as the siblings: (1) the cartridge's Kotlin API — the
// `embedding/v1` engine over `skainet-transformers-inference-bert`'s commonMain encoder
// (tokenize → 6-layer BERT → mean pooling → 2_Dense 384→768 → L2 norm, the exact
// sentence-transformers module stack of the checkpoint); (2) the recipe: fetch the pinned
// Hugging Face snapshot, verify the golden space, pack, sign.
//
// Targets are jvm + macosArm64 (Android consumes the jvm artifact): upstream publishes no iOS
// klibs for inference-bert yet, and a real-weights cartridge cannot out-wait that — the day
// they appear, adding iosArm64()/iosSimulatorArm64() here is the whole port.
//
// Tests need the model snapshot on disk (~92 MB, never in git):
//   downloaded once into ~/.cache/skainet/models/MongoDB_mdbr-leaf-ir (the same cache
//   BertEmbeddingModel.fromHuggingFace uses), or point LEAF_MODEL_DIR somewhere else.
//   Without it the golden tests skip loudly instead of failing.
plugins {
    alias(libs.plugins.kotlinMultiplatform)
    id("sk.ainet.cartridge.blueprint")
    // Hosts consume the cartridge API from ~/.m2, never as a nested source clone:
    //   ./gradlew :blueprints:embedding-leaf-skainet:publishToMavenLocal
    `maven-publish`
}

kotlin {
    explicitApi()

    jvm { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_21) } }
    jvmToolchain(21)
    macosArm64()

    sourceSets {
        commonMain.dependencies {
            // The graduated embedding/v1 contract — see profiles/embedding-v1.
            api(project(":profiles:embedding-v1"))

            // The BOM is api() because llm-api below is: an implementation-scoped platform
            // would not reach consumers' metadata, leaving the api dependency versionless.
            api(project.dependencies.platform("sk.ainet.transformers:skainet-transformers-bom:0.62.2"))
            // llm-api's EmbeddingModel SPI is api(): the engine implements it alongside
            // embedding/v1, which is the convergence the profile was designed around.
            api("sk.ainet.transformers:skainet-transformers-api")
            implementation("sk.ainet.transformers:skainet-transformers-inference-bert")

            implementation(project.dependencies.platform("sk.ainet:skainet-bom:0.61.0"))
            implementation("sk.ainet.core:skainet-lang-core")
            implementation("sk.ainet.core:skainet-backend-cpu")
            implementation("sk.ainet.core:skainet-io-core")
            implementation("sk.ainet.core:skainet-io-safetensors")

            implementation(libs.kotlinx.coroutines.core)
        }
        commonTest.dependencies {
            implementation(libs.kotlin.test)
            implementation(libs.kotlinx.serialization.json)
        }
    }
}

tasks.withType<org.jetbrains.kotlin.gradle.targets.jvm.tasks.KotlinJvmTest>().configureEach {
    jvmArgs("--add-modules", "jdk.incubator.vector")
    environment("LEAF_MODEL_DIR", providers.gradleProperty("leafModelDir").orNull ?: "")
    environment("REGENERATE_GOLDENS", providers.gradleProperty("regenerateGoldens").orNull ?: "")
}
