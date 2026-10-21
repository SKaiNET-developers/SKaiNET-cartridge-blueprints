import sk.ainet.cartridge.blueprint.gradle.IreeCompileTask

// Blueprint: speech-to-text with Whisper tiny (en + de) over a fixed 4 s window, IREE on Vulkan or CPU.
//
// (1) The cartridge's Kotlin API — a Kotlin Multiplatform library with NO model in it: the batch contract, the pack
// locator, language selection, the detokenizer and the transcript tidy-up in commonMain; the Android binding over
// SKaiNET-transformers' IreeWhisperSession (androidMain) with SKaiNET-audio's Whisper log-mel frontend.
// (2) The recipe: the tasks below are the `steps` of blueprint.json, per flavor (language), registered as named
// products for `materializeCartridge`.
//
//   ./gradlew :blueprints:asr-whisper-iree:materializeCartridge -Pprofile=profiles/<yours>.json
plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.kotlinSerialization)
    id("sk.ainet.cartridge.blueprint")
}

/**
 * The SKaiNET-transformers release that ships `WhisperExportCli` (PR #510) and `IreeWhisperSession` (PR #512): 0.63.0,
 * against engine 0.62.0 — ahead of the catalog's `transformers`, which the other blueprints share. To try an unreleased
 * exporter, build with `-PwhisperExportVersion=<x>-SNAPSHOT -PuseMavenLocal=true` against a snapshot from its branch.
 */
val whisperExportVersion = providers.gradleProperty("whisperExportVersion").getOrElse("0.63.0")

/**
 * The SKaiNET-transformers runtime that ships `IreeWhisperSession` + `libskainet_whisper_session.so`. To try an
 * unreleased runtime, build with `-PwhisperRuntimeVersion=<x>-SNAPSHOT -PuseMavenLocal=true` against a snapshot
 * published from its branch (`:llm-runtime:iree-android:publishToMavenLocal -PVERSION_NAME=<x>-SNAPSHOT`).
 */
val whisperRuntimeVersion = providers.gradleProperty("whisperRuntimeVersion").getOrElse(whisperExportVersion)

kotlin {
    explicitApi()

    android {
        namespace = "sk.ainet.cartridge.asr.whisper"
        compileSdk = libs.versions.androidCompileSdk.get().toInt()
        minSdk = libs.versions.androidMinSdk.get().toInt()
        compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) }
    }
    jvm { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }
    linuxX64()
    linuxArm64()
    macosArm64()

    sourceSets {
        commonMain.dependencies {
            implementation(libs.kotlinx.serialization.json)
            api(libs.kotlinx.io.core)                             // kotlinx.io.files.Path is part of the public API (PackDir)
        }
        commonTest.dependencies {
            implementation(libs.kotlin.test)
        }
        androidMain.dependencies {
            // IreeWhisperSession + libskainet_whisper_session.so (both ABIs)
            api("sk.ainet.transformers:skainet-transformers-runtime-iree-android:$whisperRuntimeVersion")
            // Whisper's log-mel frontend (MelConfig.whisper, Slaney filterbank)
            implementation(libs.audio.mel)
        }
    }
}

tasks.withType<Test>().configureEach {
    systemProperty("blueprint.dir", layout.projectDirectory.asFile.absolutePath)
}

// ------------------------------------------------------------------------------------------------------------
// The recipe (blueprint.json `steps`), once per flavor. `fetch`, `pack` and `sign` are the plugin's.
// ------------------------------------------------------------------------------------------------------------

/** The released exporter, straight from Maven Central: `WhisperExportCli` (Kotlin, SKaiNET-transformers). */
val exportTool: Configuration by configurations.creating {
    isCanBeConsumed = false
    attributes {
        attribute(Usage.USAGE_ATTRIBUTE, objects.named(Usage.JAVA_RUNTIME))
        attribute(org.jetbrains.kotlin.gradle.plugin.KotlinPlatformType.attribute, org.jetbrains.kotlin.gradle.plugin.KotlinPlatformType.jvm)
    }
}
dependencies {
    exportTool(project.dependencies.platform("sk.ainet.transformers:skainet-transformers-bom:$whisperExportVersion"))
    exportTool(libs.transformers.inference.whisper)
}

val work = layout.buildDirectory.dir("blueprint/work")
val selected = blueprint.target

// flavor → (the source that holds its Hugging Face snapshot, the manifest's prompt language)
val flavors = mapOf("en" to ("weights-en" to "en"), "de" to ("weights-de" to "de"))

// The exporter's shapes (blueprint.json steps[export].params): a 4 s encoder window and a 48-position KV cache.
val exportEnv = mapOf("WHISPER_AUDIO_CTX" to "200", "WHISPER_MAX_POSITIONS" to "48")
val graphs = listOf("encoder", "prefill", "step")

for ((flavor, spec) in flavors) {
    val (source, language) = spec
    val f = flavor.replaceFirstChar(Char::uppercase)
    val snapshotDir = blueprint.sourceFile(source, "config.json").map { it.asFile.parentFile }
    val exportDir = work.map { it.dir("export/$flavor") }

    // export — Hugging Face snapshot → three StableHLO graphs, one shared params.irpa, the contract manifest.
    val export = tasks.register<JavaExec>("export$f") {
        group = "cartridge blueprint steps"
        description = "export: trace encoder, prefill and step of the $flavor checkpoint to StableHLO (Kotlin, SKaiNET-transformers)"
        classpath = exportTool
        mainClass.set("sk.ainet.models.whisper.WhisperExportCliKt")
        maxHeapSize = "8g"
        inputs.files(blueprint.sourceFile(source, "config.json"), blueprint.sourceFile(source, "model.safetensors")).withPathSensitivity(PathSensitivity.NONE)
        inputs.property("env", exportEnv + ("WHISPER_LANGUAGE" to language))
        outputs.dir(exportDir)
        environment(exportEnv)
        environment("WHISPER_LANGUAGE", language)
        doFirst {
            environment("WHISPER_MODEL_DIR", snapshotDir.get().absolutePath)
            environment("WHISPER_OUT_DIR", exportDir.get().asFile.apply { mkdirs() }.absolutePath)
        }
    }

    // compile — each graph → .vmfb; prefill and step keep their parameter references. The encoder follows the target's
    // backend; prefill and step are compiled for the CPU of the target's ABI on every target (`decoder_iree_target`):
    // the step graph does not compile for Vulkan with IREE 3.11, and its M = 1 matrix-vector products cost about the
    // same on both.
    for (graph in graphs) {
        val g = graph.replaceFirstChar(Char::uppercase)
        val onDecoder = graph != "encoder"
        val compile = tasks.register<IreeCompileTask>("compile$f$g") {
            group = "cartridge blueprint steps"
            image.set(selected.map { it.image("iree-tools") })
            dependsOn(export)
            input.set(exportDir.map { it.file("whisper_$graph.mlir") })
            backend.set(selected.map { if (onDecoder) "cpu" else it.param("backend") })
            target.set(selected.map { it.param(if (onDecoder) "decoder_iree_target" else "iree_target") })
            output.set(work.zip(selected) { w, t -> w.file("compile/${t.id}/$flavor/$graph.vmfb") })
        }
        blueprint.product("$flavor-$graph-vmfb", compile.flatMap { it.output }, flavor = flavor)
    }
    blueprint.product("$flavor-params", export.map { exportDir.get().file("params.irpa") }, flavor = flavor)
    blueprint.product("$flavor-contract", export.map { exportDir.get().file("manifest.json") }, flavor = flavor)
    blueprint.product("$flavor-tokenizer", blueprint.sourceFile(source, "tokenizer.json"), flavor = flavor)
}

// build-runtime — the JNI library for the target ABI, from the published Android runtime (AAR).
val runtimeAar: Configuration by configurations.creating { isCanBeConsumed = false; isTransitive = false }
dependencies { runtimeAar("sk.ainet.transformers:skainet-transformers-runtime-iree-android:$whisperRuntimeVersion@aar") }

val extractRuntime = tasks.register<Copy>("extractRuntime") {
    group = "cartridge blueprint steps"
    description = "build-runtime: libskainet_whisper_session.so for the target ABI, from the released SKaiNET-transformers runtime"
    val abi = selected.map { it.abi }
    from(runtimeAar.elements.map { files -> files.map { zipTree(it) } }) {
        include { it.path == "jni/${abi.get()}/libskainet_whisper_session.so" }
        eachFile { path = name }
        includeEmptyDirs = false
    }
    into(work.zip(selected) { w, t -> w.dir("runtime/${t.abi}") })
}
blueprint.product("runtime-so", extractRuntime.map { work.get().file("runtime/${selected.get().abi}/libskainet_whisper_session.so") })
