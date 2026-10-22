import sk.ainet.cartridge.blueprint.gradle.Conv1dIm2colTask
import sk.ainet.cartridge.blueprint.gradle.IreeCompileTask
import sk.ainet.cartridge.blueprint.gradle.IreeExportParametersTask
import sk.ainet.cartridge.blueprint.gradle.Int8WeightsTask
import sk.ainet.cartridge.blueprint.steps.Int8WeightsRewrite

// Blueprint: streaming speech-to-text with Moonshine v2 (tiny, en + de), IREE on Vulkan or CPU.
//
// (1) The cartridge's Kotlin API — a Kotlin Multiplatform library with NO model in it: the streaming contract, the
// pack locator and language selection in commonMain; a runtime binding where SKaiNET-transformers ships the
// streaming runtime — today Android (`IreeMoonshineStream`).
// (2) The recipe: the tasks below are the `steps` of blueprint.json, per flavor (language), registered as named
// products for `materializeCartridge`.
//
//   ./gradlew :blueprints:asr-moonshine-v2-streaming-iree:materializeCartridge -Pprofile=profiles/<yours>.json
plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.kotlinSerialization)
    id("sk.ainet.cartridge.blueprint")
}

/**
 * The SKaiNET-transformers runtime that ships `IreeMoonshineStream` (PR #450, first released with 0.56.2). To try an
 * unreleased runtime, build with `-PmoonshineRuntimeVersion=<x>-SNAPSHOT -PuseMavenLocal=true` against a snapshot
 * published from its branch (`:llm-runtime:iree-android:publishToMavenLocal -PVERSION_NAME=<x>-SNAPSHOT`).
 */
val moonshineRuntimeVersion = providers.gradleProperty("moonshineRuntimeVersion").getOrElse("0.63.1")

kotlin {
    explicitApi()

    android {
        namespace = "sk.ainet.cartridge.asr.moonshine"
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
            // IreeMoonshineStream + libskainet_moonshine_stream.so (both ABIs): SKaiNET-transformers PR #450, released since 0.56.2.
            api("sk.ainet.transformers:skainet-transformers-runtime-iree-android:$moonshineRuntimeVersion")
        }
    }
}

tasks.withType<Test>().configureEach {
    systemProperty("blueprint.dir", layout.projectDirectory.asFile.absolutePath)
}

// ------------------------------------------------------------------------------------------------------------
// The recipe (blueprint.json `steps`), once per flavor. `fetch`, `pack` and `sign` are the plugin's.
// ------------------------------------------------------------------------------------------------------------

/** The released exporter, straight from Maven Central: `MoonshineV2ExportCli` (Kotlin, SKaiNET-transformers). */
val exportTool: Configuration by configurations.creating {
    isCanBeConsumed = false
    attributes {
        attribute(Usage.USAGE_ATTRIBUTE, objects.named(Usage.JAVA_RUNTIME))
        attribute(org.jetbrains.kotlin.gradle.plugin.KotlinPlatformType.attribute, org.jetbrains.kotlin.gradle.plugin.KotlinPlatformType.jvm)
    }
}
dependencies {
    exportTool(project.dependencies.platform(libs.transformers.bom))
    exportTool(libs.transformers.inference.moonshine)
}

val work = layout.buildDirectory.dir("blueprint/work")
val selected = blueprint.target

// flavor → the source that holds its Hugging Face snapshot (blueprint.json flavors[].sources)
val flavors = mapOf("en" to "weights-en", "de" to "weights-de")

// The exporter's shapes, fixed by the runtime's compiled-in streaming geometry (blueprint.json steps[export].params).
val exportEnv = mapOf("MOONSHINE_FE_SAMPLES" to "21760", "MOONSHINE_ENC_FRAMES" to "64", "MOONSHINE_MAX_MEM" to "256")
// The incremental frontend: the same graph for 16 + 4 frames, which the runtime runs as the audio arrives when
// frontend_step.vmfb sits next to frontend.vmfb (blueprint.json steps[export].params.frontend_step_env).
val frontendStepEnv = mapOf("MOONSHINE_FE_SAMPLES" to "6400", "MOONSHINE_GRAPH" to "frontend")
val graphs = listOf("frontend", "encoder", "adapter", "prefill", "with_past", "frontend_step")

for ((language, source) in flavors) {
    val lang = language.replaceFirstChar(Char::uppercase)
    val snapshotDir = blueprint.sourceFile(source, "config.json").map { it.asFile.parentFile }
    val exportDir = work.map { it.dir("export/$language") }

    // export — Hugging Face snapshot → five StableHLO graphs (weights folded) + dec_embed.bin + vocab.bin.
    val export = tasks.register<JavaExec>("export$lang") {
        group = "cartridge blueprint steps"
        description = "export: trace the five streaming graphs of the $language checkpoint to StableHLO (Kotlin, SKaiNET-transformers)"
        classpath = exportTool
        mainClass.set("sk.ainet.models.moonshine.MoonshineV2ExportCliKt")
        maxHeapSize = "12g"
        inputs.files(blueprint.sourceFile(source, "config.json"), blueprint.sourceFile(source, "model.safetensors"), blueprint.sourceFile(source, "tokenizer.json")).withPathSensitivity(PathSensitivity.NONE)
        inputs.property("env", exportEnv)
        outputs.dir(exportDir)
        environment(exportEnv)
        doFirst {
            environment("MOONSHINE_MODEL_DIR", snapshotDir.get().absolutePath)
            environment("MOONSHINE_OUT_DIR", exportDir.get().asFile.apply { mkdirs() }.absolutePath)
        }
    }

    // export — the frontend once more, for the incremental frontend's 6400-sample input.
    val exportStepDir = work.map { it.dir("export-frontend-step/$language") }
    val exportStep = tasks.register<JavaExec>("exportFrontendStep$lang") {
        group = "cartridge blueprint steps"
        description = "export: the $language frontend for the incremental frontend's 16 + 4 frames (Kotlin, SKaiNET-transformers)"
        classpath = exportTool
        mainClass.set("sk.ainet.models.moonshine.MoonshineV2ExportCliKt")
        maxHeapSize = "12g"
        inputs.files(blueprint.sourceFile(source, "config.json"), blueprint.sourceFile(source, "model.safetensors"), blueprint.sourceFile(source, "tokenizer.json")).withPathSensitivity(PathSensitivity.NONE)
        inputs.property("env", frontendStepEnv)
        outputs.dir(exportStepDir)
        environment(frontendStepEnv)
        doFirst {
            environment("MOONSHINE_MODEL_DIR", snapshotDir.get().absolutePath)
            environment("MOONSHINE_OUT_DIR", exportStepDir.get().asFile.apply { mkdirs() }.absolutePath)
        }
    }

    // convert — the decoder weights out of the graph into one parameter archive (identical for prefill and step).
    val parameters = tasks.register<IreeExportParametersTask>("parameters$lang") {
        group = "cartridge blueprint steps"
        image.set(selected.map { it.image("iree-tools") })
        dependsOn(export)
        input.set(exportDir.map { it.file("prefill.mlir") })
        output.set(work.map { it.file("parameters/$language/params.irpa") })
        scope.set("model")
    }

    // convert (targets with params.conv1d_im2col = "true") — the frontend's 1-D convolutions as im2col + matmul.
    val im2col = tasks.register<Conv1dIm2colTask>("im2colFrontend$lang") {
        group = "cartridge blueprint steps"
        dependsOn(export)
        input.set(exportDir.map { it.file("frontend.mlir") })
        output.set(work.map { it.file("convert/$language/frontend-im2col.mlir") })
    }
    val im2colStep = tasks.register<Conv1dIm2colTask>("im2colFrontendStep$lang") {
        group = "cartridge blueprint steps"
        dependsOn(exportStep)
        input.set(exportStepDir.map { it.file("frontend.mlir") })
        output.set(work.map { it.file("convert/$language/frontend_step-im2col.mlir") })
    }
    val useIm2col = selected.map { it.params["conv1d_im2col"] == "true" }
    // graphs whose large matrices go int8 (params.int8_weights = "encoder,adapter,…")
    val int8Graphs = selected.map { t -> t.params["int8_weights"].orEmpty().split(',').map(String::trim).filter(String::isNotEmpty).toSet() }

    // compile — each graph → .vmfb for the profile's target; the decoder graphs reference the archive.
    for (graph in graphs) {
        val g = graph.split('_').joinToString("") { it.replaceFirstChar(Char::uppercase) }
        val outName = if (graph == "with_past") "step" else graph
        val source = when (graph) {
            "frontend" -> useIm2col.flatMap { if (it) im2col.flatMap { t -> t.output } else exportDir.map { d -> d.file("frontend.mlir") } }
            "frontend_step" -> useIm2col.flatMap { if (it) im2colStep.flatMap { t -> t.output } else exportStepDir.map { d -> d.file("frontend.mlir") } }
            else -> exportDir.map { it.file("$graph.mlir") }
        }
        val sourceTasks = when (graph) {
            "frontend" -> useIm2col.map { if (it) listOf(im2col) else emptyList() }
            "frontend_step" -> useIm2col.map { if (it) listOf(im2colStep) else listOf(exportStep) }
            else -> provider { emptyList() }
        }

        // convert (graphs listed in params.int8_weights) — weight-only int8, dequantized in the graph.
        val int8 = tasks.register<Int8WeightsTask>("int8$lang$g") {
            group = "cartridge blueprint steps"
            dependsOn(export, sourceTasks)
            input.set(source)
            output.set(work.map { it.file("convert/$language/$graph-int8.mlir") })
        }
        val useInt8 = int8Graphs.map { graph in it }

        // A target may name its own compiler (params.toolchain): SL26xx boards need the Torq-fork compiler.
        val compile = tasks.register<IreeCompileTask>("compile$lang$g") {
            group = "cartridge blueprint steps"
            image.set(selected.map { it.image(it.params["toolchain"] ?: "iree-tools") })
            dependsOn(export, sourceTasks, useInt8.map { if (it) listOf(int8) else emptyList() })
            input.set(useInt8.flatMap { if (it) int8.flatMap { t -> t.output } else source })
            // without hoisting and const-eval, so the dequantization is not folded back into f32 constants
            extraArgs.set(useInt8.map { if (it) Int8WeightsRewrite.COMPILE_FLAGS else emptyList() })
            backend.set(selected.map { it.param("backend") })
            target.set(selected.map { it.param("iree_target") })
            // f32 decoder graphs read their weights from the archive; int8 ones carry them inline
            if (graph == "prefill" || graph == "with_past") parameterScope.set(useInt8.map { if (it) null else "model" })
            output.set(work.zip(selected) { w, t -> w.file("compile/${t.id}/$language/$outName.vmfb") })
        }
        blueprint.product("$language-${outName.replace('_', '-')}-vmfb", compile.flatMap { it.output }, flavor = language)
    }
    // The decoder's weight files: the embedding rows, and the archive unless both decoder graphs are int8.
    val archiveRead = int8Graphs.map { !("prefill" in it && "with_past" in it) }
    val weights = tasks.register<Sync>("weights$lang") {
        group = "cartridge blueprint steps"
        dependsOn(export, archiveRead.map { if (it) listOf(parameters) else emptyList() })
        from(exportDir.map { it.file("dec_embed.bin") })
        from(archiveRead.flatMap { if (it) parameters.flatMap { t -> t.output }.map { f -> listOf(f) } else provider { emptyList() } })
        into(work.map { it.dir("weights/$language") })
    }
    blueprint.product("$language-weights", weights.map { work.get().dir("weights/$language") }, flavor = language)
    blueprint.product("$language-vocab", export.map { exportDir.get().file("vocab.bin") }, flavor = language)
}

// build-runtime — the streaming JNI library for the target ABI, from the published Android runtime (AAR).
val runtimeAar: Configuration by configurations.creating { isCanBeConsumed = false; isTransitive = false }
dependencies { runtimeAar("sk.ainet.transformers:skainet-transformers-runtime-iree-android:$moonshineRuntimeVersion@aar") }

val extractRuntime = tasks.register<Copy>("extractRuntime") {
    group = "cartridge blueprint steps"
    description = "build-runtime: libskainet_moonshine_stream.so for the target ABI, from the released SKaiNET-transformers runtime"
    val abi = selected.map { it.abi }
    from(runtimeAar.elements.map { files -> files.map { zipTree(it) } }) {
        include { it.path == "jni/${abi.get()}/libskainet_moonshine_stream.so" }
        eachFile { path = name }
        includeEmptyDirs = false
    }
    into(work.zip(selected) { w, t -> w.dir("runtime/${t.abi}") })
}
blueprint.product("runtime-so", extractRuntime.map { work.get().file("runtime/${selected.get().abi}/libskainet_moonshine_stream.so") })
