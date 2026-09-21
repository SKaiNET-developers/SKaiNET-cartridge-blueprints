import sk.ainet.cartridge.blueprint.gradle.HostGatherTask
import sk.ainet.cartridge.blueprint.gradle.IreeCompileTask
import sk.ainet.cartridge.blueprint.gradle.IreeConvertParametersTask

// Blueprint: function-calling NLU with FunctionGemma 270M (IREE, Vulkan or CPU).
//
// Two things live here. (1) The cartridge's Kotlin API — a Kotlin Multiplatform library with NO model in it:
// contract, tool catalog, prompt and the whole resolve loop in commonMain, a thin runtime binding per platform.
// (2) The recipe: the tasks below are the `steps` of blueprint.json, registered as named products for
// `materializeCartridge`.
//
//   ./gradlew :blueprints:nlu-functiongemma-270m-iree:materializeCartridge -Pprofile=profiles/<yours>.json
plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.kotlinSerialization)
    id("sk.ainet.cartridge.blueprint")
}

kotlin {
    explicitApi()

    // Same target set as the Moonshine cartridge modules, plus Android. Every target compiles the full API and the
    // engine; a *runtime binding* exists where SKaiNET-transformers ships the KV-session runtime — today: Android.
    android {
        namespace = "sk.ainet.cartridge.nlu.functiongemma"
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
            implementation(project.dependencies.platform(libs.transformers.bom))
            implementation(libs.transformers.agent)               // ChatMessage, ToolDefinition, ToolCall
            implementation(libs.transformers.runtime.gemma.iree)  // FunctionGemmaOfficialChatTemplate + tool-call parser
            implementation(libs.kotlinx.serialization.json)
            api(libs.kotlinx.io.core)                             // kotlinx.io.files.Path is part of the public API (PackDir)
        }
        commonTest.dependencies {
            implementation(libs.kotlin.test)
        }
        androidMain.dependencies {
            implementation(project.dependencies.platform(libs.transformers.bom))
            implementation(libs.transformers.core)                // TokenizerFactory
            api(libs.transformers.runtime.iree.android)           // IreeKvSession + libskainet_iree_kv.so (both ABIs)
            implementation(libs.skainet.io.core)                  // AndroidRandomAccessSource
            implementation(libs.skainet.io.gguf)                  // StreamingGGUFReader
        }
    }
}

tasks.withType<Test>().configureEach {
    systemProperty("blueprint.dir", layout.projectDirectory.asFile.absolutePath)
}

// ------------------------------------------------------------------------------------------------------------
// The recipe (blueprint.json `steps`). `fetch`, `pack` and `sign` are the plugin's; these are the model's.
// ------------------------------------------------------------------------------------------------------------

/** The released exporter, straight from Maven Central: `FunctionGemmaExportCli` (Kotlin, SKaiNET-transformers). */
val exportTool: Configuration by configurations.creating {
    isCanBeConsumed = false
    attributes {
        attribute(Usage.USAGE_ATTRIBUTE, objects.named(Usage.JAVA_RUNTIME))
        attribute(org.jetbrains.kotlin.gradle.plugin.KotlinPlatformType.attribute, org.jetbrains.kotlin.gradle.plugin.KotlinPlatformType.jvm)
    }
}
dependencies {
    exportTool(libs.transformers.inference.functiongemma)
    exportTool(libs.skainet.backend.cpu)
}

val gguf = blueprint.sourceFile("weights", "functiongemma-270m-it-Q8_0.gguf")
val work = layout.buildDirectory.dir("blueprint/work")
val selected = blueprint.target

// name → the exporter's environment contract for that graph (blueprint.json steps[export].params.graphs)
val graphs = mapOf(
    "with-past" to mapOf("GEMMA_GRAPH" to "with_past"),
    "prefill-with-past" to mapOf("GEMMA_GRAPH" to "prefill_with_past", "GEMMA_CHUNK" to "32"),
    "prefill-at" to mapOf("GEMMA_GRAPH" to "prefill_at", "GEN_SEQ" to "1024"),
)

for ((name, env) in graphs) {
    val task = name.split('-').joinToString("") { it.replaceFirstChar(Char::uppercase) }
    val exportDir = work.map { it.dir("export/$name") }

    // export — GGUF (Q8_0) → StableHLO module + external parameters (bf16), one graph per run.
    val export = tasks.register<JavaExec>("export$task") {
        group = "cartridge blueprint steps"
        description = "export: trace the $name graph to StableHLO (Kotlin, SKaiNET-transformers)"
        classpath = exportTool
        mainClass.set("sk.ainet.models.functiongemma.FunctionGemmaExportCliKt")
        maxHeapSize = "8g"
        inputs.file(gguf).withPathSensitivity(PathSensitivity.NONE)
        inputs.property("env", env)
        outputs.dir(exportDir)
        environment(env)
        doFirst {
            environment("GEMMA_GGUF", gguf.get().asFile.absolutePath)
            environment("GEMMA_OUT_DIR", exportDir.get().asFile.apply { mkdirs() }.absolutePath)
        }
    }

    // convert — move the token-embedding lookup to the host; numerics unchanged.
    val hostGather = tasks.register<HostGatherTask>("hostGather$task") {
        group = "cartridge blueprint steps"
        // Explicit: a provider that is not one of the exporter's declared output *properties* carries no task dependency.
        dependsOn(export)
        input.set(exportDir.map { it.file("gemma-$name.mlir") })
        output.set(work.map { it.file("hostgather/gemma-$name.mlir") })
    }

    // convert — external parameters: safetensors → IREE parameter archive. Format change only.
    val parameters = tasks.register<IreeConvertParametersTask>("parameters$task") {
        group = "cartridge blueprint steps"
        image.set(selected.map { it.image("iree-tools") })
        dependsOn(export)
        input.set(exportDir.map { it.file("gemma-$name.safetensors") })
        output.set(work.map { it.file("parameters/gemma-$name.irpa") })
        scope.set("model")
    }

    // compile — StableHLO → .vmfb for the profile's target.
    val compile = tasks.register<IreeCompileTask>("compile$task") {
        group = "cartridge blueprint steps"
        image.set(selected.map { it.image("iree-tools") })
        input.set(hostGather.flatMap { it.output })
        backend.set(selected.map { it.param("backend") })
        target.set(selected.map { it.param("iree_target") })
        output.set(work.zip(selected) { w, t -> w.file("compile/${t.id}/gemma-$name.vmfb") })
    }

    blueprint.product("$name-irpa", parameters.flatMap { it.output })
    blueprint.product("$name-vmfb", compile.flatMap { it.output })
}

// build-runtime — the KV-session JNI library for the target ABI, from the published Android runtime (AAR).
val runtimeAar: Configuration by configurations.creating { isCanBeConsumed = false; isTransitive = false }
dependencies { runtimeAar("sk.ainet.transformers:skainet-transformers-runtime-iree-android:${libs.versions.transformers.get()}@aar") }

val extractRuntime = tasks.register<Copy>("extractRuntime") {
    group = "cartridge blueprint steps"
    description = "build-runtime: libskainet_iree_kv.so for the target ABI, from the released SKaiNET-transformers runtime"
    val abi = selected.map { it.abi }
    from(runtimeAar.elements.map { files -> files.map { zipTree(it) } }) {
        include { it.path == "jni/${abi.get()}/libskainet_iree_kv.so" }
        eachFile { path = name }
        includeEmptyDirs = false
    }
    into(work.zip(selected) { w, t -> w.dir("runtime/${t.abi}") })
}
blueprint.product("runtime-so", extractRuntime.map { work.get().file("runtime/${selected.get().abi}/libskainet_iree_kv.so") })
