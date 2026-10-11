import sk.ainet.cartridge.blueprint.gradle.HostGatherTask
import sk.ainet.cartridge.blueprint.gradle.IreeCompileTask
import sk.ainet.cartridge.blueprint.gradle.IreeConvertParametersTask

// Blueprint: function-calling NLU with Qwen instruct models (IREE, Vulkan or CPU), one flavor per checkpoint.
//
// Two things live here. (1) The cartridge's Kotlin API — a Kotlin Multiplatform library with NO model in it:
// contract, tool catalog, prompt and the whole resolve loop in commonMain, a thin runtime binding per platform.
// (2) The recipe: the tasks below are the `steps` of blueprint.json, once per flavor, registered as named products
// for `materializeCartridge`.
//
//   ./gradlew :blueprints:nlu-qwen-iree:materializeCartridge -Pprofile=profiles/<yours>.json
plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.kotlinSerialization)
    id("sk.ainet.cartridge.blueprint")
}

kotlin {
    explicitApi()

    // Same target set as the other blueprints. Every target compiles the full API and the engine; a *runtime
    // binding* exists where SKaiNET-transformers ships the KV-session runtime — today: Android.
    android {
        namespace = "sk.ainet.cartridge.nlu.qwen"
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
            implementation(libs.transformers.agent)               // ChatMessage, ToolDefinition, Qwen chat templates, ToolCallParser
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
// The recipe (blueprint.json `steps`), once per flavor. `fetch`, `pack` and `sign` are the plugin's.
// ------------------------------------------------------------------------------------------------------------

/** The released exporter, straight from Maven Central: `QwenExportCli` (Kotlin, SKaiNET-transformers). */
val exportTool: Configuration by configurations.creating {
    isCanBeConsumed = false
    attributes {
        attribute(Usage.USAGE_ATTRIBUTE, objects.named(Usage.JAVA_RUNTIME))
        attribute(org.jetbrains.kotlin.gradle.plugin.KotlinPlatformType.attribute, org.jetbrains.kotlin.gradle.plugin.KotlinPlatformType.jvm)
    }
}
dependencies {
    exportTool(libs.transformers.inference.qwen)
    exportTool(libs.skainet.backend.cpu)
}

val work = layout.buildDirectory.dir("blueprint/work")
val selected = blueprint.target

// flavor → the source that holds its GGUF (blueprint.json flavors[].sources) and the file in it
val flavors = mapOf(
    "qwen3-600m" to ("weights-qwen3-600m" to "Qwen3-0.6B-Q8_0.gguf"),
)

// the three graphs of the qwen-kv-v1 contract, by the file stem the exporter writes
val graphs = listOf("with-past", "prefill-with-past", "prefill-at")

for ((flavor, src) in flavors) {
    val (source, file) = src
    val gguf = blueprint.sourceFile(source, file)
    val flavorTask = flavor.split('-').joinToString("") { it.replaceFirstChar(Char::uppercase) }
    val exportDir = work.map { it.dir("export/$flavor") }

    // export — GGUF (Q8_0) → the three StableHLO modules + their external parameters (bf16) + manifest.json, one run.
    val export = tasks.register<JavaExec>("export$flavorTask") {
        group = "cartridge blueprint steps"
        description = "export: trace the three qwen-kv-v1 graphs of $flavor to StableHLO (Kotlin, SKaiNET-transformers)"
        classpath = exportTool
        mainClass.set("sk.ainet.models.qwen.QwenExportCliKt")
        maxHeapSize = "16g"
        inputs.file(gguf).withPathSensitivity(PathSensitivity.NONE)
        val env = mapOf("QWEN_GRAPH" to "all", "QWEN_SEQ" to "1024", "QWEN_CHUNK" to "32", "QWEN_DTYPE" to "bf16", "QWEN_HOST_GATHER" to "0")
        inputs.property("env", env)
        outputs.dir(exportDir)
        environment(env)
        doFirst {
            environment("QWEN_GGUF", gguf.get().asFile.absolutePath)
            environment("QWEN_OUT_DIR", exportDir.get().asFile.apply { mkdirs() }.absolutePath)
        }
    }
    blueprint.product("$flavor-manifest", export.map { exportDir.get().file("manifest.json") }, flavor = flavor)

    for (name in graphs) {
        val task = flavorTask + name.split('-').joinToString("") { it.replaceFirstChar(Char::uppercase) }

        // convert — move the token-embedding lookup to the host; numerics unchanged.
        val hostGather = tasks.register<HostGatherTask>("hostGather$task") {
            group = "cartridge blueprint steps"
            // Explicit: a provider that is not one of the exporter's declared output *properties* carries no task dependency.
            dependsOn(export)
            input.set(exportDir.map { it.file("qwen-$name.mlir") })
            output.set(work.map { it.file("hostgather/$flavor/qwen-$name.mlir") })
        }

        // convert — external parameters: safetensors → IREE parameter archive. Format change only.
        val parameters = tasks.register<IreeConvertParametersTask>("parameters$task") {
            group = "cartridge blueprint steps"
            image.set(selected.map { it.image("iree-tools") })
            dependsOn(export)
            input.set(exportDir.map { it.file("qwen-$name.safetensors") })
            output.set(work.map { it.file("parameters/$flavor/qwen-$name.irpa") })
            scope.set("model")
        }

        // compile — StableHLO → .vmfb for the profile's target.
        val compile = tasks.register<IreeCompileTask>("compile$task") {
            group = "cartridge blueprint steps"
            image.set(selected.map { it.image("iree-tools") })
            input.set(hostGather.flatMap { it.output })
            backend.set(selected.map { it.param("backend") })
            target.set(selected.map { it.param("iree_target") })
            output.set(work.zip(selected) { w, t -> w.file("compile/${t.id}/$flavor/qwen-$name.vmfb") })
        }

        blueprint.product("$flavor-$name-irpa", parameters.flatMap { it.output }, flavor = flavor)
        blueprint.product("$flavor-$name-vmfb", compile.flatMap { it.output }, flavor = flavor)
    }
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
