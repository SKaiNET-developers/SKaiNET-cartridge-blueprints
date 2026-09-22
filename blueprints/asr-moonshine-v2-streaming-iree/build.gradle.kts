import sk.ainet.cartridge.blueprint.gradle.IreeCompileTask
import sk.ainet.cartridge.blueprint.gradle.IreeExportParametersTask

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
 * The SKaiNET-transformers runtime that ships `IreeMoonshineStream` (PR #450, released with 0.56.2). Until that
 * release is on Maven Central, build with `-PmoonshineRuntimeVersion=0.56.2-SNAPSHOT -PuseMavenLocal=true` against a
 * snapshot published from that branch (`:llm-runtime:iree-android:publishToMavenLocal -PVERSION_NAME=0.56.2-SNAPSHOT`).
 */
val moonshineRuntimeVersion = providers.gradleProperty("moonshineRuntimeVersion").getOrElse("0.56.2")

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
            // IreeMoonshineStream + libskainet_moonshine_stream.so (both ABIs): SKaiNET-transformers PR #450, released with 0.56.2.
            // UNTIL 0.56.2 IS ON MAVEN CENTRAL: the snapshot published to mavenLocal from that branch.
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
val graphs = listOf("frontend", "encoder", "adapter", "prefill", "with_past")

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

    // convert — the decoder weights out of the graph into one parameter archive (identical for prefill and step).
    val parameters = tasks.register<IreeExportParametersTask>("parameters$lang") {
        group = "cartridge blueprint steps"
        image.set(selected.map { it.image("iree-tools") })
        dependsOn(export)
        input.set(exportDir.map { it.file("prefill.mlir") })
        output.set(work.map { it.file("parameters/$language/params.irpa") })
        scope.set("model")
    }

    // compile — each graph → .vmfb for the profile's target; the decoder graphs reference the archive.
    for (graph in graphs) {
        val g = graph.split('_').joinToString("") { it.replaceFirstChar(Char::uppercase) }
        val outName = if (graph == "with_past") "step" else graph
        val compile = tasks.register<IreeCompileTask>("compile$lang$g") {
            group = "cartridge blueprint steps"
            image.set(selected.map { it.image("iree-tools") })
            dependsOn(export)
            input.set(exportDir.map { it.file("$graph.mlir") })
            backend.set(selected.map { it.param("backend") })
            target.set(selected.map { it.param("iree_target") })
            if (graph == "prefill" || graph == "with_past") parameterScope.set("model")
            output.set(work.zip(selected) { w, t -> w.file("compile/${t.id}/$language/$outName.vmfb") })
        }
        blueprint.product("$language-$outName-vmfb", compile.flatMap { it.output }, flavor = language)
    }
    blueprint.product("$language-params-irpa", parameters.flatMap { it.output }, flavor = language)
    blueprint.product("$language-dec-embed", export.map { exportDir.get().file("dec_embed.bin") }, flavor = language)
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
