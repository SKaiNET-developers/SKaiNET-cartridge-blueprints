import sk.ainet.cartridge.blueprint.gradle.IreeCompileTask
import sk.ainet.cartridge.blueprint.gradle.TorqBf16RetypeTask
import java.security.MessageDigest

// Blueprint: text generation with LFM2.5-350M, one decode step per call on the Synaptics Torq NPU.
//
// (1) The cartridge's Kotlin API, with NO model in it: the decode step's port table, the host-side controls (RoPE,
// attention and write masks), the bf16 embedding rows and a greedy decoder over an abstract DecodeStepRunner. The
// runner that calls the Torq runtime belongs to whoever materializes the cartridge.
// (2) The recipe: the tasks below are the `steps` of blueprint.json, registered as named products for
// `materializeCartridge`.
//
//   ./gradlew :blueprints:generator-lfm25-350m-torq:materializeCartridge -Pprofile=profiles/<yours>.json
plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.kotlinSerialization)
    id("sk.ainet.cartridge.blueprint")
}

/**
 * The SKaiNET-transformers release whose `Lfm2DecodeStepExportCli` the export step runs. To export with an
 * unreleased exporter, pass `-Plfm2TransformersCheckout=/path/to/SKaiNET-transformers` (the root settings substitute
 * the module with the checkout's project).
 */
val lfm2TorqTransformersVersion = providers.gradleProperty("lfm2TorqTransformersVersion").getOrElse("0.64.2")

kotlin {
    explicitApi()
    jvm { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }
    linuxX64()
    linuxArm64()

    sourceSets {
        commonTest.dependencies {
            implementation(libs.kotlin.test)
        }
        jvmTest.dependencies {
            implementation(libs.kotlinx.serialization.json)
        }
    }
}

tasks.withType<Test>().configureEach {
    systemProperty("blueprint.dir", layout.projectDirectory.asFile.absolutePath)
}

// ------------------------------------------------------------------------------------------------------------
// The recipe (blueprint.json `steps`). `fetch`, `pack` and `sign` are the plugin's.
// ------------------------------------------------------------------------------------------------------------

/** The exporter: `Lfm2DecodeStepExportCli` (Kotlin, SKaiNET-transformers), resolved only when the export runs. */
val exportTool: Configuration by configurations.creating {
    isCanBeConsumed = false
    attributes {
        attribute(Usage.USAGE_ATTRIBUTE, objects.named(Usage.JAVA_RUNTIME))
        attribute(org.jetbrains.kotlin.gradle.plugin.KotlinPlatformType.attribute, org.jetbrains.kotlin.gradle.plugin.KotlinPlatformType.jvm)
    }
}
dependencies {
    exportTool(project.dependencies.platform("sk.ainet.transformers:skainet-transformers-bom:$lfm2TorqTransformersVersion"))
    exportTool("sk.ainet.transformers:skainet-transformers-inference-lfm2:$lfm2TorqTransformersVersion")
}

val work = layout.buildDirectory.dir("blueprint/work")
val selected = blueprint.target
val exportDir = work.map { it.dir("export") }
val config = blueprint.sourceFile("weights", "config.json")
val checkpoint = blueprint.sourceFile("weights", "model.safetensors")

fun sha256(file: File): String = file.inputStream().buffered().use { input ->
    val digest = MessageDigest.getInstance("SHA-256")
    val buffer = ByteArray(1 shl 20)
    while (true) {
        val n = input.read(buffer)
        if (n < 0) break
        digest.update(buffer, 0, n)
    }
    digest.digest().joinToString("") { "%02x".format(it) }
}

// export: the verified checkpoint → decode_step.mlir (packed block-quant matmuls), io.txt, embed.bf16.bin, export.json.
val export = tasks.register<JavaExec>("exportDecodeStep") {
    group = "cartridge blueprint steps"
    description = "export: trace the LFM2.5 decode step with Q8_0 projections and a Q4_0 head (Kotlin, SKaiNET-transformers)"
    classpath = exportTool
    mainClass.set("sk.ainet.models.lfm2.export.Lfm2DecodeStepExportCli")
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
    maxHeapSize = "8g"
    inputs.files(config, checkpoint).withPathSensitivity(PathSensitivity.NONE)
    val params = selected.map { t -> listOf("capacity", "weights", "head", "conv", "attention").associateWith { t.param(it) } }
    inputs.property("params", params)
    outputs.dir(exportDir)
    doFirst {
        val out = exportDir.get().asFile
        out.deleteRecursively()
        val p = params.get()
        args(
            "--config", config.get().asFile.absolutePath, "--config-sha256", sha256(config.get().asFile),
            "--checkpoint", checkpoint.get().asFile.absolutePath, "--checkpoint-sha256", sha256(checkpoint.get().asFile),
            "--capacity", p.getValue("capacity"), "--weights", p.getValue("weights"), "--head", p.getValue("head"),
            "--conv", p.getValue("conv"), "--attention", p.getValue("attention"), "--output", out.absolutePath,
        )
    }
}

// convert: every f32 to bf16 (Torq's f32 host fallback returns NaN).
val retype = tasks.register<TorqBf16RetypeTask>("torqBf16Retype") {
    group = "cartridge blueprint steps"
    dependsOn(export)
    input.set(exportDir.map { it.file("decode_step.mlir") })
    output.set(work.map { it.file("convert/decode_step-bf16.mlir") })
}

// compile: one Torq dispatch, the block-quant matmuls on the mixed-quant kernel.
val compile = tasks.register<IreeCompileTask>("compileDecodeStep") {
    group = "cartridge blueprint steps"
    image.set(selected.map { it.image(it.param("toolchain")) })
    input.set(retype.flatMap { it.output })
    backend.set("torq-stablehlo")
    target.set(selected.map { it.params["torq_hw"] ?: "SL2610" })
    output.set(work.zip(selected) { w, t -> w.file("compile/${t.id}/decode_step.vmfb") })
}

blueprint.product("decode-step-vmfb", compile.flatMap { it.output })
blueprint.product("io-table", export.map { exportDir.get().file("io.txt") })
blueprint.product("export-manifest", export.map { exportDir.get().file("export.json") })
blueprint.product("embedding-rows", export.map { exportDir.get().file("embed.bf16.bin") })
for (file in listOf("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "config.json")) {
    blueprint.product("tokenizer-" + file.substringBefore('.').replace('_', '-'), blueprint.sourceFile("weights", file))
}
blueprint.product("model-license", blueprint.sourceFile("weights", "LICENSE"))
