import kotlinx.serialization.json.JsonObject
import org.gradle.api.DefaultTask
import org.gradle.api.file.DirectoryProperty
import org.gradle.api.file.RegularFileProperty
import org.gradle.api.tasks.*
import sk.ainet.cartridge.blueprint.core.CanonicalJson
import sk.ainet.cartridge.blueprint.core.DescriptorResolver
import sk.ainet.cartridge.blueprint.core.Documents
import sk.ainet.cartridge.blueprint.gradle.BlueprintValidateTask
import sk.ainet.cartridge.blueprint.gradle.MaterializeCartridgeTask
import java.io.File
import java.nio.file.Files
import java.nio.file.StandardOpenOption

plugins {
    alias(libs.plugins.kotlinJvm)
    application
    id("sk.ainet.cartridge.blueprint")
}

abstract class ResolveNativeDescriptors : DefaultTask() {
    @get:InputFile @get:PathSensitive(PathSensitivity.NONE) abstract val recipe: RegularFileProperty
    @get:InputFile @get:PathSensitive(PathSensitivity.NONE) abstract val previewProfile: RegularFileProperty
    @get:OutputDirectory abstract val destination: DirectoryProperty

    @TaskAction fun resolve() {
        val bp = Documents.blueprint(recipe.get().asFile)
        val pf = Documents.profile(previewProfile.get().asFile)
        Documents.requireMatch(bp, pf)
        // EffectiveLicense.compute is necessarily NOASSERTION when the recipe's
        // own license is NOASSERTION, whatever the model/input component licenses.
        require(bp.value.license == "NOASSERTION")
        val descriptor = DescriptorResolver.resolve(bp.value, pf.value, "NOASSERTION")
        val raw = Documents.json.encodeToString(JsonObject.serializer(), descriptor) + "\n"
        require(raw.all { it.code < 128 }) { "This bridge recipe requires ASCII descriptor content" }
        val compact = CanonicalJson.encode(descriptor)
        val output = destination.get().asFile
        require(!Files.isSymbolicLink(output.toPath()) && (!output.exists() || output.listFiles()?.isEmpty() == true)) {
            "Preserve previous descriptor output; use a fresh -PnativeRun"
        }
        output.mkdirs()
        Files.writeString(output.resolve("descriptor.json").toPath(), raw, StandardOpenOption.CREATE_NEW)
        Files.writeString(output.resolve("descriptor.canonical.json").toPath(), compact, StandardOpenOption.CREATE_NEW)
        Files.writeString(output.resolve("preview-status.json").toPath(),
            "{\"status\":\"unmeasured_descriptor_preview\",\"actual_plugin_resolver\":true}\n", StandardOpenOption.CREATE_NEW)
    }
}


abstract class GenerationExportArguments : org.gradle.process.CommandLineArgumentProvider {
    @get:InputFile abstract val config: RegularFileProperty
    @get:InputFile abstract val checkpoint: RegularFileProperty
    @get:Internal abstract val destination: DirectoryProperty
    override fun asArguments(): Iterable<String> = listOf(
        "--config", config.get().asFile.absolutePath,
        "--config-sha256", "720b43d6ddc2ed25be23eed355aefcf342434a176dedad23dbe0a5e3ac24bbb8",
        "--checkpoint", checkpoint.get().asFile.absolutePath,
        "--checkpoint-sha256", "1c9c77a4471a7f590f85240f74ed1fc26df7fbde88c3006724e2f93ca993ea4e",
        "--capacity", "64", "--rows-per-shard", "8192", "--output", destination.get().asFile.absolutePath)
}

group = "sk.ainet.cartridge.generator"
version = "0.1.0-SNAPSHOT"
val exportTool: Configuration by configurations.creating {
    isCanBeConsumed = false
    attributes {
        attribute(Usage.USAGE_ATTRIBUTE, objects.named(Usage.JAVA_RUNTIME))
        attribute(org.jetbrains.kotlin.gradle.plugin.KotlinPlatformType.attribute, org.jetbrains.kotlin.gradle.plugin.KotlinPlatformType.jvm)
    }
}
val transformersVersion = providers.gradleProperty("lfm2TransformersVersion").orElse("0.63.4-SNAPSHOT")
dependencies {
    implementation(project(":runtimes:lfm2-jvm"))
    exportTool("sk.ainet.transformers:skainet-transformers-inference-lfm2:${transformersVersion.get()}")
}
kotlin { jvmToolchain(21) }
application {
    mainClass.set("sk.ainet.cartridge.generator.lfm2.worker.MainKt")
    applicationDefaultJvmArgs = listOf("--enable-preview", "--add-modules", "jdk.incubator.vector")
}

val runId = providers.gradleProperty("gpuRun").orElse("001").get()
require(runId.matches(Regex("[A-Za-z0-9][A-Za-z0-9_-]*"))) { "gpuRun must be a simple fresh identifier" }
val run = layout.buildDirectory.dir("runs/$runId")
val recipeFile = layout.projectDirectory.file("blueprint.json")
val helper = layout.projectDirectory.file("scripts/prepare.py")
val wheels = layout.projectDirectory.file("runtime-wheels.json")
val sharedRuntime = project(":runtimes:lfm2-jvm")
val nativeSource = sharedRuntime.layout.projectDirectory.dir("native")
val nativeBuilder = sharedRuntime.layout.projectDirectory.file("scripts/build_native.py")
val settingsFile = providers.gradleProperty("lfm2GpuSettings").orElse(layout.projectDirectory.file("private-settings.json").asFile.absolutePath)
val compilerImage = providers.gradleProperty("nativeCompilerImage").orElse("skainet/iree-compiler:3.11.0")
val toolsCheckout = providers.gradleProperty("lfm2IreeToolsCheckout").orElse(layout.projectDirectory.dir("missing-iree-tools-checkout").asFile.absolutePath)
val compilerWrapper = toolsCheckout.map { File(it).resolve("docker/compiler/entrypoint.sh") }
val rangeAuditor = toolsCheckout.map { File(it).resolve("verify/verify_vulkan_ranges.py") }
val preview = run.map { it.file("profile/preview.json") }
val descriptors = run.map { it.dir("descriptors") }
val nativeOutput = run.map { it.dir("native") }
val exportRoot = run.map { it.dir("export") }
val exported = exportRoot.map { it.dir("bundle") }
// Compiler-policy iterations may reuse an immutable source export. The compiler
// rehashes its complete module/parameter/contract closure; this never imports a
// compiled binary or measurement acceptance.
val preparedExport = providers.gradleProperty("lfm2PreparedExport").orNull
val selectedExport = if (preparedExport == null) exported else layout.dir(providers.provider { File(preparedExport) })
val compiledRoot = run.map { it.dir("compile") }
val compiled = compiledRoot.map { it.dir("bundle") }
val execution = run.map { it.dir("execution") }
val installer = tasks.named<Sync>("installDist")
val workerLibraries = layout.dir(installer.map { it.destinationDir.resolve("lib") })
val workerLibrariesPath = installer.get().destinationDir.resolve("lib").absolutePath

val preparePreviewProfile = tasks.register<Exec>("preparePreviewProfile") {
    group = "cartridge blueprint"
    description = "Verifies original source and wheel pins and records explicit private Vulkan selection; this preview is unmeasured."
    inputs.files(recipeFile, helper, wheels, settingsFile)
    outputs.files(preview, run.map { it.file("profile/preview-status.json") })
    commandLine("python3", helper.asFile.absolutePath, "profile", "--settings", settingsFile.get(),
        "--blueprint", recipeFile.asFile.absolutePath, "--wheels", wheels.asFile.absolutePath,
        "--output", preview.get().asFile.absolutePath)
}
blueprint { profileFile.set(preview); outputDir.set(run.map { it.dir("materialized") }) }
tasks.named("blueprintFetch") { dependsOn(preparePreviewProfile) }
tasks.named("blueprintValidate") { dependsOn(preparePreviewProfile) }
val resolveNativeDescriptors = tasks.register<ResolveNativeDescriptors>("resolveNativeDescriptors") {
    group = "cartridge blueprint"; dependsOn(preparePreviewProfile, "blueprintValidate")
    this.recipe.set(recipeFile); previewProfile.set(preview); destination.set(descriptors)
}
val buildNativeRuntime = tasks.register<Exec>("buildNativeRuntime") {
    group = "cartridge blueprint"; dependsOn(resolveNativeDescriptors)
    inputs.file(nativeBuilder); inputs.dir(nativeSource); inputs.dir(descriptors); inputs.property("image", compilerImage)
    outputs.dir(nativeOutput)
    commandLine("python3", nativeBuilder.asFile.absolutePath, "--source", nativeSource.asFile.absolutePath,
        "--descriptors", descriptors.get().asFile.absolutePath, "--output", nativeOutput.get().asFile.absolutePath,
        "--image", compilerImage.get())
}
val exportGeneration = tasks.register<JavaExec>("exportGeneration") {
    group = "cartridge blueprint"; dependsOn("blueprintFetch")
    description = "Exports the actual complete cached generation graph through SKaiNET-transformers at context64."
    classpath = exportTool
    mainClass.set("sk.ainet.models.lfm2.export.Lfm2GenerationExportCli")
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector"); maxHeapSize = "8g"
    val model = blueprint.sourceFile("liquid-lfm25-350m", "model.safetensors")
    val modelConfig = blueprint.sourceFile("liquid-lfm25-350m", "config.json")
    inputs.files(model, modelConfig)
    // Gradle creates the parent, while the exporter requires a brand-new child.
    // Changed inputs preserve the old child and require a fresh gpuRun.
    outputs.dir(exportRoot)
    argumentProviders.add(objects.newInstance<GenerationExportArguments>().apply {
        this.config.set(modelConfig); checkpoint.set(model); destination.set(exported)
    })
}
val compileGeneration = tasks.register<Exec>("compileGeneration") {
    group = "cartridge blueprint"
    if (preparedExport == null) dependsOn(exportGeneration)
    description = "Source-mounted IREE-tools wrapper converts IRPA and compiles generic Vulkan with FP32 preserved."
    inputs.dir(selectedExport); inputs.files(compilerWrapper, rangeAuditor); inputs.file(layout.projectDirectory.file("scripts/compile_generation.py"))
    inputs.property("image", compilerImage)
    outputs.dir(compiledRoot)
    commandLine("python3", layout.projectDirectory.file("scripts/compile_generation.py").asFile.absolutePath,
        "--export", selectedExport.get().asFile.absolutePath, "--output", compiled.get().asFile.absolutePath,
        "--compiler-wrapper", compilerWrapper.get().absolutePath, "--range-auditor", rangeAuditor.get().absolutePath,
        "--image", compilerImage.get())
}
val prepareGpuExecution = tasks.register<Exec>("prepareGpuExecution") {
    group = "cartridge blueprint"; dependsOn(buildNativeRuntime, compileGeneration, installer, "blueprintFetch")
    description = "Stages an explicitly unqualified complete-GPU candidate with actual build products and no original-weight duplicate."
    inputs.files(helper,wheels,settingsFile); inputs.dir(workerLibraries); inputs.dir(compiled); inputs.dir(descriptors); inputs.dir(nativeOutput)
    outputs.dir(execution)
    commandLine("python3",helper.asFile.absolutePath,"prepare","--settings",settingsFile.get(),"--wheels",wheels.asFile.absolutePath,
        "--worker-libs",workerLibrariesPath,"--bundle",compiled.get().asFile.absolutePath,
        "--descriptor",descriptors.get().file("descriptor.json").asFile.absolutePath,
        "--canonical-descriptor",descriptors.get().file("descriptor.canonical.json").asFile.absolutePath,
        "--native-library",nativeOutput.get().file("liblfm2_cartridge.so").asFile.absolutePath,
        "--native-build-report",nativeOutput.get().file("liblfm2_cartridge.so.build.json").asFile.absolutePath,
        "--compiler-report",nativeOutput.get().file("compiler-container.json").asFile.absolutePath,
        "--native-license",nativeSource.file("LICENSE.upstream").asFile.absolutePath,"--output",execution.get().asFile.absolutePath)
}
blueprint {
    val prepared = prepareGpuExecution.flatMap { execution }
    product("gpu-module", prepared.map { it.file("artifacts/gpu/model.vmfb") })
    product("gpu-parameters", prepared.map { it.file("artifacts/gpu/model.irpa") })
    product("gpu-runtime-manifest", prepared.map { it.file("artifacts/gpu/runtime-manifest.json") })
    product("gpu-export-provenance", prepared.map { it.file("artifacts/gpu/source-export-manifest.json") })
    product("gpu-contract", prepared.map { it.file("artifacts/gpu/generation-contract.json") })
    product("gpu-compilation-report", prepared.map { it.file("artifacts/gpu/compilation-report.json") })
    product("gpu-parameter-log", prepared.map { it.file("artifacts/gpu/parameters.log") })
    product("gpu-compiler-log", prepared.map { it.file("artifacts/gpu/vulkan.log") })
    product("gpu-compiler-wrapper", prepared.map { it.file("artifacts/gpu/compiler-entrypoint.sh") })
    product("gpu-compiler-driver", prepared.map { it.file("artifacts/gpu/compile-driver.py") })
    product("gpu-range-audit", prepared.map { it.file("artifacts/gpu/vulkan-range-audit.json") })
    product("gpu-range-audit-log", prepared.map { it.file("artifacts/gpu/range-audit.log") })
    product("gpu-range-audit-dump", prepared.map { it.file("artifacts/gpu/vulkan-range-audit.json.all.txt") })
    product("gpu-range-audit-flatbuffer", prepared.map { it.file("artifacts/gpu/vulkan-range-audit.json.flatbuffer-json.json") })
    product("gpu-range-auditor", prepared.map { it.file("artifacts/gpu/verify_vulkan_ranges.py") })
    product("python-packages", prepared.map { it.dir("artifacts/python") })
    product("python-wheels", prepared.map { it.file("artifacts/python-wheels.json") })
    product("metadata", prepared.map { it.dir("artifacts/metadata") })
    product("worker-jars", prepared.map { it.dir("artifacts/jvm/lib") })
    product("worker-config", prepared.map { it.file("artifacts/runtime/worker.json") })
    product("native-runtime", prepared.map { it.file("artifacts/runtime/liblfm2_cartridge.so") })
    product("native-build-report", prepared.map { it.file("artifacts/runtime/native-build.json") })
    product("native-compiler-report", prepared.map { it.file("artifacts/runtime/compiler-container.json") })
    product("native-license", prepared.map { it.file("artifacts/runtime/LICENSE.upstream") })
}
val requireFullGpuQualification = tasks.register("requireFullGpuQualification") {
    group = "verification"
    description = "Fails closed until complete GPU target, frontend, C ABI and exact-artifact measurements have independent acceptance."
    doLast { throw GradleException("GPU materialization is pending complete GPU numerical/state/generation, frontend, C ABI and exact-artifact measurement qualification. Source export/compilation alone cannot authorize an accepted profile.") }
}
tasks.named<MaterializeCartridgeTask>("materializeCartridge") {
    dependsOn(requireFullGpuQualification)
    signingKeyFile.unset(); signingKeyFile.unsetConvention()
}
val validateRecipe by tasks.registering(BlueprintValidateTask::class) { blueprintFile.set(recipeFile) }
val testPackaging by tasks.registering(Exec::class) {
    commandLine("python3","-m","unittest","discover","-s",layout.projectDirectory.dir("scripts").asFile.absolutePath,"-p","test_*.py","-v")
}
tasks.named("check") { dependsOn(":runtimes:lfm2-jvm:check", validateRecipe,testPackaging) }
