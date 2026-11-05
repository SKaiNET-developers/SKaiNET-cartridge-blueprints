import kotlinx.serialization.json.JsonObject
import org.gradle.api.DefaultTask
import org.gradle.api.file.DirectoryProperty
import org.gradle.api.file.ConfigurableFileCollection
import org.gradle.api.file.RegularFileProperty
import org.gradle.api.tasks.InputFile
import org.gradle.api.tasks.Internal
import org.gradle.api.tasks.OutputDirectory
import org.gradle.api.tasks.PathSensitive
import org.gradle.api.tasks.PathSensitivity
import org.gradle.api.tasks.Sync
import org.gradle.api.tasks.TaskAction
import org.gradle.api.tasks.Classpath
import org.gradle.process.CommandLineArgumentProvider
import sk.ainet.cartridge.blueprint.core.CanonicalJson
import sk.ainet.cartridge.blueprint.core.DescriptorResolver
import sk.ainet.cartridge.blueprint.core.Documents
import sk.ainet.cartridge.blueprint.gradle.MaterializeCartridgeTask
import sk.ainet.cartridge.blueprint.gradle.BlueprintValidateTask
import java.time.Duration
import java.nio.file.Files
import java.nio.file.StandardOpenOption

plugins {
    alias(libs.plugins.kotlinJvm)
    `java-library`
    application
    `maven-publish`
    id("sk.ainet.cartridge.blueprint")
}

// These typed tasks use the actual pinned plugin API and serializer. Their task
// actions capture no Project/script object and support Gradle's configuration cache.
abstract class ResolveNativeDescriptors : DefaultTask() {
    @get:InputFile @get:PathSensitive(PathSensitivity.NONE) abstract val recipe: RegularFileProperty
    @get:InputFile @get:PathSensitive(PathSensitivity.NONE) abstract val previewProfile: RegularFileProperty
    @get:OutputDirectory abstract val destination: DirectoryProperty

    @TaskAction fun resolve() {
        val bp = Documents.blueprint(recipe.get().asFile)
        val pf = Documents.profile(previewProfile.get().asFile)
        Documents.requireMatch(bp, pf)
        // The pinned plugin computes the effective license from the recipe license and
        // each output's upstream source/input licenses: first-occurrence order, non-SPDX
        // tokens rewritten to LicenseRef- ids. This literal must match the Materializer's
        // own computation and EFFECTIVE_LICENSE in scripts/prepare.py.
        require(bp.value.license == "MIT")
        val effectiveLicense = "MIT AND LicenseRef-LFM-Open-License-1.0 AND Apache-2.0 AND " +
            "LicenseRef-GPL-2.0-only-WITH-Classpath-exception-2.0"
        val descriptor = DescriptorResolver.resolve(bp.value, pf.value, effectiveLicense)
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

abstract class GuardFreshMaterialization : DefaultTask() {
    @get:InputFile @get:PathSensitive(PathSensitivity.NONE) abstract val recipe: RegularFileProperty
    @get:InputFile @get:PathSensitive(PathSensitivity.NONE) abstract val acceptedProfile: RegularFileProperty
    @get:Internal abstract val parentDirectory: DirectoryProperty

    @TaskAction fun guard() {
        val bp = Documents.blueprint(recipe.get().asFile)
        val pf = Documents.profile(acceptedProfile.get().asFile)
        val parent = parentDirectory.get().asFile
        val destination = parent.resolve(DescriptorResolver.cartridgeId(bp.value, pf.value))
        require(!destination.exists() && !Files.isSymbolicLink(destination.toPath())) {
            "The upstream materializer deletes existing packs; preserve this run and choose a fresh -PnativeRun"
        }
        parent.mkdirs()
        Files.writeString(parent.resolve(".materialization-started").toPath(),
            "Fresh destination claimed for the actual unsigned materializer; preserve failed attempts.\n",
            StandardOpenOption.CREATE_NEW)
    }
}

group = "sk.ainet.cartridge.generator"
version = "0.1.0-SNAPSHOT"

dependencies { implementation(project(":runtimes:lfm2-jvm")) }
kotlin { jvmToolchain(21) }
application {
    mainClass.set("sk.ainet.cartridge.generator.lfm2.worker.MainKt")
    applicationDefaultJvmArgs = listOf("--enable-preview", "--add-modules", "jdk.incubator.vector")
}

val runId = providers.gradleProperty("nativeRun").orElse("001").get()
require(runId.matches(Regex("[A-Za-z0-9][A-Za-z0-9_-]*"))) { "nativeRun must be a simple fresh identifier" }
val runDirectory = layout.buildDirectory.dir("runs/$runId")
val recipeFile = layout.projectDirectory.file("blueprint.json")
val helper = layout.projectDirectory.file("scripts/prepare.py")
val sharedRuntime = project(":runtimes:lfm2-jvm")
val builder = sharedRuntime.layout.projectDirectory.file("scripts/build_native.py")
val modelDirectory = providers.gradleProperty("lfm2ModelDir").orElse(
    layout.buildDirectory.dir("source-model").get().asFile.absolutePath
).get()
val previewProfileFile = runDirectory.map { it.file("profile/preview.json") }
val acceptedProfileFile = runDirectory.map { it.file("profile/accepted.json") }
val descriptors = runDirectory.map { it.dir("descriptors") }
val nativeOutput = runDirectory.map { it.dir("native") }
val execution = runDirectory.map { it.dir("execution") }
// Acceptance may intentionally consume an already frozen source-built execution
// while unrelated development continues. The independent validator rehashes the
// complete staged closure; this mode never substitutes a newly rebuilt JAR.
val usePreparedExecution = providers.gradleProperty("usePreparedNativeExecution").orNull == "true"
val acceptance = runDirectory.map { it.file("measurements/validation.json") }
val materializedParent = runDirectory.map { it.dir("materialized") }
val materializedPack = materializedParent.map { it.dir("generator-lfm25-350m-skainet-linux-aarch64") }
val verifiedMaterialization = runDirectory.map { it.file("measurements/materialized-closure.json") }
val workload = layout.projectDirectory.file("samples/native-abi-workload.json")
val nativeSource = sharedRuntime.layout.projectDirectory.dir("native")
val benchmarkDirectory = providers.gradleProperty("nativeBenchmarkDir").orElse(
    runDirectory.get().dir("benchmark").asFile.absolutePath
).get()
val benchmarkValidator = layout.projectDirectory.file("scripts/validate_native_benchmark.py")
val installWorker = tasks.named<Sync>("installDist")
val workerLibraries = layout.dir(installWorker.map { it.destinationDir.resolve("lib") })
val workerLibrariesPath = installWorker.get().destinationDir.resolve("lib").absolutePath

val preparePreviewProfile = tasks.register<Exec>("preparePreviewProfile") {
    group = "cartridge blueprint"
    description = "Verifies seven pinned sources and creates an explicitly unmeasured descriptor-preview profile."
    inputs.files(recipeFile, helper)
    inputs.dir(modelDirectory)
    outputs.files(previewProfileFile, runDirectory.map { it.file("profile/preview-status.json") })
    commandLine("python3", helper.asFile.absolutePath, "profile", "--blueprint", recipeFile.asFile.absolutePath,
        "--model-dir", modelDirectory, "--output", previewProfileFile.get().asFile.absolutePath)
}

blueprint {
    // Keep fetch/target selection on the preview: routing fetch through accepted
    // measurements would create fetch -> execution -> measurement -> fetch cycles.
    profileFile.set(previewProfileFile)
    outputDir.set(materializedParent)
    product("model-weights", sourceFile("liquid-lfm25-350m", "model.safetensors"))
    product("model-config", sourceFile("liquid-lfm25-350m", "config.json"))
    product("generation-config", sourceFile("liquid-lfm25-350m", "generation_config.json"))
    product("tokenizer-config", sourceFile("liquid-lfm25-350m", "tokenizer_config.json"))
    product("chat-template", sourceFile("liquid-lfm25-350m", "chat_template.jinja"))
    product("tokenizer", sourceFile("liquid-lfm25-350m", "tokenizer.json"))
    product("model-license", sourceFile("liquid-lfm25-350m", "LICENSE"))
}
tasks.named("blueprintValidate") { dependsOn(preparePreviewProfile) }
tasks.named("blueprintFetch") { dependsOn(preparePreviewProfile) }

val resolveNativeDescriptors = tasks.register<ResolveNativeDescriptors>("resolveNativeDescriptors") {
    group = "cartridge blueprint"
    description = "Uses the actual plugin resolver and serializer to freeze canonical ABI and pretty pack descriptor bytes."
    dependsOn(preparePreviewProfile, "blueprintValidate")
    recipe.set(recipeFile)
    this.previewProfile.set(previewProfileFile)
    destination.set(descriptors)
}
val nativeCompilerImage = providers.gradleProperty("nativeCompilerImage").orElse("skainet/iree-compiler:3.11.0")
val buildNativeRuntime = tasks.register<Exec>("buildNativeRuntime") {
    group = "cartridge blueprint"
    description = "Builds the real ARM64 native runtime in the pinned offline Docker image, embedding both descriptor encodings."
    dependsOn(resolveNativeDescriptors)
    inputs.file(builder)
    inputs.property("compilerImage", nativeCompilerImage)
    inputs.files(nativeSource.file("build.py"), nativeSource.file("bridge.cpp"), nativeSource.file("include/cartridge_abi.h"),
        nativeSource.file("LICENSE.upstream"), nativeSource.file("upstream.json"))
    inputs.dir(descriptors)
    outputs.dir(nativeOutput)
    commandLine("python3", builder.asFile.absolutePath, "--source", nativeSource.asFile.absolutePath,
        "--descriptors", descriptors.get().asFile.absolutePath, "--output", nativeOutput.get().asFile.absolutePath,
        "--image", nativeCompilerImage.get())
}
val prepareNativeExecution = tasks.register<Exec>("prepareNativeExecution") {
    group = "cartridge blueprint"
    description = "Stages an explicitly unmeasured execution pack from actual native/JVM build products and pinned local model files."
    dependsOn(buildNativeRuntime, installWorker, "blueprintFetch")
    inputs.files(helper, workload, nativeSource.file("LICENSE.upstream"))
    inputs.dir(descriptors)
    inputs.dir(nativeOutput)
    inputs.dir(workerLibraries)
    inputs.dir(modelDirectory)
    outputs.dir(execution)
    commandLine("python3", helper.asFile.absolutePath, "prepare", "--model-dir", modelDirectory,
        "--worker-libs", workerLibrariesPath,
        "--descriptor", descriptors.get().file("descriptor.json").asFile.absolutePath,
        "--canonical-descriptor", descriptors.get().file("descriptor.canonical.json").asFile.absolutePath,
        "--native-library", nativeOutput.get().file("liblfm2_cartridge.so").asFile.absolutePath,
        "--native-build-report", nativeOutput.get().file("liblfm2_cartridge.so.build.json").asFile.absolutePath,
        "--compiler-report", nativeOutput.get().file("compiler-container.json").asFile.absolutePath,
        "--native-license", nativeSource.file("LICENSE.upstream").asFile.absolutePath,
        "--workload", workload.asFile.absolutePath, "--output", execution.get().asFile.absolutePath)
}

blueprint {
    val prepared = if (usePreparedExecution) execution else prepareNativeExecution.flatMap { execution }
    product("native-runtime", prepared.map { it.file("artifacts/runtime/liblfm2_cartridge.so") })
    product("worker-jars", prepared.map { it.dir("artifacts/jvm/lib") })
    product("worker-config", prepared.map { it.file("artifacts/runtime/worker.json") })
    product("native-build-report", prepared.map { it.file("artifacts/runtime/native-build.json") })
    product("native-compiler-report", prepared.map { it.file("artifacts/runtime/compiler-container.json") })
    product("native-license", prepared.map { it.file("artifacts/runtime/LICENSE.upstream") })
    product("measurement-workload", prepared.map { it.file("artifacts/evaluation/native-abi-workload.json") })
}

val validateNativeMeasurements = tasks.register<Exec>("validateNativeMeasurements") {
    group = "verification"
    description = "Independently verifies target raw timing/evidence and exact execution bytes before materialization."
    if (!usePreparedExecution) dependsOn(prepareNativeExecution)
    inputs.property("usePreparedExecution", usePreparedExecution)
    inputs.file(benchmarkValidator)
    inputs.files(layout.projectDirectory.file("scripts/validate_native_cartridge.py"),
        layout.projectDirectory.file("scripts/benchmark_native_cartridge.py"),
        layout.projectDirectory.file("scripts/consume_native_cartridge.py"), workload,
        layout.projectDirectory.file("samples/golden-transcripts.json"))
    inputs.files(fileTree("samples/requests") { include("*.json") })
    inputs.dir(benchmarkDirectory)
    inputs.dir(execution)
    outputs.file(acceptance)
    commandLine("python3", benchmarkValidator.asFile.absolutePath, "--benchmark-dir", benchmarkDirectory,
        "--expected-execution", execution.get().asFile.absolutePath, "--output", acceptance.get().asFile.absolutePath)
}
val prepareAcceptedProfile = tasks.register<Exec>("prepareAcceptedProfile") {
    group = "cartridge blueprint"
    description = "Derives stable performance provenance and digest-pinned report inputs from independent measurement acceptance."
    dependsOn(validateNativeMeasurements, preparePreviewProfile)
    inputs.files(helper, previewProfileFile, acceptance)
    inputs.dir(benchmarkDirectory)
    inputs.dir(execution)
    outputs.file(acceptedProfileFile)
    commandLine("python3", helper.asFile.absolutePath, "accepted-profile", "--preview-profile", previewProfileFile.get().asFile.absolutePath,
        "--validation", acceptance.get().asFile.absolutePath, "--benchmark-dir", benchmarkDirectory,
        "--execution", execution.get().asFile.absolutePath, "--output", acceptedProfileFile.get().asFile.absolutePath)
}
val guardFreshMaterialization = tasks.register<GuardFreshMaterialization>("guardFreshMaterialization") {
    group = "cartridge blueprint"
    dependsOn(prepareAcceptedProfile)
    recipe.set(recipeFile)
    this.acceptedProfile.set(acceptedProfileFile)
    parentDirectory.set(materializedParent)
}
tasks.named<MaterializeCartridgeTask>("materializeCartridge") {
    // Only this final task uses an accepted profile; preview performance strings
    // alone cannot authorize materialization through this task graph.
    profileFile.set(acceptedProfileFile)
    // Other blueprints may use the root -PsigningKeyFile; this evaluation task
    // deliberately carries no signing configuration or key.
    signingKeyFile.unset()
    signingKeyFile.unsetConvention()
    dependsOn(guardFreshMaterialization)
    outputs.upToDateWhen { false }
}
tasks.register<Exec>("verifyMaterializedExecution") {
    group = "verification"
    description = "Checks the actual materializer's manifest and every copied execution byte against the measured provisional pack."
    dependsOn("materializeCartridge")
    inputs.file(helper)
    inputs.file(acceptedProfileFile)
    inputs.dir(execution)
    inputs.dir(materializedPack)
    outputs.file(verifiedMaterialization)
    commandLine("python3", helper.asFile.absolutePath, "verify-materialized", "--execution", execution.get().asFile.absolutePath,
        "--pack", materializedPack.get().asFile.absolutePath,
        "--accepted-profile", acceptedProfileFile.get().asFile.absolutePath,
        "--output", verifiedMaterialization.get().asFile.absolutePath)
}
val testPackaging by tasks.registering(Exec::class) {
    group = "verification"
    description = "Runs synthetic staging, ABI-consumer and benchmark evidence tests without weights or device access."
    inputs.dir(layout.projectDirectory.dir("scripts"))
    inputs.dir(layout.projectDirectory.dir("samples"))
    commandLine("python3", "-m", "unittest", "discover", "-s", layout.projectDirectory.dir("scripts").asFile.absolutePath, "-p", "test_*.py", "-v")
}
val validateRecipe by tasks.registering(BlueprintValidateTask::class) {
    description = "Validates the source blueprint schema without a checkpoint or private profile."
    blueprintFile.set(recipeFile)
}
tasks.named("check") { dependsOn(":runtimes:lfm2-jvm:check", testPackaging, validateRecipe) }
