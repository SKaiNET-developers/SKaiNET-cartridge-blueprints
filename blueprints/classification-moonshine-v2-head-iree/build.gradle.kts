// Blueprint: a classification head on the rows of a frozen Moonshine v2 streaming encoder, trained on the
// materializing party's own label set and recordings, compiled with IREE.
//
// (1) The cartridge's Kotlin API — the head in the SKaiNET DSL (`model/ClassificationHead.kt`) and the
// `classification/v1` decision (temperature, energy score, accept / undecided / reject) in commonMain.
// (2) The recipe: the tasks below are the `steps` of blueprint.json, registered as named products for
// `materializeCartridge`.
//
//   ./gradlew :blueprints:classification-moonshine-v2-head-iree:materializeCartridge -Pprofile=<yours>.json
import groovy.json.JsonSlurper
import sk.ainet.cartridge.blueprint.gradle.IreeCompileTask
import java.nio.file.Files

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.kotlinSerialization)
    id("sk.ainet.cartridge.blueprint")
}

/** SKaiNET core: NN DSL, training DSL, autograd, StableHLO export. `-PskainetVersion=` tries another engine. */
val skainetVersion = providers.gradleProperty("skainetVersion").getOrElse("0.63.0")

/** SKaiNET-transformers: the Moonshine v2 frontend and encoder with the checkpoint loader. */
val transformersVersion = providers.gradleProperty("transformersVersion").getOrElse("0.64.2")

/** The SKaiNET-transformers Android runtime with the generic module runner `libskainet_iree_module.so`. */
val moduleRuntimeVersion = providers.gradleProperty("moduleRuntimeVersion").getOrElse("0.64.2")

/** The example's data adapter only (src/jvmExample): parquet through DuckDB. Never on the recipe's classpath. */
val duckdbVersion = "2.0.0.1"

kotlin {
    explicitApi()

    // SKaiNET inlines from JVM 21 bytecode.
    jvm {
        compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_21) }
        // The recipe's own tools (src/jvmBuild): features, train, calibrate, measure, export. Their own compilation,
        // so the library never carries the trainer, the StableHLO compiler or the encoder.
        val main by compilations.getting
        val build = compilations.create("build") {
            associateWith(main)
            defaultSourceSet.dependencies {
                implementation("sk.ainet.core:skainet-compile-hlo:$skainetVersion")
                implementation("sk.ainet.core:skainet-compile-dag:$skainetVersion")
                implementation("sk.ainet.core:skainet-backend-cpu:$skainetVersion")
                implementation("sk.ainet.core:skainet-io-core:$skainetVersion")
                implementation("sk.ainet.core:skainet-io-safetensors:$skainetVersion")
                implementation("sk.ainet.transformers:skainet-transformers-inference-moonshine:$transformersVersion")
                implementation(libs.kotlinx.serialization.json)
            }
        }
        // The example's data adapter (PrepareSpeechMassiveCli): its own compilation, so parquet and DuckDB stay
        // out of the recipe and out of the library.
        compilations.create("example") {
            associateWith(build)
            defaultSourceSet.dependencies {
                implementation("org.duckdb:duckdb_jdbc:$duckdbVersion")
                implementation(libs.kotlinx.serialization.json)
            }
        }
    }
    jvmToolchain(21)

    sourceSets {
        commonMain.dependencies {
            implementation(libs.kotlinx.serialization.json)
            api(libs.kotlinx.io.core)
            api("sk.ainet.core:skainet-lang-core:$skainetVersion")
        }
        commonTest.dependencies {
            implementation(libs.kotlin.test)
        }
    }
}

// The recipe's tools are tested like the library: jvmTest sees the `build` compilation and what it depends on.
val buildCompilation = kotlin.jvm().compilations.getByName("build")
kotlin.jvm().compilations.getByName("test").apply {
    associateWith(buildCompilation)
    defaultSourceSet.dependencies {
        implementation(files(buildCompilation.output.allOutputs))
        implementation(files(buildCompilation.runtimeDependencyFiles))
    }
}

tasks.withType<Test>().configureEach {
    systemProperty("blueprint.dir", layout.projectDirectory.asFile.absolutePath)
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
    maxHeapSize = "4g"
}

// ------------------------------------------------------------------------------------------------------------
// The recipe (blueprint.json `steps`). `fetch`, `pack` and `sign` are the plugin's.
// ------------------------------------------------------------------------------------------------------------

val work = layout.buildDirectory.dir("blueprint/work")
val selected = blueprint.target

// The profile picks the encoder with exactly one flavor: a head belongs to the encoder it was trained on.
val flavor = selected.map {
    it.flavors.singleOrNull() ?: throw GradleException("The profile must select exactly one flavor (the encoder): \"de\" or \"en\"; it selects ${it.flavors}.")
}
val encoderWeights = flavor.flatMap { blueprint.sourceFile("encoder-$it", "model.safetensors") }
val datasetManifest = blueprint.inputFile("dataset")
val labelSet = blueprint.inputFile("label-set")

// A step's `params` are read from blueprint.json, so the recipe has one place that states them.
@Suppress("UNCHECKED_CAST")
val stepParams: Map<String, Map<String, Any?>> =
    ((JsonSlurper().parse(file("blueprint.json")) as Map<String, Any?>)["steps"] as List<Map<String, Any?>>)
        .associate { it["id"] as String to ((it["params"] as? Map<String, Any?>) ?: emptyMap()) }
fun param(step: String, name: String): String = stepParams.getValue(step)[name]?.toString() ?: error("blueprint.json: steps[$step].params.$name")

/** A step tool of this module's `build` compilation, on JDK 21 with the vector API SKaiNET's CPU backend uses. */
fun tool(name: String, main: String, compilation: org.jetbrains.kotlin.gradle.plugin.KotlinCompilation<*> = buildCompilation, configure: JavaExec.() -> Unit) =
    tasks.register<JavaExec>(name) {
        group = "cartridge blueprint steps"
        classpath = files(
            compilation.output.allOutputs, compilation.runtimeDependencyFiles,
            buildCompilation.output.allOutputs, buildCompilation.runtimeDependencyFiles,
        )
        mainClass.set("sk.ainet.cartridge.classification.moonshine.head.$main")
        javaLauncher.set(javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(21)) })
        jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
        maxHeapSize = "8g"
        configure()
    }

// features — every recording through the frozen encoder, cached. Hours for a large dataset; resumable.
val featuresDir = work.zip(flavor) { w, f -> w.dir("features/$f") }
val features = tool("features", "recipe.FeaturesCliKt") {
    description = "features: the dataset's recordings -> frozen encoder rows, in safetensors shards (FeaturesCli)"
    inputs.file(datasetManifest).withPathSensitivity(PathSensitivity.NONE)
    inputs.file(encoderWeights).withPathSensitivity(PathSensitivity.NONE)
    inputs.property("maxRows", param("features", "max_rows"))
    outputs.dir(featuresDir)
    val workers = providers.gradleProperty("featureWorkers")
    doFirst {
        args(
            "--manifest", datasetManifest.get().asFile.absolutePath,
            "--encoder", encoderWeights.get().asFile.parentFile.absolutePath,
            "--out", featuresDir.get().asFile.absolutePath,
            "--max-rows", param("features", "max_rows"),
        )
        workers.orNull?.let { args("--workers", it) }
    }
}

// train — the head, fitted (ADR-018).
val trainDir = work.zip(flavor) { w, f -> w.dir("train/$f") }
val train = tool("train", "recipe.TrainCliKt") {
    description = "train: the head fitted on the train split with the SKaiNET training DSL (TrainCli)"
    inputs.dir(features.map { featuresDir.get() }).withPathSensitivity(PathSensitivity.RELATIVE)
    inputs.file(labelSet).withPathSensitivity(PathSensitivity.NONE)
    inputs.property("params", stepParams.getValue("train").toString())
    outputs.dir(trainDir)
    doFirst {
        args(
            "--features", featuresDir.get().asFile.absolutePath,
            "--label-set", labelSet.get().asFile.absolutePath,
            "--out", trainDir.get().asFile.absolutePath,
            "--seed", param("train", "seed"),
            "--hidden", param("train", "hidden"),
            "--dropout", param("train", "dropout"),
            "--learning-rate", param("train", "learning_rate"),
            "--batch", param("train", "batch"),
            "--max-epochs", param("train", "max_epochs"),
            "--patience", param("train", "patience"),
        )
    }
}
// `map`, not `flatMap`: the provider must carry the task that writes the file.
val headWeights = train.map { trainDir.get().file("head.safetensors") }

// calibrate — temperature and thresholds, fitted on dev (ADR-018).
val calibrationFile = work.zip(flavor) { w, f -> w.file("calibrate/$f/calibration.json") }
val calibrate = tool("calibrate", "recipe.CalibrateCliKt") {
    description = "calibrate: temperature and accept / reject thresholds on the dev split (CalibrateCli)"
    inputs.file(headWeights).withPathSensitivity(PathSensitivity.NONE)
    inputs.dir(features.map { featuresDir.get() }).withPathSensitivity(PathSensitivity.RELATIVE)
    inputs.file(labelSet).withPathSensitivity(PathSensitivity.NONE)
    inputs.property("params", stepParams.getValue("calibrate").toString())
    outputs.file(calibrationFile)
    doFirst {
        args(
            "--weights", headWeights.get().asFile.absolutePath,
            "--features", featuresDir.get().asFile.absolutePath,
            "--label-set", labelSet.get().asFile.absolutePath,
            "--out", calibrationFile.get().asFile.absolutePath,
            "--seed", param("calibrate", "seed"),
            "--score", param("calibrate", "score"),
            "--target-risk", param("calibrate", "target_risk"),
            "--reject-keep", param("calibrate", "reject_keep"),
        )
    }
}

// measure — the test split, once. Run it by name; copy `quality` from the report into your profile.
val qualityReport = layout.buildDirectory.zip(flavor) { b, f -> b.file("blueprint/quality-report.$f.json") }
tool("measure", "recipe.MeasureCliKt") {
    description = "measure: top-1, macro F1, out-of-set separation and accepted share on the test split (MeasureCli)"
    inputs.file(headWeights).withPathSensitivity(PathSensitivity.NONE)
    inputs.file(calibrate.map { calibrationFile.get() }).withPathSensitivity(PathSensitivity.NONE)
    inputs.dir(features.map { featuresDir.get() }).withPathSensitivity(PathSensitivity.RELATIVE)
    inputs.file(labelSet).withPathSensitivity(PathSensitivity.NONE)
    outputs.file(qualityReport)
    doFirst {
        args(
            "--weights", headWeights.get().asFile.absolutePath,
            "--calibration", calibrationFile.get().asFile.absolutePath,
            "--features", featuresDir.get().asFile.absolutePath,
            "--label-set", labelSet.get().asFile.absolutePath,
            "--manifest", datasetManifest.get().asFile.absolutePath,
            "--out", qualityReport.get().asFile.absolutePath,
        )
    }
}

// export — the same head definition, traced with the fitted weights -> StableHLO.
val headMlir = work.zip(flavor) { w, f -> w.file("export/$f/head.mlir") }
val export = tool("exportStableHlo", "recipe.ExportCliKt") {
    description = "export: the fitted head -> head.mlir, one static StableHLO function (ExportCli)"
    inputs.file(headWeights).withPathSensitivity(PathSensitivity.NONE)
    outputs.file(headMlir)
    doFirst { args("--weights", headWeights.get().asFile.absolutePath, "--out", headMlir.get().asFile.absolutePath) }
}

// compile — the graph -> .vmfb for the profile's target.
val compile = tasks.register<IreeCompileTask>("compile") {
    group = "cartridge blueprint steps"
    image.set(selected.map { it.image("iree-tools") })
    input.set(export.map { headMlir.get() })
    backend.set(selected.map { it.param("backend") })
    target.set(selected.map { it.param("iree_target") })
    output.set(work.zip(selected) { w, t -> w.file("compile/${t.id}/${t.flavors.single()}/head.vmfb") })
}

// build-runtime — the generic single-shot module runner for the target ABI, from the released Android runtime (AAR).
val runtimeAar: Configuration = configurations.create("runtimeAar") { isCanBeConsumed = false; isTransitive = false }
dependencies { runtimeAar("sk.ainet.transformers:skainet-transformers-runtime-iree-android:$moduleRuntimeVersion@aar") }

val extractRuntime = tasks.register<Copy>("extractRuntime") {
    group = "cartridge blueprint steps"
    description = "build-runtime: libskainet_iree_module.so for the target ABI, from the released SKaiNET-transformers runtime"
    val abi = selected.map { it.abi }
    from(runtimeAar.elements.map { files -> files.map { zipTree(it) } }) {
        include { it.path == "jni/${abi.get()}/libskainet_iree_module.so" }
        eachFile { path = name }
        includeEmptyDirs = false
    }
    into(work.zip(selected) { w, t -> w.dir("runtime/${t.abi}") })
}

blueprint.product("runtime-so", extractRuntime.map { work.get().file("runtime/${selected.get().abi}/libskainet_iree_module.so") })
blueprint.product("head-vmfb", compile.flatMap { it.output })
blueprint.product("head-weights", headWeights)
blueprint.product("label-set-json", train.map { trainDir.get().file("label-set.json") })
blueprint.product("calibration-json", calibrate.map { calibrationFile.get() })

// ------------------------------------------------------------------------------------------------------------
// Verification `host-parity`: the compiled function against the eager head, on test rows, on this machine.
// ------------------------------------------------------------------------------------------------------------

val parityDir = layout.buildDirectory.zip(flavor) { b, f -> b.dir("parity/$f") }
val parityInputs = tool("parityInputs", "recipe.ParityCliKt") {
    group = "verification"
    description = "parity: test rows and the eager head's logits for them (ParityCli --mode write)"
    inputs.file(headWeights).withPathSensitivity(PathSensitivity.NONE)
    inputs.dir(features.map { featuresDir.get() }).withPathSensitivity(PathSensitivity.RELATIVE)
    outputs.dir(parityDir.map { it.dir("io") })
    doFirst {
        args(
            "--mode", "write", "--weights", headWeights.get().asFile.absolutePath, "--features", featuresDir.get().asFile.absolutePath,
            "--label-set", labelSet.get().asFile.absolutePath, "--dir", parityDir.get().dir("io").asFile.absolutePath,
        )
    }
}
val parityCompile = tasks.register<IreeCompileTask>("parityCompile") {
    group = "verification"
    image.set(selected.map { it.image("iree-tools") })
    input.set(export.map { headMlir.get() })
    backend.set("cpu")
    target.set("host")
    output.set(parityDir.map { it.file("head.vmfb") })
}
val parityRun = tasks.register<Exec>("parityRun") {
    group = "verification"
    description = "parity: iree-run-module on the host build of the head, one call per input"
    dependsOn(parityInputs)
    val vmfb = parityCompile.flatMap { it.output }
    inputs.file(vmfb).withPathSensitivity(PathSensitivity.NONE)
    doFirst {
        val io = parityDir.get().dir("io").asFile
        val uid = runCatching { Files.getAttribute(io.toPath(), "unix:uid").toString() }.getOrNull()
        commandLine(buildList {
            addAll(listOf("docker", "run", "--rm"))
            if (uid != null) addAll(listOf("--user", uid))
            addAll(listOf("-v", "${vmfb.get().asFile.parentFile}:/module:ro", "-v", "$io:/io", selected.get().image("iree-tools")))
            addAll(listOf("shell", "-c", "set -e; for f in /io/input-*.npy; do o=\$(echo \$f | sed s/input-/output-/); " +
                "iree-run-module --device=local-task --module=/module/head.vmfb --function=classify --input=@\$f --output=@\$o; done"))
        })
    }
}
tool("parityHost", "recipe.ParityCliKt") {
    group = "verification"
    description = "parity: the compiled head vs the eager head on test rows (blueprint.json verification host-parity)"
    dependsOn(parityRun)
    doFirst { args("--mode", "compare", "--dir", parityDir.get().dir("io").asFile.absolutePath, "--tolerance", "0.001") }
}

// ------------------------------------------------------------------------------------------------------------
// Not a recipe step: the example's data adapter. Speech-MASSIVE (parquet) -> a dataset manifest + 16 kHz WAV.
//
//   ./gradlew :blueprints:classification-moonshine-v2-head-iree:prepareSpeechMassive \
//       -PdatasetDir=<where the manifest and WAVs go> -PacceptDatasetLicense=CC-BY-NC-SA-4.0
// ------------------------------------------------------------------------------------------------------------

tool("prepareSpeechMassive", "example.PrepareSpeechMassiveCliKt", kotlin.jvm().compilations.getByName("example")) {
    group = "cartridge blueprint example"
    description = "example data: the pinned Speech-MASSIVE de-DE parquet files -> dataset.json + 16 kHz WAV (DuckDB)"
    val out = providers.gradleProperty("datasetDir")
    val accept = providers.gradleProperty("acceptDatasetLicense")
    val only = providers.gradleProperty("datasetOnly")
    val cache = providers.gradleProperty("datasetCache")
    doFirst {
        val dir = file(out.orNull ?: throw GradleException("pass -PdatasetDir=<directory for dataset.json and the WAV files>"))
        args(
            "--source", file("samples/speech-massive-de.source.json").absolutePath,
            "--out", dir.absolutePath,
            "--cache", (cache.orNull?.let(::file) ?: dir.resolve(".parquet")).absolutePath,
            "--accept-license", accept.getOrElse("none"),
        )
        only.orNull?.let { args("--only", it) }
    }
}
