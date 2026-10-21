import sk.ainet.cartridge.blueprint.gradle.IreeCompileTask
import sk.ainet.cartridge.blueprint.gradle.IreeExportParametersTask
import java.nio.file.Files

// Blueprint: single-shot object detection with YOLOv8n (COCO, 80 classes), IREE on Vulkan or CPU.
//
// (1) The cartridge's Kotlin API — a Kotlin Multiplatform library: the YOLOv8 graph in the SKaiNET DSL
// (`model/YoloV8.kt`, owned here, not by SKaiNET core), the `yolo/v1` engine (letterbox → head → NMS → source pixels) and
// the pack locator in commonMain; the runtime binding where SKaiNET-transformers ships the generic module runner.
// (2) The recipe: the tasks below are the `steps` of blueprint.json, registered as named products for
// `materializeCartridge`.
//
//   ./gradlew :blueprints:detection-yolov8-iree:materializeCartridge -Pprofile=profiles/<yours>.json
plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.kotlinSerialization)
    id("sk.ainet.cartridge.blueprint")
}

/**
 * SKaiNET core with `Letterbox` / `nonMaxSuppression` (skainet-data-media). Released: SKaiNET 0.62.1
 * carries skainet-data-media (first publish) and the lowering fixes of SKaiNET PR #1437 — everything
 * here resolves from Maven Central. Intentionally ahead of the catalog's `skainet`, which the other
 * blueprints share; override with `-PskainetVersion=` to try another engine.
 */
val skainetVersion = providers.gradleProperty("skainetVersion").getOrElse("0.62.1")

/**
 * The SKaiNET-transformers runtime with `IreeModuleRunner` + `libskainet_iree_module.so` (branch
 * `feature/iree-module-runner`, SKaiNET-transformers PR #513 — NOT in 0.63.0). UNTIL A RELEASE CARRIES IT:
 * `-PmoduleRuntimeVersion=0.63.1-SNAPSHOT` with `-PuseMavenLocal=true`, against
 * `:llm-runtime:iree-android:publishToMavenLocal -PVERSION_NAME=0.63.1-SNAPSHOT` from that branch.
 * This default is the smallest release number that could ship it; the module stays out of the default
 * build until one does (see settings.gradle.kts).
 */
val moduleRuntimeVersion = providers.gradleProperty("moduleRuntimeVersion").getOrElse("0.63.1")

kotlin {
    explicitApi()

    android {
        namespace = "sk.ainet.cartridge.detection.yolov8"
        compileSdk = libs.versions.androidCompileSdk.get().toInt()
        minSdk = libs.versions.androidMinSdk.get().toInt()
        compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) }
        // On-device check of the materialized pack through the public engine API (src/androidDeviceTest,
        // YoloV8CartridgeDeviceTest) — the E2 golden run.
        withDeviceTest {
            instrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        }
    }
    // SKaiNET inlines from JVM 21 bytecode (same as embedding-leaf-skainet).
    jvm {
        compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_21) }
        // The export tool (YoloV8ExportCli, src/jvmExport): its own compilation, so the library never carries the
        // StableHLO compiler or the ONNX reader.
        val main by compilations.getting
        compilations.create("export") {
            associateWith(main)
            defaultSourceSet.dependencies {
                implementation("sk.ainet.core:skainet-compile-hlo:$skainetVersion")
                implementation("sk.ainet.core:skainet-compile-dag:$skainetVersion")
                implementation("sk.ainet.core:skainet-io-onnx:$skainetVersion")
                implementation("sk.ainet.core:skainet-backend-cpu:$skainetVersion")   // eager forward of the parity check

                implementation("pro.streem.pbandk:pbandk-runtime:0.16.0")   // onnx.ModelProto decoding (SKaiNET's pin)
                implementation(libs.kotlinx.serialization.json)
            }
        }
    }
    jvmToolchain(21)
    linuxX64()
    linuxArm64()
    macosArm64()

    sourceSets {
        commonMain.dependencies {
            api(project(":profiles:yolo-v1"))                    // the yolo/v1 contract YoloV8Engine implements
            implementation(libs.kotlinx.serialization.json)
            api(libs.kotlinx.io.core)                             // kotlinx.io.files.Path is part of the public API (PackDir)
            api("sk.ainet.core:skainet-data-media:$skainetVersion")   // BoundingBox / ScoredBox in the public API
            implementation("sk.ainet.core:skainet-lang-core:$skainetVersion")
            implementation("sk.ainet.core:skainet-lang-models:$skainetVersion")
        }
        androidMain.dependencies {
            api("sk.ainet.transformers:skainet-transformers-runtime-iree-android:$moduleRuntimeVersion")  // IreeModuleRunner
        }
        commonTest.dependencies {
            implementation(libs.kotlin.test)
            implementation(libs.kotlinx.coroutines.test)
        }
        getByName("androidDeviceTest").dependencies {
            implementation(libs.androidx.test.junit)
            implementation(libs.androidx.test.runner)
            implementation(libs.kotlinx.coroutines.core)
        }
    }
}

tasks.withType<Test>().configureEach {
    systemProperty("blueprint.dir", layout.projectDirectory.asFile.absolutePath)
}

// ------------------------------------------------------------------------------------------------------------
// The recipe (blueprint.json `steps`). `fetch`, `pack` and `sign` are the plugin's.
// ------------------------------------------------------------------------------------------------------------

val work = layout.buildDirectory.dir("blueprint/work")
val selected = blueprint.target

// convert — the PyTorch checkpoint → ONNX, in the pinned Ultralytics container. Only the initializers are used
// downstream; the graph is the SKaiNET one in model/.
val onnxVersion = "1.23.2"  // blueprint.json steps[convert].params.onnx
val convertOnnx = tasks.register<Exec>("convertOnnx") {
    group = "cartridge blueprint steps"
    description = "convert: yolov8n.pt -> yolov8n.onnx (Ultralytics exporter, pinned container)"
    val pt = blueprint.sourceFile("weights", "yolov8n.pt")
    val outDir = work.map { it.dir("convert") }
    inputs.file(pt).withPathSensitivity(PathSensitivity.NONE)
    outputs.file(outDir.map { it.file("yolov8n.onnx") })
    doFirst {
        val out = outDir.get().asFile.apply { mkdirs() }
        val src = pt.get().asFile.canonicalFile
        val image = selected.get().image("ultralytics")
        val uid = runCatching { Files.getAttribute(out.toPath(), "unix:uid").toString() }.getOrNull()
        commandLine(buildList {
            addAll(listOf("docker", "run", "--rm"))
            if (uid != null) addAll(listOf("--user", uid))
            addAll(listOf("-e", "HOME=/out", "-e", "YOLO_CONFIG_DIR=/out/.ultralytics", "-e", "PYTHONPATH=/out/.pydeps"))
            addAll(listOf("-v", "${src.parentFile}:/in:ro", "-v", "${out}:/out", "-w", "/out", image))
            // The CPU image has no `onnx` module; the exporter needs it. Pinned, installed into the work dir only.
            addAll(listOf("sh", "-c", "set -e; pip install -q --no-cache-dir --target /out/.pydeps onnx==$onnxVersion; " +
                "cp /in/${src.name} /out/yolov8n.pt; yolo export model=/out/yolov8n.pt format=onnx imgsz=640 opset=17 simplify=False; rm /out/yolov8n.pt"))
        })
    }
}

// export — ONNX initializers → the SKaiNET YoloV8 graph (in-graph DFL decode) → StableHLO, with this module's own
// export compilation as the tool.
val exportDir = work.map { it.dir("export") }
val exportCompilation = kotlin.jvm().compilations.getByName("export")
val export = tasks.register<JavaExec>("exportStableHlo") {
    group = "cartridge blueprint steps"
    description = "export: ONNX initializers -> SKaiNET YoloV8 graph -> yolov8n.mlir (YoloV8ExportCli)"
    classpath = files(exportCompilation.output.allOutputs, exportCompilation.runtimeDependencyFiles)
    mainClass.set("sk.ainet.cartridge.detection.yolov8.export.YoloV8ExportCliKt")
    javaLauncher.set(javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(21)) })
    maxHeapSize = "4g"
    val onnx = convertOnnx.map { it.outputs.files.singleFile }
    inputs.file(onnx).withPathSensitivity(PathSensitivity.NONE)
    outputs.dir(exportDir)
    doFirst {
        environment("YOLO_ONNX", onnx.get().absolutePath)
        environment("YOLO_OUT", exportDir.get().file("yolov8n.mlir").asFile.absolutePath)
        environment("YOLO_INPUT_SIZE", "640")   // blueprint.json attributes.input_resolution
    }
}

// Not a recipe step: the B4 host parity of the eager SKaiNET forward against an ONNX Runtime reference
// (YoloV8ParityCli — see its doc for the two-phase flow). `-Pimage=<photo>` picks the test image,
// `-PparityDir=<dir>` the work directory (default build/parity), so a set of images keeps separate artifacts.
tasks.register<JavaExec>("parityEager") {
    group = "verification"
    description = "parity: ONNX initializers -> eager SKaiNET forward vs ONNX Runtime head (YoloV8ParityCli)"
    classpath = files(exportCompilation.output.allOutputs, exportCompilation.runtimeDependencyFiles)
    mainClass.set("sk.ainet.cartridge.detection.yolov8.export.YoloV8ParityCliKt")
    javaLauncher.set(javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(21)) })
    maxHeapSize = "4g"
    val onnx = convertOnnx.map { it.outputs.files.singleFile }
    val parityDir = providers.gradleProperty("parityDir")
        .map { layout.projectDirectory.dir(it) }
        .orElse(layout.buildDirectory.dir("parity").get())
    val image = providers.gradleProperty("image")
    doFirst {
        environment("YOLO_ONNX", onnx.get().absolutePath)
        environment("YOLO_IMAGE", image.orNull ?: error("pass -Pimage=<test image> (not committed; e.g. bus.jpg)"))
        environment("YOLO_WORK", parityDir.get().asFile.absolutePath)
        environment("YOLO_INPUT_SIZE", "640")
    }
}

// convert — the weights out of the graph into one target-independent parameter archive.
val parameters = tasks.register<IreeExportParametersTask>("parameters") {
    group = "cartridge blueprint steps"
    image.set(selected.map { it.image("iree-tools") })
    dependsOn(export)
    input.set(exportDir.map { it.file("yolov8n.mlir") })
    output.set(work.map { it.file("parameters/params.irpa") })
    scope.set("model")
}

// compile — the graph → .vmfb for the profile's target, referencing the archive.
val compile = tasks.register<IreeCompileTask>("compile") {
    group = "cartridge blueprint steps"
    image.set(selected.map { it.image("iree-tools") })
    dependsOn(export)
    input.set(exportDir.map { it.file("yolov8n.mlir") })
    backend.set(selected.map { it.param("backend") })
    target.set(selected.map { it.param("iree_target") })
    parameterScope.set("model")
    output.set(work.zip(selected) { w, t -> w.file("compile/${t.id}/yolov8n.vmfb") })
}
blueprint.product("yolov8n-vmfb", compile.flatMap { it.output })
blueprint.product("params-irpa", parameters.flatMap { it.output })

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
