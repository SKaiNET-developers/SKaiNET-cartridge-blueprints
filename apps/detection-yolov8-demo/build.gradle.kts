import org.jetbrains.kotlin.gradle.dsl.JvmTarget

// Hands-on demo for the detection-yolov8-iree blueprint: pick a photo, run a materialized yolo/v1
// pack on this device (CPU or Vulkan — whichever packs you pushed), see the boxes and the latency.
//
// The app ships NO weights and NO compiled model (the packs are AGPL-3.0 — see the blueprint's
// Licensing section). You push materialized packs once, like skainet-decode-android pushes its GGUF:
//
//   ./gradlew :apps:detection-yolov8-demo:installDebug
//   adb push blueprints/detection-yolov8-iree/build/cartridge/detection-yolov8-iree-vulkan-arm64 \
//     /sdcard/Android/data/sk.ainet.cartridge.demo.yolo/files/packs/
plugins {
    // AGP 9 ships built-in Kotlin support — applying kotlin("android") is an error since 9.0,
    // and AGP is already on the build classpath through the library modules' aliases, so apply
    // by bare id (skainet-decode-android in SKaiNET-transformers is the precedent app).
    id("com.android.application")
}

android {
    namespace = "sk.ainet.cartridge.demo.yolo"
    compileSdk = libs.versions.androidCompileSdk.get().toInt()

    defaultConfig {
        applicationId = "sk.ainet.cartridge.demo.yolo"
        minSdk = libs.versions.androidMinSdk.get().toInt()
        targetSdk = libs.versions.androidCompileSdk.get().toInt()
        versionCode = 1
        versionName = "0.1"

        ndk {
            // The ABIs the runner AAR (libskainet_iree_module.so) actually carries.
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
        }
    }

    packaging {
        // SKaiNET registers kernels through ServiceLoader: merge, never drop, the entries.
        resources { merges += "META-INF/services/**" }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlin {
        compilerOptions { jvmTarget.set(JvmTarget.JVM_17) }
    }

    // Packs materialized locally ride into the APK as assets (assets/packs/<pack_dir>/…), so a
    // locally built APK is self-contained — no adb pushes. The packs are AGPL-3.0 and never enter
    // git (they live under the blueprint's gitignored build/); bundling them into YOUR build for
    // YOUR device is private use, but handing that APK to anyone else is distribution and the
    // AGPL obligations travel with it (see the blueprint's Licensing section and ADR-017).
    sourceSets.getByName("main").assets.srcDir("build/bundled-assets")
}

val bundlePacks = tasks.register<Copy>("bundlePacks") {
    from(rootProject.file("blueprints/detection-yolov8-iree/build/cartridge")) {
        include("detection-yolov8-iree-*/**")
    }
    into(layout.buildDirectory.dir("bundled-assets/packs"))
}
tasks.named("preBuild") { dependsOn(bundlePacks) }

dependencies {
    // The blueprint library api-exposes the yolo/v1 profile and the IREE module runner.
    implementation(project(":blueprints:detection-yolov8-iree"))
    // runBlocking around the engine's suspend infer on the worker thread.
    implementation(libs.kotlinx.coroutines.core)
    // Live preview: ImageAnalysis frames ARE the preview (no Preview use case, so the
    // detection overlay maps 1:1 onto what is shown); ComponentActivity owns the lifecycle.
    implementation(libs.androidx.activity)
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.lifecycle)
}
