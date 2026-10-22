// The `yolo/v1` task profile: the host-facing contract every YOLO-family detection cartridge implements,
// kept separate from any one blueprint so a host can depend on the contract alone.
//
// The sources are a verbatim copy of the normative profile in SKaiNET-cartridge
// (cartridge-task-profiles/src/commonMain/kotlin/sk/ainet/cartridge/profile/yolo/v1, commit 0a40197),
// same package, until that module is published. Pure contract, no dependencies.
plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)   // published: sk.ainet.cartridge.profiles:yolo-v1 (see publish.yml)
}

group = "sk.ainet.cartridge.profiles"
version = "0.1.0"

kotlin {
    explicitApi()

    android {
        namespace = "sk.ainet.cartridge.profile.yolo.v1"
        compileSdk = libs.versions.androidCompileSdk.get().toInt()
        minSdk = libs.versions.androidMinSdk.get().toInt()
        compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) }
    }
    jvm { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }
    linuxX64()
    linuxArm64()
    macosArm64()
}
