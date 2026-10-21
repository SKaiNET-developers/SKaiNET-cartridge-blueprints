// The `llm/v1` task profile: the host-facing contract for generation cartridges, implementing
// ADR-011 over ADR-007's descriptor groundwork. Mirrors asr/v1's two-layer shape: the Flow
// contract is the normative core, the callback bridge ships in the same artifact.
//
// Pure contract plus kotlinx-coroutines (for Flow) — no engine, no model, no SKaiNET, so it
// compiles for every target any generation blueprint will ever need, including the Apple ones.
plugins {
    alias(libs.plugins.kotlinMultiplatform)
    `maven-publish`
}

group = "sk.ainet.cartridge.profiles"
version = "0.1.0"

kotlin {
    explicitApi()

    jvm { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_21) } }
    jvmToolchain(21)
    iosArm64()
    iosSimulatorArm64()
    macosArm64()

    sourceSets {
        commonMain.dependencies {
            api(libs.kotlinx.coroutines.core)
        }
        commonTest.dependencies {
            implementation(libs.kotlin.test)
            implementation(libs.kotlinx.coroutines.test)
        }
    }
}
