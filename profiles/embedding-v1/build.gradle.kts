// The `embedding/v1` task profile: the host-facing contract every text-embedding cartridge
// implements, kept separate from any one blueprint so a host can depend on the contract alone.
//
// Pure contract, no implementation, no dependencies — so it compiles for every target any
// embedding blueprint will ever need, including the Apple ones.
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
}
