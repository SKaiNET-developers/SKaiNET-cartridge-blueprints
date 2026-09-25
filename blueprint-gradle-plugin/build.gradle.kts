// The reference materializer for cartridge blueprints (spec: SKaiNET-cartridge, blueprints.adoc).
// Gradle plugin id: sk.ainet.cartridge.blueprint
plugins {
    `kotlin-dsl`
    // Keep in step with `kotlin` in gradle/libs.versions.toml — a plugins block cannot read the
    // catalog, so these two literals are the third place the version lives.
    kotlin("jvm") version "2.4.20"
    kotlin("plugin.serialization") version "2.4.20"
    `java-gradle-plugin`
}

group = "sk.ainet.cartridge"
version = providers.gradleProperty("version").getOrElse("0.1.0-SNAPSHOT")

dependencies {
    implementation(libs.kotlinx.serialization.json)
    implementation(libs.optimumcode.json.schema.validator)
    // Dog food: sources are fetched with SKaiNET's own data-source module (hf://, https, file, sha256, cache).
    implementation(libs.skainet.data.source)
    implementation(libs.kotlinx.coroutines.core)
    // skainet-data-source exposes kotlinx.io.Source in its public API but declares kotlinx-io as `implementation`
    // (upstream: should be `api`), so a consumer has to add it to compile against openSource()/copyTo().
    implementation(libs.kotlinx.io.core)

    testImplementation(kotlin("test"))
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_21)
    }
}

tasks.withType<JavaCompile>().configureEach {
    options.release.set(21)
}

gradlePlugin {
    plugins {
        register("cartridgeBlueprint") {
            id = "sk.ainet.cartridge.blueprint"
            implementationClass = "sk.ainet.cartridge.blueprint.gradle.BlueprintPlugin"
            displayName = "SKaiNET cartridge blueprint materializer"
            description = "Materializes a cartridge blueprint against a profile into a signed pack_dir."
        }
    }
}

// Functional tests run real Gradle builds (TestKit) against a synthetic blueprint.
val functionalTest: SourceSet by sourceSets.creating
gradlePlugin.testSourceSets(functionalTest)
configurations[functionalTest.implementationConfigurationName].extendsFrom(configurations.implementation.get(), configurations.testImplementation.get())
dependencies {
    // The synthetic-blueprint fixture is shared with the unit tests.
    functionalTest.implementationConfigurationName(sourceSets.main.get().output)
    functionalTest.implementationConfigurationName(sourceSets.test.get().output)
}

val functionalTestTask = tasks.register<Test>("functionalTest") {
    description = "Runs the TestKit functional tests."
    group = "verification"
    testClassesDirs = functionalTest.output.classesDirs
    classpath = functionalTest.runtimeClasspath
    useJUnitPlatform()
}

tasks.test {
    useJUnitPlatform()
}

tasks.check {
    dependsOn(functionalTestTask)
}
