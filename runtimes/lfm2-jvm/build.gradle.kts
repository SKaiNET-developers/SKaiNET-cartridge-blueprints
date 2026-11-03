import org.gradle.api.file.ConfigurableFileCollection
import org.gradle.api.tasks.Classpath
import org.gradle.process.CommandLineArgumentProvider
import java.time.Duration

plugins {
    alias(libs.plugins.kotlinJvm)
    `java-library`
    application
    `maven-publish`
}

group = "sk.ainet.cartridge.runtime"
version = "0.1.0-SNAPSHOT"

abstract class SingleWorkerClasspathArguments : CommandLineArgumentProvider {
    @get:Classpath abstract val runtimeClasspath: ConfigurableFileCollection
    override fun asArguments(): Iterable<String> =
        listOf("-Dlfm2.singleWorker.classpath=" + runtimeClasspath.asPath)
}

// Unreleased modules are supplied by the independently pinned Transformers source
// checkout, or by a builder-selected published release once available.
val lfm2TransformersVersion = providers.gradleProperty("lfm2TransformersVersion").orElse("0.63.4-SNAPSHOT")
dependencies {
    api(project(":profiles:llm-v1"))
    api("sk.ainet.transformers:skainet-transformers-inference-lfm2:${lfm2TransformersVersion.get()}")
    api("sk.ainet.transformers:skainet-transformers-runtime-klfm2:${lfm2TransformersVersion.get()}")
    implementation(libs.kotlinx.coroutines.core)
    implementation(libs.kotlinx.serialization.json)
    testImplementation(kotlin("test-junit5"))
    testRuntimeOnly("org.junit.platform:junit-platform-launcher:6.1.3")
}
kotlin { jvmToolchain(21) }
java { withSourcesJar() }
application {
    mainClass.set("sk.ainet.cartridge.generator.lfm2.worker.MainKt")
    applicationDefaultJvmArgs = listOf("--enable-preview", "--add-modules", "jdk.incubator.vector")
}
tasks.withType<Test>().configureEach {
    useJUnitPlatform()
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
    testLogging { events("passed", "skipped", "failed") }
}
tasks.test { exclude("**/Lfm2SingleWorkerTest*") }
val singleDefaultWorkerTest by tasks.registering(Test::class) {
    description = "Checks synchronous generation backpressure with one Default CPU worker"
    group = "verification"
    testClassesDirs = sourceSets.test.get().output.classesDirs
    classpath = sourceSets.test.get().runtimeClasspath
    filter { includeTestsMatching("sk.ainet.cartridge.generator.lfm2.Lfm2SingleWorkerTest") }
    jvmArgumentProviders.add(objects.newInstance<SingleWorkerClasspathArguments>().apply {
        runtimeClasspath.from(sourceSets.test.get().runtimeClasspath)
    })
    timeout.set(Duration.ofSeconds(30))
    shouldRunAfter(tasks.test)
}


val testNativeBridge by tasks.registering(Exec::class) {
    group = "verification"
    inputs.dir(layout.projectDirectory.dir("native"))
    commandLine("python3", "-m", "unittest", "discover", "-s", layout.projectDirectory.dir("native/tests").asFile.absolutePath, "-p", "test_bridge.py", "-v")
}
tasks.named("check") { dependsOn(singleDefaultWorkerTest, testNativeBridge) }
