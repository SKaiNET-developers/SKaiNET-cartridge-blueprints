plugins {
    alias(libs.plugins.kotlinMultiplatform) apply false
    alias(libs.plugins.androidMultiplatformLibrary) apply false
    alias(libs.plugins.kotlinSerialization) apply false
}

// Root aggregator. The materializer plugin is an included build (see settings.gradle.kts), so the
// lifecycle tasks are forwarded to it explicitly — `./gradlew build` must build and test everything.
listOf("build", "check", "clean", "assemble").forEach { lifecycle ->
    tasks.register(lifecycle) {
        group = "build"
        description = "Runs '$lifecycle' in the materializer plugin build and every blueprint module."
        dependsOn(gradle.includedBuild("blueprint-gradle-plugin").task(":$lifecycle"))
        // Lazily: blueprint modules register their lifecycle tasks when they are configured, after this script.
        dependsOn(provider { subprojects.mapNotNull { it.tasks.findByName(lifecycle) } })
    }
}
