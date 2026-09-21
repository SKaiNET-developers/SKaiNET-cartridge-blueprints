// Root aggregator. The materializer plugin is an included build (see settings.gradle.kts), so the
// lifecycle tasks are forwarded to it explicitly — `./gradlew build` must build and test everything.
listOf("build", "check", "clean", "assemble").forEach { lifecycle ->
    tasks.register(lifecycle) {
        group = "build"
        description = "Runs '$lifecycle' in the materializer plugin build and every blueprint module."
        dependsOn(gradle.includedBuild("blueprint-gradle-plugin").task(":$lifecycle"))
        dependsOn(subprojects.mapNotNull { it.tasks.findByName(lifecycle) })
    }
}
