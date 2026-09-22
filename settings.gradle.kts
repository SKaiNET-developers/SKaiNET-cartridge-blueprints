pluginManagement {
    // The materializer plugin lives in this repository and is applied by the blueprint modules by id.
    includeBuild("blueprint-gradle-plugin")
    repositories {
        gradlePluginPortal()
        mavenCentral()
        google()
    }
}

dependencyResolutionManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        // Opt-in for developing against an unreleased SKaiNET / SKaiNET-transformers snapshot; never the default.
        if (providers.gradleProperty("useMavenLocal").orNull == "true") mavenLocal()
    }
}

rootProject.name = "SKaiNET-cartridge-blueprints"

include(":blueprints:nlu-functiongemma-270m-iree")
include(":blueprints:asr-moonshine-v2-streaming-iree")
