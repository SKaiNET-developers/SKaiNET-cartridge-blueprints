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
include(":blueprints:asr-whisper-iree")
include(":blueprints:embedding-leaf-skainet")
include(":profiles:embedding-v1")
include(":blueprints:nlu-qwen-iree")
include(":profiles:yolo-v1")
// detection-yolov8-iree builds against SKaiNET pieces that are not released yet (skainet-data-media,
// and IreeModuleRunner from SKaiNET-transformers), so it is part of the build only under the same
// opt-in that makes those resolvable (see its build script for the snapshot coordinates). Once the
// upstream releases land, bump the catalog and make this include unconditional.
if (providers.gradleProperty("useMavenLocal").orNull == "true") {
    include(":blueprints:detection-yolov8-iree")
}
