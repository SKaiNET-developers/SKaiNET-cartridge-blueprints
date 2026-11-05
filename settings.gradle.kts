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
include(":blueprints:generator-smollm2-skainet")
include(":blueprints:generator-llama32-1b-skainet")
include(":blueprints:asr-whisper-iree")
include(":blueprints:embedding-leaf-skainet")
include(":blueprints:nlu-qwen-iree")
include(":profiles:embedding-v1")
include(":profiles:llm-v1")
include(":profiles:yolo-v1")
include(":blueprints:detection-yolov8-iree")
// Hands-on demo app over that blueprint's packs. It builds without the packs; the APK bundles
// whatever `materializeCartridge` has produced locally (AGPL weights never enter the repo).
// The Compose demo graduated to SKaiNET-examples/VisionDemo.
include(":apps:detection-yolov8-demo")

// Independent source dependency while LFM2 is unreleased. No proof-workspace
// source paths or duplicate profile/model classes are included by this recipe.
//
// The LFM2 projects need the sk.ainet.transformers LFM2 modules, which are not
// published yet (SKaiNET-transformers PR #527). Until a release exists they are
// built only when the dependency is reachable: pass
// -Plfm2TransformersCheckout=/path/to/SKaiNET-transformers for a source build,
// or -Plfm2TransformersVersion/-PuseMavenLocal=true for a published/local
// snapshot. Without any of these (e.g. CI), the LFM2 projects are skipped so
// the rest of the repository still builds; remove this gate once the modules
// are released and the default version pin is immutable.
val lfm2Checkout = providers.gradleProperty("lfm2TransformersCheckout").orNull
val lfm2Resolvable = lfm2Checkout != null ||
    providers.gradleProperty("lfm2TransformersVersion").orNull != null ||
    providers.gradleProperty("useMavenLocal").orNull == "true"
lfm2Checkout?.let { checkout ->
    includeBuild(checkout) {
        dependencySubstitution {
            substitute(module("sk.ainet.transformers:skainet-transformers-inference-lfm2"))
                .using(project(":llm-inference:lfm2"))
            substitute(module("sk.ainet.transformers:skainet-transformers-runtime-klfm2"))
                .using(project(":llm-runtime:klfm2"))
        }
    }
}
if (lfm2Resolvable) {
    include(":blueprints:generator-lfm25-350m-skainet")
    include(":runtimes:lfm2-jvm")
    include(":blueprints:generator-lfm25-350m-iree")
} else {
    logger.lifecycle(
        "LFM2 projects skipped: sk.ainet.transformers LFM2 modules are unreleased. " +
            "Pass -Plfm2TransformersCheckout=<SKaiNET-transformers checkout> to build them.",
    )
}
