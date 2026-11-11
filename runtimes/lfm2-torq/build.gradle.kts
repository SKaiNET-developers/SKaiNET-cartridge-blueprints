// liblfm2_torq_cartridge.so: the Cartridge Runtime ABI 0.1 (native/include/cartridge_abi.h) for the
// generator-lfm25-350m-torq blueprint, as a Kotlin/Native shared library for linux-aarch64. It runs the blueprint's
// greedy decoder (GreedyDecoder, its Kotlin API) over the decode step on the Torq NPU, in the host's process: the
// Torq-fork IREE runtime and the IREE session shim are linked in statically.
//
// The native parts come from source, through Docker:
//   buildTorqNative    skainet/iree-runtime-torq (SKaiNET-embedded-vendors synaptics-torq/docker-runtime, built
//                      locally with its build.sh) runs native/build-native.sh: the runtime archive and its licenses,
//                      the shim, and the descriptor bytes the library embeds (ctg_descriptor_json).
//   linkReleaseSharedLinuxArm64   links them into the library.
// The blueprint wires both into `materializeCartridge`; `build` compiles the Kotlin only (no Docker, no link).
import org.jetbrains.kotlin.gradle.plugin.mpp.SharedLibrary

plugins {
    alias(libs.plugins.kotlinMultiplatform)
}

group = "sk.ainet.cartridge.runtime"
version = "0.1.0-SNAPSHOT"

val runtimeImage = providers.gradleProperty("torqRuntimeImage").orElse("skainet/iree-runtime-torq:2.3.0")
val nativeSource = layout.projectDirectory.dir("native")
val nativeOut = layout.buildDirectory.dir("torq-native")
/** descriptor.json + descriptor.canonical.json; the blueprint points this at what it resolves for the pack. */
val descriptorDir = objects.directoryProperty()
extensions.add("torqDescriptorDir", descriptorDir)

val buildTorqNative = tasks.register<Exec>("buildTorqNative") {
    group = "cartridge runtime"
    description = "Copies the Torq-fork IREE runtime out of $runtimeImage and compiles the IREE session shim and the embedded descriptor for linux-aarch64."
    inputs.dir(nativeSource)
    inputs.dir(descriptorDir)
    inputs.property("image", runtimeImage)
    outputs.dir(nativeOut)
    doFirst {
        val out = nativeOut.get().asFile
        out.deleteRecursively(); out.mkdirs()
        commandLine(
            "docker", "run", "--rm", "--network", "none", "--platform", "linux/amd64", "--entrypoint", "bash",
            "-v", "${nativeSource.asFile.absolutePath}:/src:ro",
            "-v", "${descriptorDir.get().asFile.absolutePath}:/descriptors:ro",
            "-v", "${out.absolutePath}:/out",
            runtimeImage.get(), "/src/build-native.sh",
        )
    }
}

kotlin {
    linuxArm64 {
        compilations.getByName("main").cinterops {
            create("skainet_iree") {
                definitionFile.set(project.file("src/nativeInterop/cinterop/skainet_iree.def"))
                includeDirs(nativeSource.dir("iree-session"))
            }
            create("cartridge_abi") {
                definitionFile.set(project.file("src/nativeInterop/cinterop/cartridge_abi.def"))
                includeDirs(nativeSource.dir("include"))
            }
        }
        binaries.sharedLib("cartridge", listOf(org.jetbrains.kotlin.gradle.plugin.mpp.NativeBuildType.RELEASE)) {
            baseName = "lfm2_torq_cartridge"
            val lib = nativeOut.get().asFile
            linkerOpts(
                // Kotlin/Native adds -lcrypt -lutil -lresolv … by default; newer glibc boards have no libcrypt.so.1.
                // Keep only what the library uses.
                "-Wl,--as-needed",
                lib.resolve("lib/liblfm2_torq_descriptor.a").absolutePath,
                lib.resolve("lib/libskainet_iree.a").absolutePath,
                lib.resolve("dist/lib/libiree_torq_runtime.a").absolutePath,
                "-Wl,-Bstatic", "-lstdc++", "-lstdc++fs", "-Wl,-Bdynamic", "-lm", "-ldl", "-lpthread",
            )
        }
    }

    sourceSets {
        linuxArm64Main.dependencies {
            implementation(project(":blueprints:generator-lfm25-350m-torq"))
        }
    }
}

val linkCartridge = tasks.named("linkCartridgeReleaseSharedLinuxArm64") {
    dependsOn(buildTorqNative)
    // linkerOpts are plain strings: without this the link stays up to date when only the archives (e.g. the
    // embedded descriptor) change, and the library no longer matches its pack.
    inputs.dir(nativeOut).withPathSensitivity(PathSensitivity.RELATIVE)
}

// `build`/`assemble` must not need Docker or the local-only image: the link runs only when asked for by name or
// through the blueprint's materializeCartridge.
// A disabled task's dependencies still run, so the blueprint's resolveRuntimeDescriptor is gated the same way.
gradle.taskGraph.whenReady {
    val wanted = allTasks.any { it.path.endsWith(":materializeCartridge") } ||
        gradle.startParameter.taskNames.any { it.contains("linkCartridge") || it.contains("buildTorqNative") }
    allTasks.filter { it.name == "buildTorqNative" || it.name.startsWith("linkCartridge") || it.name == "resolveRuntimeDescriptor" }
        .forEach { it.enabled = wanted }
}

/** The library and its native report, for the blueprint's products. */
val cartridgeLibrary = linkCartridge.map { (it as org.jetbrains.kotlin.gradle.tasks.KotlinNativeLink).outputFile.get() }
extensions.add("cartridgeLibrary", cartridgeLibrary)
extensions.add("torqNativeOut", nativeOut)
