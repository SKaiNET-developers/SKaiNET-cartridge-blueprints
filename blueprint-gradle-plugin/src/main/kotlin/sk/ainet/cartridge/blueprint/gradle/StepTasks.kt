package sk.ainet.cartridge.blueprint.gradle

import org.gradle.api.DefaultTask
import org.gradle.api.GradleException
import org.gradle.api.file.RegularFileProperty
import org.gradle.api.provider.ListProperty
import org.gradle.api.provider.Property
import org.gradle.api.tasks.CacheableTask
import org.gradle.api.tasks.Input
import org.gradle.api.tasks.InputFile
import org.gradle.api.tasks.Optional
import org.gradle.api.tasks.OutputFile
import org.gradle.api.tasks.PathSensitive
import org.gradle.api.tasks.PathSensitivity
import org.gradle.api.tasks.TaskAction
import org.gradle.process.ExecOperations
import org.gradle.work.DisableCachingByDefault
import sk.ainet.cartridge.blueprint.steps.HostGatherRewrite
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.file.Files
import javax.inject.Inject

/**
 * Reusable building blocks for the model-specific steps of a blueprint (`convert`, `compile`). A blueprint module
 * registers and wires them; the plugin's own tasks (`blueprintFetch`, `materializeCartridge`) stay model-agnostic.
 */

/** `convert`: replaces the token-embedding gather of an exported StableHLO module with a `%emb` argument. */
@CacheableTask
abstract class HostGatherTask : DefaultTask() {
    @get:InputFile @get:PathSensitive(PathSensitivity.NONE) abstract val input: RegularFileProperty
    @get:OutputFile abstract val output: RegularFileProperty

    @TaskAction
    fun run() = materializing {
        val result = HostGatherRewrite.rewrite(input.get().asFile.readText())
        output.get().asFile.apply { parentFile.mkdirs() }.writeText(result.mlir)
        logger.lifecycle("host-gather: @${result.function} takes %emb : ${result.embeddingType} (was a gather from ${result.table})")
    }
}

/**
 * Runs one IREE tools command (StableHLO → IREE) in the pinned container image. Input and output directories are
 * mounted separately, the input read-only; the container runs as the invoking user so outputs are not root-owned.
 */
@DisableCachingByDefault(because = "Runs an external container")
abstract class IreeToolsTask @Inject constructor(private val exec: ExecOperations) : DefaultTask() {
    /** Container image of the IREE tools compiler, from the blueprint's `toolchain`. */
    @get:Input abstract val image: Property<String>
    @get:Input @get:Optional abstract val containerRuntime: Property<String>

    protected fun runInContainer(input: File, output: File, entrypoint: String?, args: (inPath: String, outPath: String) -> List<String>) {
        output.parentFile.mkdirs()
        val runtime = containerRuntime.getOrElse("docker")
        val cmd = mutableListOf(runtime, "run", "--rm")
        uid()?.let { cmd += listOf("--user", it) }
        cmd += listOf("-v", "${input.parentFile.absolutePath}:/in:ro", "-v", "${output.parentFile.absolutePath}:/out", "-w", "/out")
        entrypoint?.let { cmd += listOf("--entrypoint", it) }
        cmd += image.get()
        cmd += args("/in/${input.name}", "/out/${output.name}")
        val log = ByteArrayOutputStream()
        val result = exec.exec {
            commandLine(cmd)
            standardOutput = log
            errorOutput = log
            isIgnoreExitValue = true
        }
        val text = log.toString()
        if (result.exitValue != 0 || !output.isFile) {
            throw GradleException(
                "IREE tools failed (exit ${result.exitValue}) — ${cmd.joinToString(" ")}\n${text.lines().takeLast(30).joinToString("\n")}\n" +
                    "The image '${image.get()}' comes from IREE tools (StableHLO → IREE). Without it, run the equivalent " +
                    "plain IREE command shown in this blueprint's README.",
            )
        }
        logger.info(text)
        logger.lifecycle("${name}: ${output.name} (${output.length() / 1024} KiB)")
    }

    private fun uid(): String? = runCatching {
        Files.getAttribute(File(System.getProperty("user.home")).toPath(), "unix:uid").toString()
    }.getOrNull()
}

/**
 * `compile`: StableHLO module → `.vmfb`.
 * `backend = "vulkan"` ⇒ `iree-compile --iree-hal-target-backends=vulkan-spirv --iree-vulkan-target=<target>`;
 * `backend = "cpu"` ⇒ `iree-compile --iree-hal-target-backends=llvm-cpu` for `target` = `host` | `arm32` | `arm64`.
 */
abstract class IreeCompileTask @Inject constructor(exec: ExecOperations) : IreeToolsTask(exec) {
    @get:InputFile @get:PathSensitive(PathSensitivity.NONE) abstract val input: RegularFileProperty
    @get:OutputFile abstract val output: RegularFileProperty
    @get:Input abstract val backend: Property<String>
    @get:Input abstract val target: Property<String>
    @get:Input @get:Optional abstract val extraArgs: ListProperty<String>
    /**
     * When set, the module's constant weights are left OUT of the `.vmfb` and referenced as parameters under this
     * scope (`--iree-opt-export-parameters=<scope>=…`) — the archive itself comes from [IreeExportParametersTask].
     * The compile then runs `iree-compile` directly with the flags the image's `compile-vulkan` / `compile-cpu`
     * wrappers use, since the wrappers do not pass the flag through.
     */
    @get:Input @get:Optional abstract val parameterScope: Property<String>

    @TaskAction
    fun run() {
        val be = backend.get()
        if (be != "vulkan" && be != "cpu") throw GradleException("IreeCompileTask.backend must be 'vulkan' or 'cpu', got '$be'")
        val scope = parameterScope.orNull
        if (scope == null) {
            runInContainer(input.get().asFile, output.get().asFile, entrypoint = null) { i, o ->
                listOf(if (be == "vulkan") "compile-vulkan" else "compile-cpu", i, "--out", o, "--target", target.get()) + extraArgs.getOrElse(emptyList())
            }
            return
        }
        // Same flags as the image's wrappers (skainet/iree-compiler entrypoint.sh), plus the parameter export; the
        // archive path is a scratch file — the real archive is produced once by IreeExportParametersTask.
        val backendFlags: String = when (be) {
            "vulkan" -> "--iree-hal-target-backends=vulkan-spirv --iree-vulkan-target=${target.get()}"
            else -> when (target.get()) {
                "host" -> "--iree-hal-target-backends=llvm-cpu"
                "arm32" -> cpuFlags("armv7a-linux-androideabi29", "cortex-a55", "+neon", "--iree-llvmcpu-stack-allocation-limit=1048576")
                "arm64" -> cpuFlags("aarch64-linux-android29", "cortex-a76", "+v8.2a,+dotprod", "")
                else -> throw GradleException("IreeCompileTask.target must be host|arm32|arm64 for backend=cpu, got '${target.get()}'")
            }
        }
        runInContainer(input.get().asFile, output.get().asFile, entrypoint = null) { i, o ->
            val extra = extraArgs.getOrElse(emptyList()).joinToString(" ")
            listOf("shell", "-c", "set -e; iree-compile $i $backendFlags --iree-opt-export-parameters=$scope=/out/.${output.get().asFile.nameWithoutExtension}.scratch.irpa $extra -o $o")
        }
    }

    private fun cpuFlags(triple: String, cpu: String, features: String, more: String): String {
        // The clang wrapper the image's compile-cpu creates on the fly: IREE calls the system linker without --target.
        // Written next to the output (mounted at /out) so the container can execute it.
        val wrapper = File(output.get().asFile.parentFile.apply { mkdirs() }, ".linker-$triple.sh")
        wrapper.writeText("#!/bin/sh\nexec clang --target=$triple -fuse-ld=lld \"$@\"\n")
        wrapper.setExecutable(true)
        return "--iree-hal-target-backends=llvm-cpu --iree-llvmcpu-link-embedded=false --iree-llvmcpu-link-static=false " +
            "--iree-llvmcpu-system-linker-path=/out/${wrapper.name} --iree-llvmcpu-target-triple=$triple --iree-llvmcpu-target-cpu=$cpu " +
            "--iree-llvmcpu-target-cpu-features=$features $more"
    }
}

/** `convert`: external parameters (safetensors) → IREE parameter archive (`iree-convert-parameters`). */
abstract class IreeConvertParametersTask @Inject constructor(exec: ExecOperations) : IreeToolsTask(exec) {
    @get:InputFile @get:PathSensitive(PathSensitivity.NONE) abstract val input: RegularFileProperty
    @get:OutputFile abstract val output: RegularFileProperty
    /** Parameter scope the exported module refers to (`#stablehlo… dense_resource<scope::name>`); `model` by convention. */
    @get:Input abstract val scope: Property<String>

    @TaskAction
    fun run() {
        runInContainer(input.get().asFile, output.get().asFile, entrypoint = "iree-convert-parameters") { i, o ->
            listOf("--parameters=${scope.get()}=$i", "--output=$o")
        }
    }
}

/**
 * `convert`: the weights an exported StableHLO module carries as constants → an IREE parameter archive
 * (`iree-compile --iree-opt-export-parameters=<scope>=<out>`). The archive does not depend on the HAL target, so
 * it is produced once (with a host CPU compile whose `.vmfb` is discarded) and shared by every target and by
 * every graph exported from the same weights — graphs that share weights produce byte-identical archives.
 * The modules compiled for the device then reference the archive by [scope] instead of embedding the weights.
 */
abstract class IreeExportParametersTask @Inject constructor(exec: ExecOperations) : IreeToolsTask(exec) {
    @get:InputFile @get:PathSensitive(PathSensitivity.NONE) abstract val input: RegularFileProperty
    @get:OutputFile abstract val output: RegularFileProperty
    /** Parameter scope the archive is written under; `model` by convention. */
    @get:Input abstract val scope: Property<String>

    @TaskAction
    fun run() {
        runInContainer(input.get().asFile, output.get().asFile, entrypoint = null) { i, o ->
            listOf("compile", "--", i, "--iree-hal-target-backends=llvm-cpu", "--iree-opt-export-parameters=${scope.get()}=$o", "-o", "/out/.${output.get().asFile.nameWithoutExtension}-host.vmfb")
        }
    }
}

