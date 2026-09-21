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

    @TaskAction
    fun run() {
        val sub = when (backend.get()) {
            "vulkan" -> "compile-vulkan"
            "cpu" -> "compile-cpu"
            else -> throw GradleException("IreeCompileTask.backend must be 'vulkan' or 'cpu', got '${backend.get()}'")
        }
        runInContainer(input.get().asFile, output.get().asFile, entrypoint = null) { i, o ->
            listOf(sub, i, "--out", o, "--target", target.get()) + extraArgs.getOrElse(emptyList())
        }
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
