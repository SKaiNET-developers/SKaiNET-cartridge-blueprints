package sk.ainet.cartridge.generator.lfm2.worker

import sk.ainet.models.lfm2.*
import sk.ainet.apps.klfm2.*
import sk.ainet.cartridge.generator.lfm2.Lfm2LlmEngine
import kotlinx.serialization.json.*
import sk.ainet.cartridge.profiles.llm.*
import java.nio.file.Files
import java.nio.file.Path
import java.security.MessageDigest

internal val METADATA_FILES = setOf("config.json", "generation_config.json", "tokenizer_config.json",
    "tokenizer.json", "chat_template.jinja", "LICENSE")

internal fun JsonElement.stringValue(): String {
    require(this is JsonPrimitive && isString) { "Expected a string" }
    return content
}
internal fun JsonElement.integerValue(): Int {
    require(this is JsonPrimitive && !isString) { "Expected an integer" }
    return requireNotNull(intOrNull) { "Expected an integer" }
}
internal fun JsonElement.numberValue(): Double {
    require(this is JsonPrimitive && !isString) { "Expected a number" }
    return requireNotNull(doubleOrNull?.takeIf(Double::isFinite)) { "Expected a finite number" }
}
internal fun hash(path: Path): String {
    val digest = MessageDigest.getInstance("SHA-256")
    Files.newInputStream(path).buffered().use { input ->
        val buffer = ByteArray(1024 * 1024)
        while (true) {
            val length = input.read(buffer)
            if (length < 0) break
            digest.update(buffer, 0, length)
        }
    }
    return digest.digest().joinToString("") { "%02x".format(it) }
}

internal fun confined(root: Path, relative: String): Path {
    val item = Path.of(relative)
    require(!item.isAbsolute && item.nameCount > 0 && item.none { it.toString() in setOf(".", "..") }) {
        "Package paths must be relative and cannot traverse parents"
    }
    val resolved = root.resolve(item).toRealPath()
    require(resolved.startsWith(root)) { "Package path resolves outside the package" }
    return resolved
}

internal data class WorkerConfig(
    val modelDirectory: Path,
    val checkpointSha256: String,
    val metadataSha256: Map<String, String>,
    val decoderPolicy: Lfm2DecodingPolicy,
    val defaultMaxOutputTokens: Int,
    val maxOutputTokens: Int,
    val backend: String = "eager-cpu",
    val ireeOptions: Lfm2IreeOptions? = null,
    val contextWindowTokens: Int? = null,
) {
    companion object {
        fun load(packDirectory: Path, environment: (String) -> String? = System::getenv): WorkerConfig {
            val root = packDirectory.toRealPath()
            require(Files.isDirectory(root)) { "Package root must be a directory" }
            val sidefile = confined(root, "artifacts/runtime/worker.json")
            require(Files.size(sidefile) <= MAX_PAYLOAD) { "Worker sidefile is too large" }
            val json = strictObject(Files.readAllBytes(sidefile))
            val backend = json["backend"]?.stringValue() ?: "eager-cpu"
            require(backend in setOf("eager-cpu", "iree-vulkan")) { "Unsupported package backend" }
            val common = setOf("checkpoint_sha256", "metadata_sha256", "decoder_policy",
                "default_max_output_tokens", "max_output_tokens")
            val expected = if (backend == "iree-vulkan") common + setOf("backend", "metadata_dir", "bundle_dir",
                "python_package_dir", "python_executable_env", "device_uri", "context_window_tokens")
                else common + "model_dir" + (if ("backend" in json) setOf("backend") else emptySet())
            require(json.keys == expected) { "Unexpected worker sidefile fields" }
            val model = confined(root, json.getValue(if (backend == "iree-vulkan") "metadata_dir" else "model_dir").stringValue())
            require(Files.isDirectory(model)) { "Model path must be a directory" }
            fun sha(value: JsonElement): String = value.stringValue().also {
                require(it.matches(Regex("[0-9a-f]{64}"))) { "Invalid SHA-256 pin" }
            }
            val metadata = json.getValue("metadata_sha256") as? JsonObject
                ?: throw IllegalArgumentException("metadata_sha256 must be an object")
            require(metadata.keys == METADATA_FILES) { "All six original metadata files must be pinned" }
            val hashes = metadata.mapValues { sha(it.value) }
            for ((name, expected) in hashes) {
                val path = model.resolve(name).toRealPath()
                require(path.startsWith(root) && Files.isRegularFile(path)) { "Metadata path escapes the package" }
                require(hash(path) == expected) { "Metadata SHA-256 mismatch: $name" }
            }
            if (backend == "eager-cpu") {
                val checkpoint = model.resolve("model.safetensors").toRealPath()
                require(checkpoint.startsWith(root) && Files.isRegularFile(checkpoint)) { "Checkpoint path escapes the package" }
            }
            val maximum = json.getValue("max_output_tokens").integerValue()
            val default = json.getValue("default_max_output_tokens").integerValue()
            require(maximum > 0 && default in 1..maximum) { "Invalid worker output budget" }
            val capacity = if (backend == "iree-vulkan") json.getValue("context_window_tokens").integerValue().also {
                require(it > 0 && maximum <= it) { "Output limit exceeds the compiled context capacity" }
            } else null
            val options = if (backend == "iree-vulkan") {
                val variable = json.getValue("python_executable_env").stringValue()
                require(variable == "LFM2_PYTHON_EXECUTABLE") { "Only the documented external Python variable is supported" }
                val python = requireNotNull(environment(variable)?.takeIf { it.isNotBlank() }) {
                    "Set LFM2_PYTHON_EXECUTABLE to an external Python 3.12 executable"
                }
                require(Path.of(python).isAbsolute) { "External Python executable must be absolute" }
                Lfm2IreeOptions(confined(root, json.getValue("bundle_dir").stringValue()).toFile(),
                    Path.of(python).toRealPath().toFile(), json.getValue("device_uri").stringValue(),
                    confined(root, json.getValue("python_package_dir").stringValue()).toFile())
            } else null
            return WorkerConfig(model, sha(json.getValue("checkpoint_sha256")), hashes,
                Lfm2DecodingPolicy.valueOf(json.getValue("decoder_policy").stringValue()), default, maximum,
                backend, options, capacity)
        }
    }
}

internal data class RunRequest(val messages: List<LlmMessage>, val maximum: Int) {
    companion object {
        fun parse(bytes: ByteArray, defaultMaximum: Int, maximum: Int): RunRequest {
            val json = strictObject(bytes)
            require(json.keys.all { it in setOf("messages", "max_output_tokens", "temperature", "top_k", "top_p") }) {
                "Unsupported request fields"
            }
            require(json["temperature"]?.numberValue()?.let { it == 0.0 } != false) { "Only greedy temperature=0 is supported" }
            require(json["top_k"]?.integerValue()?.let { it == 1 } != false) { "Only greedy top_k=1 is supported" }
            require(json["top_p"]?.numberValue()?.let { it == 1.0 } != false) { "Only greedy top_p=1 is supported" }
            val budget = json["max_output_tokens"]?.integerValue() ?: defaultMaximum
            require(budget in 1..maximum) { "Output budget exceeds the package limit" }
            val source = json["messages"] as? JsonArray ?: throw IllegalArgumentException("messages must be an array")
            require(source.isNotEmpty()) { "Conversation must not be empty" }
            val messages = source.mapIndexed { index, element ->
                val item = element as? JsonObject ?: throw IllegalArgumentException("Each message must be an object")
                require(item.keys == setOf("role", "content")) { "Messages require role and content only" }
                val role = when (item.getValue("role").stringValue()) {
                    "system" -> { require(index == 0) { "System message must be first" }; LlmRole.SYSTEM }
                    "user" -> LlmRole.USER
                    "assistant" -> LlmRole.ASSISTANT
                    else -> throw IllegalArgumentException("Unsupported message role")
                }
                LlmMessage(role, item.getValue("content").stringValue())
            }
            return RunRequest(messages, budget)
        }
    }
}

internal class WorkerRuntime(
    val engine: LlmEngine,
    val promptIds: (List<LlmMessage>) -> IntArray,
    val metadata: JsonObject,
    val defaultMaximum: Int,
    val maximum: Int,
    private val closeBackend: () -> Unit = {},
) : AutoCloseable {
    override fun close() { try { engine.close() } finally { closeBackend() } }
}

internal fun loadRuntime(packDirectory: Path): WorkerRuntime {
    val settings = WorkerConfig.load(packDirectory)
    val directory = settings.modelDirectory.toFile()
    // The existing loader additionally enforces the immutable checkpoint metadata pins.
    val frontend = loadLfm2TextFrontend(directory)
    require(frontend.metadataHashes == settings.metadataSha256) { "Frontend metadata differs from package pins" }
    if (settings.backend == "iree-vulkan") return loadIreeRuntime(settings, frontend)
    val config = Lfm2Config.parse(directory.resolve("config.json").readText())
    val limit = config.toDimensions().contextLimit
    require(settings.maxOutputTokens in 1..limit) { "Package output limit exceeds model context" }
    val loaded = loadLfm2Model(config, directory.resolve("model.safetensors"),
        expectedCheckpointSha256 = settings.checkpointSha256)
    val engine = Lfm2LlmEngine(loaded.model, frontend, settings.decoderPolicy, settings.defaultMaxOutputTokens)
    val metadata = buildJsonObject {
        put("protocol_version", 1); put("task_profile", "llm/v1"); put("model", "LiquidAI/LFM2.5-350M")
        put("checkpoint_sha256", loaded.checkpointSha256)
        put("metadata_sha256", JsonObject(settings.metadataSha256.mapValues { JsonPrimitive(it.value) }))
        put("decoder_policy", settings.decoderPolicy.name)
        put("context_window_tokens", limit); put("default_max_output_tokens", settings.defaultMaxOutputTokens)
        put("max_output_tokens", settings.maxOutputTokens); put("loaded_tensors", loaded.records.size)
        put("worker_pid", ProcessHandle.current().pid())
    }
    return WorkerRuntime(engine, { messages ->
        frontend.encodeMessages(messages.map { Lfm2ChatMessage(it.role.name.lowercase(), it.content) })
    }, metadata, settings.defaultMaxOutputTokens, settings.maxOutputTokens)
}

/** Dependency readiness only: metadata and runtime/device checks, with no model weights loaded. */
internal fun preflightRuntime(packDirectory: Path): JsonObject {
    val settings = WorkerConfig.load(packDirectory)
    val probe = settings.ireeOptions?.let { Lfm2IreeGenerationRuntime.preflight(it) }
    return buildJsonObject {
        put("status", "ready"); put("backend", settings.backend); put("model_loaded", false)
        probe?.let { put("device_probe", it) }
    }
}

private fun loadIreeRuntime(settings: WorkerConfig, frontend: Lfm2TextFrontend): WorkerRuntime {
    val runtime = Lfm2IreeGenerationRuntime(requireNotNull(settings.ireeOptions))
    try {
        require(runtime.capacity == settings.contextWindowTokens) { "Package context differs from the compiled GPU capacity" }
        require(runtime.vocabularySize == frontend.vocabularySize) { "GPU and tokenizer vocabularies differ" }
        val engine = Lfm2LlmEngine(runtime.textGenerator(frontend, settings.decoderPolicy), frontend,
            runtime.capacity, "LFM2.5-350M (IREE Vulkan)", settings.defaultMaxOutputTokens)
        val metadata = buildJsonObject {
            put("protocol_version", 1); put("task_profile", "llm/v1"); put("model", "LiquidAI/LFM2.5-350M")
            put("backend", "iree-vulkan"); put("cpu_model_fallback", false)
            put("checkpoint_sha256", settings.checkpointSha256)
            put("metadata_sha256", JsonObject(settings.metadataSha256.mapValues { JsonPrimitive(it.value) }))
            put("decoder_policy", settings.decoderPolicy.name); put("context_window_tokens", runtime.capacity)
            put("default_max_output_tokens", settings.defaultMaxOutputTokens); put("max_output_tokens", settings.maxOutputTokens)
            put("worker_pid", ProcessHandle.current().pid()); put("iree_worker", runtime.workerInfo)
            put("runtime_source_pins", runtime.implementationSourcePins)
        }
        return WorkerRuntime(engine, { messages ->
            frontend.encodeMessages(messages.map { Lfm2ChatMessage(it.role.name.lowercase(), it.content) })
        }, metadata, settings.defaultMaxOutputTokens, settings.maxOutputTokens, runtime::close)
    } catch (failure: Throwable) { runtime.close(); throw failure }
}
