package sk.ainet.cartridge.generator.lfm2.worker

import kotlinx.serialization.json.*
import java.nio.file.Files
import java.security.MessageDigest
import kotlin.test.*

class ConfigurationTest {
    @Test fun sidefilePinsAllSixMetadataFilesAndKeepsPackagePathsConfined() {
        val root = Files.createTempDirectory("worker-pack-").toRealPath()
        try {
            val model = Files.createDirectories(root.resolve("artifacts/model"))
            val runtime = Files.createDirectories(root.resolve("artifacts/runtime"))
            val hashes = METADATA_FILES.associateWith { name ->
                val bytes = "test-$name".toByteArray()
                Files.write(model.resolve(name), bytes)
                MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
            }
            Files.write(model.resolve("model.safetensors"), byteArrayOf(1))
            fun config(modelPath: String = "artifacts/model", metadata: Map<String, String> = hashes) = buildJsonObject {
                put("model_dir", modelPath); put("checkpoint_sha256", "0".repeat(64))
                put("metadata_sha256", JsonObject(metadata.mapValues { JsonPrimitive(it.value) }))
                put("decoder_policy", "CHECKPOINT"); put("default_max_output_tokens", 32); put("max_output_tokens", 256)
            }
            val sidefile = runtime.resolve("worker.json")
            Files.writeString(sidefile, config().toString())
            val settings = WorkerConfig.load(root)
            assertEquals(hashes, settings.metadataSha256)
            assertEquals(32, settings.defaultMaxOutputTokens)
            Files.writeString(sidefile, config("../outside").toString())
            assertFailsWith<IllegalArgumentException> { WorkerConfig.load(root) }
            Files.writeString(sidefile, config(metadata = hashes - "LICENSE").toString())
            assertFailsWith<IllegalArgumentException> { WorkerConfig.load(root) }
            Files.writeString(sidefile, config().toString())
            Files.writeString(model.resolve("LICENSE"), "changed")
            assertFailsWith<IllegalArgumentException> { WorkerConfig.load(root) }
        } finally { root.toFile().deleteRecursively() }
    }

    @Test fun gpuPackageRequiresExplicitBackendPythonDeviceAndCompiledCapacityWithoutOriginalWeights() {
        val root = Files.createTempDirectory("worker-gpu-pack-").toRealPath()
        try {
            val metadata = Files.createDirectories(root.resolve("artifacts/metadata"))
            Files.createDirectories(root.resolve("artifacts/gpu")); Files.createDirectories(root.resolve("artifacts/python"))
            val sidefile = Files.createDirectories(root.resolve("artifacts/runtime")).resolve("worker.json")
            val hashes = METADATA_FILES.associateWith { name ->
                Files.writeString(metadata.resolve(name), "test-$name"); hash(metadata.resolve(name))
            }
            val python = root.resolve("external-python").toFile().also { it.writeText("#!/bin/sh\nexit 0\n"); it.setExecutable(true) }
            fun config(device: String = "vulkan://test-device", maximum: Int = 32) = buildJsonObject {
                put("backend", "iree-vulkan"); put("metadata_dir", "artifacts/metadata")
                put("checkpoint_sha256", "0".repeat(64)); put("metadata_sha256", JsonObject(hashes.mapValues { JsonPrimitive(it.value) }))
                put("decoder_policy", "CHECKPOINT"); put("default_max_output_tokens", 8); put("max_output_tokens", maximum)
                put("bundle_dir", "artifacts/gpu"); put("python_package_dir", "artifacts/python")
                put("python_executable_env", "LFM2_PYTHON_EXECUTABLE"); put("device_uri", device)
                put("context_window_tokens", 64)
            }
            Files.writeString(sidefile, config().toString())
            val settings = WorkerConfig.load(root) { python.absolutePath }
            assertEquals("iree-vulkan", settings.backend); assertEquals(64, settings.contextWindowTokens)
            assertEquals("vulkan://test-device", settings.ireeOptions!!.deviceUri)
            assertFalse(Files.exists(metadata.resolve("model.safetensors")))
            assertFailsWith<IllegalArgumentException> { WorkerConfig.load(root) { null } }
            assertFailsWith<IllegalArgumentException> { WorkerConfig.load(root) { "python3" } }
            Files.writeString(sidefile, config("local-task://").toString())
            assertFailsWith<IllegalArgumentException> { WorkerConfig.load(root) { python.absolutePath } }
            Files.writeString(sidefile, config(maximum = 65).toString())
            assertFailsWith<IllegalArgumentException> { WorkerConfig.load(root) { python.absolutePath } }
        } finally { root.toFile().deleteRecursively() }
    }

    @Test fun strictParserRejectsAmbiguousKeysAndDeepInputsButPreservesValidUnicode() {
        assertFailsWith<IllegalArgumentException> { strictObject("""{"a":1,"\u0061":2}""".toByteArray()) }
        assertFailsWith<IllegalArgumentException> { strictObject(("{\"x\":" + "[".repeat(66) + "0" + "]".repeat(66) + "}").toByteArray()) }
        val valid = """{"messages":[{"role":"user","content":"é 😀"}],"temperature":0,"top_k":1,"top_p":1}"""
        assertEquals("é 😀", RunRequest.parse(valid.toByteArray(), 32, 256).messages.single().content)
        assertFailsWith<IllegalArgumentException> { RunRequest.parse(valid.replace("\"top_p\":1", "\"top_p\":\"1\"").toByteArray(), 32, 256) }
    }
}
