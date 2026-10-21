package sk.ainet.cartridge.asr.whisper

import java.io.File
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class PackDirTest {
    // The manifest WhisperExportCli writes for an English-only tiny checkpoint at audioCtx 200 / 48 positions.
    private val contractEn = """
        {"contractVersion": 2,
         "functions": {"encoder": "whisper_encoder", "prefill": "whisper_prefill", "step": "whisper_step"},
         "nLayers": 4, "dModel": 384, "audioCtx": 200, "maxPositions": 48, "vocab": 51864, "eot": 50256,
         "promptIds": [50257,50362], "melFrames": 400, "parameters": "params.irpa", "parameterScope": "model",
         "prefillArgs": ["promptIds","feat"], "prefillOutputs": ["logits"], "stepArgs": ["tok","pos","addMask","wf"], "stepOutputs": ["logits"]}
    """.trimIndent()

    private fun pack(languages: List<String>, accelerator: Boolean): File {
        val root = Files.createTempDirectory("whisper-pack").toFile()
        val target = if (accelerator) """{"hardware":"IREE-armeabi-v7a","abi":"armeabi-v7a","accelerator":"Vulkan GPU (valhall4)"}""" else """{"hardware":"IREE-armeabi-v7a","abi":"armeabi-v7a"}"""
        File(root, "descriptor.json").writeText("""{"id":"asr-whisper-iree","target":$target,"attributes":{"languages":[${languages.joinToString(",") { "\"$it\"" }}]}}""")
        for (l in languages) {
            for (g in listOf("encoder", "prefill", "step")) File(root, "artifacts/model/$l/$g.vmfb").apply { parentFile.mkdirs() }.writeBytes(byteArrayOf(1))
            File(root, "artifacts/weights/$l/params.irpa").apply { parentFile.mkdirs() }.writeBytes(byteArrayOf(2))
            File(root, "artifacts/other/$l/whisper-contract.json").apply { parentFile.mkdirs() }.writeText(contractEn)
            File(root, "artifacts/tokenizer/$l/tokenizer.json").apply { parentFile.mkdirs() }.writeText("{}")
        }
        return root
    }

    @Test
    fun locatesTheSixFilesOfAFlavor() {
        val p = PackDir(pack(listOf("en", "de"), accelerator = true).path)
        assertEquals("asr-whisper-iree", p.id)
        assertTrue(p.targetsAccelerator)
        assertEquals(listOf("en", "de"), p.languages)
        val de = p.model("de")
        assertEquals(6, de.all.size)
        assertTrue(de.all.all { it.toString().contains("/de/") })
        assertTrue(de.stepVmfb.toString().endsWith("artifacts/model/de/step.vmfb"))
        assertTrue(de.paramsIrpa.toString().endsWith("artifacts/weights/de/params.irpa"))
    }

    @Test
    fun readsTheGraphContract() {
        val c = PackDir(pack(listOf("en"), accelerator = false).path).contract("en")
        assertEquals("whisper_prefill", c.prefillFunction)
        assertEquals("model", c.parameterScope)
        assertEquals(listOf(50257, 50362), c.promptIds)
        assertEquals(50256, c.eot)
        assertEquals(400, c.melFrames)
        assertEquals(48, c.maxPositions)
    }

    @Test
    fun anotherContractVersionIsRejected() {
        val e = assertFailsWith<IllegalArgumentException> { PackDir.Contract.parse(contractEn.replace("\"contractVersion\": 2", "\"contractVersion\": 3")) }
        assertTrue(e.message!!.contains("version 3"))
    }

    @Test
    fun aMissingFlavorOrFileIsReportedByName() {
        val p = PackDir(pack(listOf("en"), accelerator = false).path)
        assertTrue(!p.targetsAccelerator)
        val e = assertFailsWith<IllegalArgumentException> { p.model("de") }
        assertTrue(e.message!!.contains("artifacts/model/de/encoder.vmfb"))
        assertFailsWith<IllegalArgumentException> { PackDir(Files.createTempDirectory("empty").toString()) }
    }
}
