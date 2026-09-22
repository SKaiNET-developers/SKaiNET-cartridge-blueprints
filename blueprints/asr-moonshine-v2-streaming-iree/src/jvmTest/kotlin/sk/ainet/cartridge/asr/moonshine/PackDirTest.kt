package sk.ainet.cartridge.asr.moonshine

import java.io.File
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class PackDirTest {
    private fun pack(languages: List<String>, accelerator: Boolean): File {
        val root = Files.createTempDirectory("moonshine-pack").toFile()
        val target = if (accelerator) """{"hardware":"IREE-armeabi-v7a","abi":"armeabi-v7a","accelerator":"Vulkan GPU (valhall4)"}""" else """{"hardware":"IREE-armeabi-v7a","abi":"armeabi-v7a"}"""
        File(root, "descriptor.json").writeText("""{"id":"asr-moonshine-v2-streaming-iree","target":$target,"attributes":{"languages":[${languages.joinToString(",") { "\"$it\"" }}]}}""")
        for (l in languages) {
            for (g in listOf("frontend", "encoder", "adapter", "prefill", "step")) File(root, "artifacts/model/$l/$g.vmfb").apply { parentFile.mkdirs() }.writeBytes(byteArrayOf(1))
            File(root, "artifacts/weights/$l/params.irpa").apply { parentFile.mkdirs() }.writeBytes(byteArrayOf(2))
            File(root, "artifacts/weights/$l/dec_embed.bin").writeBytes(byteArrayOf(3))
            File(root, "artifacts/tokenizer/$l/vocab.bin").apply { parentFile.mkdirs() }.writeBytes(byteArrayOf(4))
        }
        return root
    }

    @Test
    fun locatesTheEightFilesOfAFlavor() {
        val p = PackDir(pack(listOf("en", "de"), accelerator = true).path)
        assertEquals("asr-moonshine-v2-streaming-iree", p.id)
        assertTrue(p.targetsAccelerator)
        assertEquals(listOf("en", "de"), p.languages)
        val de = p.model("de")
        assertEquals(8, de.all.size)
        assertTrue(de.all.all { it.toString().contains("/de/") })
        assertTrue(de.stepVmfb.toString().endsWith("artifacts/model/de/step.vmfb"))
        assertTrue(de.paramsIrpa.toString().endsWith("artifacts/weights/de/params.irpa"))
    }

    @Test
    fun aMissingFlavorOrFileIsReportedByName() {
        val p = PackDir(pack(listOf("en"), accelerator = false).path)
        assertTrue(!p.targetsAccelerator)
        val e = assertFailsWith<IllegalArgumentException> { p.model("de") }
        assertTrue(e.message!!.contains("artifacts/model/de/frontend.vmfb"))
        assertFailsWith<IllegalArgumentException> { PackDir(Files.createTempDirectory("empty").toString()) }
    }
}
