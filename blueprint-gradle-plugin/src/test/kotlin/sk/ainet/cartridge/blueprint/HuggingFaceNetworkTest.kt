package sk.ainet.cartridge.blueprint

import sk.ainet.cartridge.blueprint.core.Digests
import sk.ainet.cartridge.blueprint.core.SourceFetcher
import sk.ainet.cartridge.blueprint.model.MaterializationException
import sk.ainet.cartridge.blueprint.model.Redistribution
import sk.ainet.cartridge.blueprint.model.Source
import sk.ainet.cartridge.blueprint.model.SourceFile
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Real `hf://` downloads through SKaiNET's data-source module. Opt-in — `BLUEPRINT_NETWORK_TESTS=1` — so the
 * default build stays offline. Pins two small files of an MIT, ungated checkpoint at an immutable revision.
 */
class HuggingFaceNetworkTest {
    private val enabled = System.getenv("BLUEPRINT_NETWORK_TESTS") == "1"

    private val source = Source(
        name = "moonshine-de",
        role = "weights",
        uri = "hf://moonshine-ai/moonshine-streaming-tiny-de",
        revision = "928bfb925992902b7e187fda7921d72114142e1f",
        files = listOf(
            SourceFile("config.json", "sha256:6c49ec398e57e1c3146c77772089287746fe3f6209ad2579c70bc8e8eb304424", 1495),
            SourceFile("tokenizer.json", "sha256:983ec5f1b76300b74be6a764edc034a16e2912c38a8a8c0e4d51af355b6b755d", 588216),
        ),
        license = "MIT",
        licenseUrl = "https://huggingface.co/moonshine-ai/moonshine-streaming-tiny-de",
        redistribution = Redistribution.ALLOWED,
    )

    @Test
    fun `fetches a pinned revision from the hub, verifies it, and serves it from cache offline`() {
        if (!enabled) return
        val cache = Files.createTempDirectory("hf-cache").toFile()
        val log = mutableListOf<String>()

        val files = SourceFetcher(cacheDir = cache, log = { log += it }).fetch(source)
        for (f in source.files) assertEquals(f.digest, Digests.sha256(files.getValue(f.path)), f.path)
        assertTrue(log.all { it.contains("fetched") }, log.toString())

        log.clear()
        val again = SourceFetcher(cacheDir = cache, offline = true, log = { log += it }).fetch(source)
        assertEquals(files.keys, again.keys)
        assertTrue(log.all { it.contains("cached") }, "an offline build must be served from the cache: $log")
    }

    @Test
    fun `a wrong pin is refused and nothing unverified is kept`() {
        if (!enabled) return
        val cache = Files.createTempDirectory("hf-cache").toFile()
        val wrong = source.copy(files = listOf(source.files.first().copy(digest = "sha256:" + "0".repeat(64))))
        val e = assertFailsWith<MaterializationException> { SourceFetcher(cacheDir = cache).fetch(wrong) }
        assertContains(e.message!!, "Digest mismatch")
        // The correct pin must still work afterwards from the same cache directory: no poisoned entry.
        val ok = SourceFetcher(cacheDir = cache).fetch(source.copy(files = listOf(source.files.first())))
        assertEquals(source.files.first().digest, Digests.sha256(ok.getValue("config.json")))
    }

    @Test
    fun `a file that does not exist at the revision is reported as such`() {
        if (!enabled) return
        val missing = source.copy(files = listOf(SourceFile("no-such-file.bin", "sha256:" + "1".repeat(64))))
        val e = assertFailsWith<MaterializationException> { SourceFetcher(cacheDir = Files.createTempDirectory("hf-cache").toFile()).fetch(missing) }
        assertContains(e.message!!, "no-such-file.bin")
    }
}
