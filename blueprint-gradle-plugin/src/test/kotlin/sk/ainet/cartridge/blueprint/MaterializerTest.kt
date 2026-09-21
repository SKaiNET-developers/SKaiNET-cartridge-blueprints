package sk.ainet.cartridge.blueprint

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import sk.ainet.cartridge.blueprint.core.CanonicalJson
import sk.ainet.cartridge.blueprint.core.Digests
import sk.ainet.cartridge.blueprint.core.Documents
import sk.ainet.cartridge.blueprint.core.EffectiveLicense
import sk.ainet.cartridge.blueprint.core.LicenseGate
import sk.ainet.cartridge.blueprint.core.Materializer
import sk.ainet.cartridge.blueprint.core.Selection
import sk.ainet.cartridge.blueprint.core.SourceFetcher
import sk.ainet.cartridge.blueprint.core.SpecSchema
import sk.ainet.cartridge.blueprint.model.MaterializationException
import sk.ainet.data.source.CachePolicy
import sk.ainet.data.source.DataSourceException
import sk.ainet.data.source.DataSourceRequest
import sk.ainet.data.source.DataSourceResolver
import sk.ainet.data.source.JvmDataSourceResolver
import java.io.File
import java.security.Signature
import java.util.Base64
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class MaterializerTest {

    private fun materialize(f: Fixture, key: String? = null): Materializer.Result {
        val blueprint = Documents.blueprint(f.blueprintFile)
        val profile = Documents.profile(f.profileFile)
        val fetcher = SourceFetcher(File(f.root, "cache"))
        val fetched = Selection.sources(blueprint.value, profile.value).associate { it.name to fetcher.fetch(it) }
        return Materializer(blueprint, profile, fetched, f.productMap, "test-materializer@0").materialize(File(f.root, "out/pack"), key)
    }

    // --- the spec's own worked example ----------------------------------------------------------------

    @Test
    fun `spec worked blueprint and profile load and match`() {
        val dir = File(javaClass.getResource("/spec-examples")!!.toURI())
        val blueprint = Documents.blueprint(File(dir, "nlu-functiongemma-270m-iree.blueprint.json"))
        val profile = Documents.profile(File(dir, "nlu-functiongemma-270m-iree.vulkan-arm64.profile.json"))
        assertEquals("nlu-functiongemma-270m-iree", blueprint.value.id)
        assertEquals(listOf("weights", "tokenizer"), Selection.sources(blueprint.value, profile.value).map { it.name })
        // The example profile pins a placeholder digest, so the match must be refused — same version, different bytes.
        val e = assertFailsWith<MaterializationException> { Documents.requireMatch(blueprint, profile) }
        assertContains(e.message!!, "digest mismatch")
    }

    @Test
    fun `schema violations are reported with their location`() {
        val f = Fixture()
        f.blueprintFile.writeText(f.blueprintFile.readText().replace(""""io":{""", """"performance":{"measured_on":"nope"},"io":{"""))
        val e = assertFailsWith<MaterializationException> { Documents.blueprint(f.blueprintFile) }
        assertContains(e.message!!, "descriptor_template")
    }

    // --- licensing --------------------------------------------------------------------------------------

    @Test
    fun `acceptance-required source without acceptance stops before any fetch`() {
        val f = Fixture(accept = false)
        val blueprint = Documents.blueprint(f.blueprintFile).value
        val profile = Documents.profile(f.profileFile).value
        val e = assertFailsWith<MaterializationException> { LicenseGate.check(blueprint, profile) }
        assertContains(e.message!!, "Toy Terms of Use")
        assertContains(e.message!!, "will not accept a license for you")
    }

    @Test
    fun `acceptance must name the exact license`() {
        val f = Fixture()
        f.profileFile.writeText(f.profileFile.readText().replace(""""license":"Toy Terms of Use"""", """"license":"Some Other Terms""""))
        assertFailsWith<MaterializationException> {
            LicenseGate.check(Documents.blueprint(f.blueprintFile).value, Documents.profile(f.profileFile).value)
        }
    }

    @Test
    fun `effective license states every obligation and never hides a proprietary input`() {
        assertEquals("MIT", EffectiveLicense.compute("MIT", listOf("MIT")))
        assertEquals("MIT AND Apache-2.0", EffectiveLicense.compute("MIT", listOf("Apache-2.0", "MIT")))
        assertEquals("MIT AND LicenseRef-Gemma-Terms-of-Use", EffectiveLicense.compute("MIT", listOf("Gemma Terms of Use")))
        assertEquals("proprietary", EffectiveLicense.compute("MIT", listOf("Gemma Terms of Use", "proprietary")))
        assertEquals("NOASSERTION", EffectiveLicense.compute("MIT", listOf("NOASSERTION", "proprietary")))
    }

    // --- fetching ---------------------------------------------------------------------------------------

    @Test
    fun `fetch verifies digests and refuses changed bytes`() {
        val f = Fixture()
        val source = Documents.blueprint(f.blueprintFile).value.source("weights")!!
        val fetcher = SourceFetcher(File(f.root, "cache"))

        val first = fetcher.fetch(source).getValue("model.bin")
        assertEquals(Digests.sha256(f.weights), Digests.sha256(first))

        f.weights.appendBytes(byteArrayOf(42))          // the "publisher" changed the file at the same revision
        val e = assertFailsWith<MaterializationException> { fetcher.fetch(source) }
        assertContains(e.message!!, "Digest mismatch")
        assertContains(e.message!!, "do not proceed")
    }

    @Test
    fun `a mirror changes where bytes come from, never which bytes`() {
        val f = Fixture()
        val source = Documents.blueprint(f.blueprintFile).value.source("weights")!!
        val mirror = File(f.root, "mirror").apply { mkdirs() }
        f.weights.copyTo(File(mirror, "model.bin"))
        f.weights.delete()                                 // only the mirror has it now
        val ok = SourceFetcher(File(f.root, "cache"), mirrors = mapOf("weights" to mirror.toURI().toString())).fetch(source)
        assertEquals(source.files.single().digest, Digests.sha256(ok.getValue("model.bin")))

        File(mirror, "model.bin").writeText("tampered")
        assertFailsWith<MaterializationException> {
            SourceFetcher(File(f.root, "cache2"), mirrors = mapOf("weights" to mirror.toURI().toString())).fetch(source)
        }
    }

    @Test
    fun `requests use skainet hf uris pinned to the revision, and plain http is refused`() {
        val f = Fixture()
        val source = Documents.blueprint(f.blueprintFile).value.source("weights")!!.copy(uri = "hf://org/model", revision = "abc123")
        val fetcher = SourceFetcher(File(f.root, "cache"))
        assertEquals("hf://org/model@abc123/model.bin", fetcher.requestUri(source, source.files.single()))
        assertEquals(
            "https://mirror.example.org/m/model.bin",
            SourceFetcher(File(f.root, "cache"), mirrors = mapOf("weights" to "https://mirror.example.org/m/")).requestUri(source, source.files.single()),
        )
        val e = assertFailsWith<MaterializationException> { fetcher.requestUri(source.copy(uri = "http://insecure.example/x"), source.files.single()) }
        assertContains(e.message!!, "plain http is not fetched")
        assertFailsWith<MaterializationException> { fetcher.requestUri(source, source.files.single().copy(path = "../outside.bin")) }
    }

    @Test
    fun `a stale cache entry is refreshed once, offline builds never touch the network`() {
        val f = Fixture()
        val source = Documents.blueprint(f.blueprintFile).value.source("weights")!!.copy(uri = "hf://org/model", revision = "abc123")
        val policies = mutableListOf<CachePolicy>()
        val real = JvmDataSourceResolver(File(f.root, "cache"))
        val resolver = object : DataSourceResolver {
            override suspend fun resolve(request: DataSourceRequest) = run {
                policies += request.cachePolicy
                when (request.cachePolicy) {
                    CachePolicy.Use -> throw DataSourceException("SHA-256 mismatch for ${request.uri}: expected x, actual y")
                    CachePolicy.Offline -> throw DataSourceException("No cached artifact available for offline source: ${request.uri}")
                    else -> real.resolve(DataSourceRequest(f.weights.path, expectedSha256 = request.expectedSha256))
                }
            }
        }
        val fetched = SourceFetcher(resolver).fetch(source).getValue("model.bin")
        assertEquals(listOf(CachePolicy.Use, CachePolicy.Refresh), policies)
        assertEquals(Digests.sha256(f.weights), Digests.sha256(fetched))

        policies.clear()
        val e = assertFailsWith<MaterializationException> { SourceFetcher(resolver, offline = true).fetch(source) }
        assertEquals(listOf(CachePolicy.Offline), policies)
        assertContains(e.message!!, "the build is offline")
    }

    // --- materialization --------------------------------------------------------------------------------

    @Test
    fun `materializes a schema-valid, signed pack_dir whose manifest names the blueprint`() {
        val f = Fixture(inputLicense = "proprietary")
        val (privatePem, publicKey) = Fixture.keyPair()
        val result = materialize(f, privatePem)
        val pack = result.packDir

        assertEquals(
            setOf("descriptor.json", "manifest.json", "artifacts/runtime/libtoy-ctg.so", "artifacts/model/prefill.vmfb",
                "artifacts/model/decode.vmfb", "artifacts/weights/model.bin", "artifacts/tokenizer/tokenizer.json",
                "artifacts/other/tool-catalog.json"),
            pack.walkTopDown().filter { it.isFile }.map { it.relativeTo(pack).invariantSeparatorsPath }.toSet(),
        )

        val descriptor = result.descriptor
        assertEquals("nlu-toy-iree-cpu-arm64", descriptor["id"]!!.jsonPrimitive.content)
        assertEquals("1.2.3", descriptor["version"]!!.jsonPrimitive.content)
        assertEquals("proprietary", descriptor["license"]!!.jsonPrimitive.content)
        assertTrue(SpecSchema.DESCRIPTOR.violations(descriptor).isEmpty())

        val manifest = result.manifest
        assertTrue(SpecSchema.MANIFEST.violations(manifest).isEmpty())
        assertEquals(CanonicalJson.sha256(descriptor), manifest["descriptor_digest"]!!.jsonPrimitive.content)
        val provenance = manifest["provenance"]!!.jsonObject
        assertEquals(Digests.sha256(f.blueprintFile), provenance["blueprint"]!!.jsonObject["digest"]!!.jsonPrimitive.content)
        assertEquals(Digests.sha256(f.profileFile), provenance["materialization"]!!.jsonObject["profile_digest"]!!.jsonPrimitive.content)
        assertEquals("3.11.0", provenance["materialization"]!!.jsonObject["toolchain"]!!.jsonObject["compiler"]!!.jsonPrimitive.content)
        assertTrue(provenance["inputs"]!!.jsonArray.any { it.jsonObject["name"]!!.jsonPrimitive.content == "input:tool-catalog" })

        // Every artifact on disk matches its manifest entry; licenses trace through derived_from.
        val artifacts = manifest["artifacts"]!!.jsonArray.map { it.jsonObject }
        for (a in artifacts) {
            val file = File(pack, a["path"]!!.jsonPrimitive.content)
            assertEquals(Digests.sha256(file), a["digest"]!!.jsonPrimitive.content, a["path"].toString())
            assertEquals(file.length(), a["size"]!!.jsonPrimitive.content.toLong())
        }
        fun licenseOf(path: String) = artifacts.single { it["path"]!!.jsonPrimitive.content == path }["license"]!!.jsonPrimitive.content
        assertEquals("MIT", licenseOf("artifacts/runtime/libtoy-ctg.so"))
        assertEquals("Toy Terms of Use", licenseOf("artifacts/model/prefill.vmfb"))
        assertEquals("proprietary", licenseOf("artifacts/other/tool-catalog.json"))

        // The signature is the spec's construction: Ed25519 over the ASCII hex digest of the canonical unsigned manifest.
        val sig = manifest["signatures"]!!.jsonArray.single().jsonObject
        val unsigned = JsonObject(manifest.filterKeys { it != "signatures" })
        val message = CanonicalJson.sha256(unsigned).removePrefix("sha256:").toByteArray(Charsets.US_ASCII)
        val verifier = Signature.getInstance("Ed25519").apply { initVerify(publicKey); update(message) }
        assertTrue(verifier.verify(Base64.getDecoder().decode(sig["signature"]!!.jsonPrimitive.content)))
        assertEquals("test-key", sig["keyid"]!!.jsonPrimitive.content)
    }

    @Test
    fun `without a key the manifest is written unsigned and says so`() {
        val result = materialize(Fixture())
        assertFalse(result.signed)
        assertFalse("signatures" in result.manifest)
        assertEquals("MIT AND LicenseRef-Toy-Terms-of-Use", result.manifest["effective_license"]!!.jsonPrimitive.content)
    }

    @Test
    fun `performance is never invented`() {
        val e = assertFailsWith<MaterializationException> { materialize(Fixture(measurements = false)) }
        assertContains(e.message!!, "MEASURED")
        assertContains(e.message!!, "reference_measurements")
    }

    @Test
    fun `inputs are validated against the schema the blueprint declares`() {
        val f = Fixture()
        f.catalog.writeText("""{"functions":[]}""")
        val e = assertFailsWith<MaterializationException> { materialize(f) }
        assertContains(e.message!!, "tool-catalog")
        assertContains(e.message!!, "does not conform")
    }

    @Test
    fun `a selected flavor brings its source, its artifact and its descriptor flavor`() {
        val base = materialize(Fixture(withGermanFlavor = true))
        assertFalse(File(base.packDir, "artifacts/weights/de/model.bin").exists())
        assertFalse("flavors" in base.descriptor)

        val de = materialize(Fixture(withGermanFlavor = true, selectFlavors = listOf("de")))
        assertTrue(File(de.packDir, "artifacts/weights/de/model.bin").isFile)
        val flavor = (de.descriptor["flavors"] as JsonArray).single().jsonObject
        assertEquals("nlu-toy-iree-cpu-arm64-de", flavor["id"]!!.jsonPrimitive.content)
        assertEquals(listOf("de"), de.manifest["provenance"]!!.jsonObject["materialization"]!!.jsonObject["flavors"]!!.jsonArray.map { it.jsonPrimitive.content })
    }

    @Test
    fun `a profile for another blueprint version is refused`() {
        val f = Fixture()
        f.profileFile.writeText(f.profileFile.readText().replace(""""version":"0.1.0"""", """"version":"0.2.0""""))
        val e = assertFailsWith<MaterializationException> { materialize(f) }
        assertContains(e.message!!, "nlu-toy-iree@0.2.0")
    }
}
