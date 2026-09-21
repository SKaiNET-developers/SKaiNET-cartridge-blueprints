package sk.ainet.cartridge.blueprint

import sk.ainet.cartridge.blueprint.core.Digests
import java.io.File
import java.nio.file.Files
import java.security.KeyPairGenerator
import java.security.PublicKey
import java.util.Base64

/**
 * A complete synthetic blueprint on disk: a "publisher" directory served over `file:`, a blueprint that pins
 * it, a toy input schema, fake build products and a profile. Nothing in it resembles a real model.
 */
class Fixture(
    val root: File = Files.createTempDirectory("blueprint-fixture").toFile(),
    redistribution: String = "acceptance-required",
    accept: Boolean = true,
    withGermanFlavor: Boolean = false,
    selectFlavors: List<String> = emptyList(),
    measurements: Boolean = true,
    inputLicense: String = "MIT",
) {
    val publisher = File(root, "publisher").apply { mkdirs() }
    val weights = File(publisher, "model.bin").apply { writeBytes(ByteArray(2048) { (it % 251).toByte() }) }
    val tokenizer = File(publisher, "tokenizer.json").apply { writeText("""{"vocab":["a","b"]}""") }
    val weightsDe = File(publisher, "de/model.bin").apply { parentFile.mkdirs(); writeBytes(ByteArray(1024) { (it % 13).toByte() }) }

    val blueprintDir = File(root, "blueprint").apply { mkdirs() }
    val products = File(root, "products").apply { mkdirs() }
    val runtimeSo = File(products, "libtoy-ctg.so").apply { writeBytes(byteArrayOf(0x7f, 0x45, 0x4c, 0x46, 1, 2, 3)) }
    val graphsDir = File(products, "graphs").apply {
        mkdirs(); File(this, "prefill.vmfb").writeBytes(byteArrayOf(1, 2, 3)); File(this, "decode.vmfb").writeBytes(byteArrayOf(4, 5))
    }

    val blueprintFile = File(blueprintDir, "blueprint.json")
    val profileFile = File(root, "profiles/device.json").apply { parentFile.mkdirs() }
    val catalog = File(root, "profiles/catalog.json").apply { writeText("""{"functions":[{"name":"lights_on"}]}""") }

    init {
        File(blueprintDir, "schema").mkdirs()
        File(blueprintDir, "schema/tool-catalog.schema.json").writeText(
            """{"${'$'}schema":"https://json-schema.org/draft/2020-12/schema","type":"object","required":["functions"],
               "properties":{"functions":{"type":"array","minItems":1,"items":{"type":"object","required":["name"]}}}}""",
        )
        val flavorSource = if (withGermanFlavor) """,
            {"name":"weights-de","role":"weights","uri":"${publisher.toURI()}","revision":"r1",
             "files":[{"path":"de/model.bin","digest":"${Digests.sha256(weightsDe)}"}],
             "license":"MIT","redistribution":"allowed"}""" else ""
        val flavors = if (withGermanFlavor) """"flavors":[{"id":"de","sources":["weights-de"],"descriptor_overrides":{"attributes":{"languages":["de"]}}}],""" else ""
        val flavorOutput = if (withGermanFlavor) """,
            {"from":"weights-de-files","role":"weights","path":"artifacts/weights/de/model.bin","derived_from":["weights-de"],"flavor":"de"}""" else ""
        val flavorFetch = if (withGermanFlavor) """,
            {"id":"fetch-de","kind":"fetch","tool":"plugin","uses":["weights-de"],"produces":["weights-de-files"]}""" else ""
        blueprintFile.writeText(
            """
            {
              "spec_version":"0.4","kind":"blueprint","id":"nlu-toy-iree","version":"0.1.0","license":"MIT",
              "descriptor_template":{
                "execution_mode":"native","family":"toy","modality":"text","task":"nlu",
                "io":{"mode":"batch","input":{"dtype":"utf8-text"},"output":{"dtype":"utf8-json"}},
                "attributes":{"languages":["en"]}
              },
              "sources":[
                {"name":"weights","role":"weights","uri":"${publisher.toURI()}","revision":"r1",
                 "files":[{"path":"model.bin","digest":"${Digests.sha256(weights)}","size":2048}],
                 "license":"Toy Terms of Use","license_url":"https://example.org/terms","redistribution":"$redistribution"},
                {"name":"tokenizer","role":"tokenizer","uri":"${publisher.toURI()}","revision":"r1",
                 "files":[{"path":"tokenizer.json","digest":"${Digests.sha256(tokenizer)}"}],
                 "license":"MIT","redistribution":"allowed"}$flavorSource
              ],
              "toolchain":{"plugin":{"version":"0.1.0"},"compiler":{"version":"3.11.0"}},
              "targets":[{"id":"cpu-arm64","target":{"hardware":"IREE-arm64-v8a","abi":"arm64-v8a"}}],
              $flavors
              "inputs":[{"name":"tool-catalog","role":"other","required":true,"schema":"schema/tool-catalog.schema.json"}],
              "steps":[
                {"id":"fetch","kind":"fetch","tool":"plugin","uses":["weights","tokenizer"],"produces":["weights-files","tokenizer-files"]}$flavorFetch,
                {"id":"compile","kind":"compile","tool":"compiler","uses":["weights-files"],"produces":["graphs"],"per_target":true},
                {"id":"runtime","kind":"build-runtime","tool":"compiler","produces":["runtime-so"],"per_target":true}
              ],
              "outputs":[
                {"from":"runtime-so","role":"runtime","path":"artifacts/runtime/libtoy-ctg.so","derived_from":[]},
                {"from":"graphs","role":"model","path":"artifacts/model/","derived_from":["weights"]},
                {"from":"weights-files","role":"weights","path":"artifacts/weights/model.bin","derived_from":["weights"]},
                {"from":"tokenizer-files","role":"tokenizer","path":"artifacts/tokenizer/tokenizer.json","derived_from":["tokenizer"]},
                {"from":"tool-catalog","role":"other","path":"artifacts/other/tool-catalog.json","derived_from":["tool-catalog"]}$flavorOutput
              ]
            }
            """.trimIndent(),
        )
        val acceptance = if (accept) """"license_acceptance":[{"source":"weights","license":"Toy Terms of Use","accepted_by":"Test Org","accepted_at":"2026-09-21"}],""" else ""
        val measured = if (measurements) """"measurements":{"performance":{"latency_p50_ms":12.5,"measured_on":"synthetic test device"}},""" else ""
        val selected = if (selectFlavors.isNotEmpty()) """"flavors":[${selectFlavors.joinToString(",") { "\"$it\"" }}],""" else ""
        profileFile.writeText(
            """
            {
              "spec_version":"0.4","kind":"materialization-profile",
              "blueprint":{"id":"nlu-toy-iree","version":"0.1.0"},
              "target":"cpu-arm64", $selected
              "cartridge":{"version":"1.2.3"},
              $acceptance
              $measured
              "inputs":{"tool-catalog":{"path":"catalog.json","license":"$inputLicense"}},
              "signing":{"keyid":"test-key","signer":"Test Org build","key_ref":"TEST_SIGNING_KEY"}
            }
            """.trimIndent(),
        )
    }

    val productMap: Map<String, File> get() = mapOf("runtime-so" to runtimeSo, "graphs" to graphsDir)

    companion object {
        /** A fresh Ed25519 key pair: PKCS#8 PEM private key, and the public key. */
        fun keyPair(): Pair<String, PublicKey> {
            val kp = KeyPairGenerator.getInstance("Ed25519").generateKeyPair()
            val b64 = Base64.getMimeEncoder(64, "\n".toByteArray()).encodeToString(kp.private.encoded)
            return "-----BEGIN PRIVATE KEY-----\n$b64\n-----END PRIVATE KEY-----\n" to kp.public
        }

        fun publicPem(key: PublicKey): String =
            "-----BEGIN PUBLIC KEY-----\n" + Base64.getMimeEncoder(64, "\n".toByteArray()).encodeToString(key.encoded) + "\n-----END PUBLIC KEY-----\n"
    }
}
