package sk.ainet.cartridge.embedding.leaf

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlinx.coroutines.runBlocking

/**
 * The space-defining goldens for `mdbr-leaf-ir-v1`, asserted on every compiled target: passing
 * on JVM and macosArm64 is the proof that one `space_fingerprint` means one space across the
 * jdk.incubator.vector and Accelerate backends. Values were produced by `GoldenRegenerator` on
 * JVM (`samples/golden-vectors.json` carries the same bytes); the tolerance below is the
 * cross-backend float budget the blueprint's verification entry declares.
 *
 * Needs the pinned snapshot on disk (blueprint fetch step, ~92 MB — never in git). Without it
 * the tests skip loudly rather than fail, so a fresh clone still builds.
 */
class LeafGoldenTest {

    private val goldenPrefixes = listOf(
        floatArrayOf(0.073538415f, -0.0078527415f, 0.10580704f, -0.017541848f, 0.041230984f, 0.020081475f, 0.030237941f, 0.0560397f),
        floatArrayOf(0.099036366f, 0.009893688f, 0.10623513f, -0.077831745f, 0.047325425f, -0.0015890392f, 0.07922801f, 0.056192916f),
        floatArrayOf(0.011190697f, 0.017022572f, 0.03868603f, 0.009751211f, 0.09829194f, -0.0025691048f, 0.045268014f, 0.03995983f),
    )
    private val goldenCosineQueryDoc = 0.80888751
    private val goldenCosineQueryUnrelated = 0.67377588

    /** Per-dimension budget between backends; drift beyond it is a new space. */
    private val tolerance = 2e-3f

    private fun <T> withEngine(block: (LeafEmbeddingEngine) -> T): T? {
        val dir = LeafEmbeddingCartridge.defaultModelDir()
        if (dir == null || readTextFile("$dir/config.json") == null) {
            println(
                "SKIPPED: mdbr-leaf-ir snapshot not found (looked in ${dir ?: "<no home dir>"}). " +
                    "Run the blueprint's fetch step or set LEAF_MODEL_DIR.",
            )
            return null
        }
        return runBlocking { LeafEmbeddingCartridge.open(dir).use(block) }
    }

    @Test
    fun goldenVectorsDefineTheSpace() {
        withEngine { engine ->
            GoldenCases.TEXTS.forEachIndexed { caseIndex, text ->
                val vector = engine.embed(text)
                assertEquals(LeafEmbeddingCartridge.descriptor.dimensions, vector.size)
                val golden = goldenPrefixes[caseIndex]
                for (i in golden.indices) {
                    assertTrue(
                        abs(golden[i] - vector[i]) < tolerance,
                        "case $caseIndex dimension $i drifted: golden ${golden[i]}, got ${vector[i]}",
                    )
                }
            }
        }
    }

    @Test
    fun crossBackendSimilaritiesHold() {
        withEngine { engine ->
            val vectors = GoldenCases.TEXTS.map { engine.embed(it) }
            val queryDoc = cosine(vectors[0], vectors[1])
            val queryUnrelated = cosine(vectors[0], vectors[2])
            assertTrue(abs(queryDoc - goldenCosineQueryDoc) < 1e-3, "cosine(query, doc) drifted to $queryDoc")
            assertTrue(
                abs(queryUnrelated - goldenCosineQueryUnrelated) < 1e-3,
                "cosine(query, unrelated) drifted to $queryUnrelated",
            )
            assertTrue(queryDoc > queryUnrelated + 0.05, "retrieval ordering collapsed")
        }
    }

    @Test
    fun engineMatchesItsDescriptor() {
        withEngine { engine ->
            val descriptor = LeafEmbeddingCartridge.descriptor
            assertEquals(descriptor.dimensions, engine.dimensions)
            val vector = engine.embed("local retrieval augmented generation on device")
            assertEquals(descriptor.dimensions, vector.size)
            var norm = 0.0
            for (value in vector) norm += value.toDouble() * value
            assertTrue(abs(norm - 1.0) < 1e-3, "descriptor says normalized, norm was $norm")
        }
    }

    @Test
    fun spiAndProfileAgree() {
        withEngine { engine ->
            // One engine, two contracts: llm-api's call() must return the same bytes as
            // embedding/v1's embed(), and the symmetric model's query/document roles too.
            val text = GoldenCases.TEXTS[0]
            val viaEngine = engine.embed(text)
            val viaSpi = engine.call(sk.ainet.llm.api.EmbeddingRequest(text)).embeddings.single().vector
            val viaQueryRole = engine.embedQuery(text)
            for (i in viaEngine.indices) {
                assertTrue(abs(viaEngine[i] - viaSpi[i]) < 1e-6f)
                assertTrue(abs(viaEngine[i] - viaQueryRole[i]) < 1e-6f)
            }
        }
    }

    private fun cosine(a: FloatArray, b: FloatArray): Double {
        var dot = 0.0
        for (i in a.indices) dot += a[i].toDouble() * b[i]
        return dot
    }
}
