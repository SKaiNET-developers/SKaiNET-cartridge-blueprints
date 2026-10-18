package sk.ainet.cartridge.embedding.leaf

import java.io.File
import kotlin.test.Test
import kotlinx.coroutines.runBlocking

/**
 * Regenerates `samples/golden-vectors.json` — the space-defining bytes for
 * `mdbr-leaf-ir-v1`. JVM-only and opt-in:
 *
 *   ./gradlew :blueprints:embedding-leaf-skainet:jvmTest -PregenerateGoldens=true --tests "*GoldenRegenerator*"
 *
 * Changing a value in that file is declaring a new space fingerprint.
 */
class GoldenRegenerator {

    @Test
    fun regenerate() {
        if (System.getenv("REGENERATE_GOLDENS").isNullOrBlank()) {
            println("GoldenRegenerator: skipped (run with -PregenerateGoldens=true to rewrite the goldens)")
            return
        }
        runBlocking {
            LeafEmbeddingCartridge.open().use { engine ->
                val vectors = GoldenCases.TEXTS.map { engine.embed(it) }
                val json = buildString {
                    appendLine("{")
                    appendLine("  \"space_fingerprint\": \"${LeafEmbeddingCartridge.SPACE_FINGERPRINT}\",")
                    appendLine("  \"model\": \"${LeafEmbeddingCartridge.MODEL_REPO}@${LeafEmbeddingCartridge.MODEL_REVISION}\",")
                    appendLine("  \"note\": \"These bytes define the space (first ${GoldenCases.PREFIX_DIMS} of ${engine.dimensions} dims per case, JVM reference). Any drift is a new space fingerprint and a full corpus re-embed.\",")
                    appendLine("  \"cases\": [")
                    GoldenCases.TEXTS.forEachIndexed { i, text ->
                        val prefix = vectors[i].take(GoldenCases.PREFIX_DIMS).joinToString(", ")
                        append("    {\"text\": ${jsonString(text)}, \"dimensions\": ${engine.dimensions}, \"vector_prefix\": [$prefix]}")
                        appendLine(if (i < GoldenCases.TEXTS.lastIndex) "," else "")
                    }
                    appendLine("  ],")
                    appendLine("  \"cosine_query_doc\": ${cosine(vectors[0], vectors[1])},")
                    appendLine("  \"cosine_query_unrelated\": ${cosine(vectors[0], vectors[2])}")
                    appendLine("}")
                }
                val out = File("samples/golden-vectors.json")
                out.parentFile.mkdirs()
                out.writeText(json)
                println("GoldenRegenerator: wrote ${out.absolutePath}")
                println("--- paste into GoldenCases ---")
                vectors.forEachIndexed { i, v ->
                    println("case $i prefix: floatArrayOf(${v.take(GoldenCases.PREFIX_DIMS).joinToString(", ") { "${it}f" }})")
                }
                println("cosine(query, doc) = ${cosine(vectors[0], vectors[1])}")
                println("cosine(query, unrelated) = ${cosine(vectors[0], vectors[2])}")
            }
        }
    }

    private fun jsonString(s: String) = "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\""

    private fun cosine(a: FloatArray, b: FloatArray): Double {
        var dot = 0.0
        for (i in a.indices) dot += a[i].toDouble() * b[i]
        return dot
    }
}
