package sk.ainet.cartridge.nlu.functiongemma

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.add
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import kotlinx.serialization.json.putJsonObject
import sk.ainet.apps.kllama.chat.ToolDefinition
import java.io.File

/** One argument of a function: JSON-Schema type, description, optional closed value set. */
@Serializable
public data class ToolParameter(val type: String, val description: String, val enum: List<String>? = null)

/** One catalog entry: a function the model may call and its argument schema. */
@Serializable
public data class ToolFunction(
    val name: String,
    val description: String,
    val parameters: Map<String, ToolParameter> = emptyMap(),
    val required: List<String> = emptyList(),
) {
    public fun toDefinition(): ToolDefinition = ToolDefinition(
        name = name, description = description,
        parameters = buildJsonObject {
            put("type", "object")
            putJsonObject("properties") {
                for ((k, v) in parameters) putJsonObject(k) {
                    put("type", v.type); put("description", v.description)
                    v.enum?.let { e -> putJsonArray("enum") { e.forEach { add(it) } } }
                }
            }
            if (required.isNotEmpty()) putJsonArray("required") { required.forEach { add(it) } }
        },
    )
}

/**
 * The functions a cartridge instance may call. A *materialization input*: it arrives as
 * `artifacts/other/tool-catalog.json` in the pack_dir (schema: `schema/tool-catalog.schema.json`) and is
 * rendered into the prompt prefix verbatim — names and descriptions are model input. This module contains no
 * catalog of its own: which functions exist is the host's knowledge, never the cartridge's.
 */
@Serializable
public data class ToolCatalog(
    val id: String,
    val languages: List<String> = emptyList(),
    val functions: List<ToolFunction>,
) {
    init {
        require(functions.isNotEmpty()) { "tool catalog '$id' has no functions" }
        val duplicates = functions.groupingBy { it.name }.eachCount().filterValues { it > 1 }.keys
        require(duplicates.isEmpty()) { "tool catalog '$id' defines functions more than once: $duplicates" }
    }

    val names: Set<String> get() = functions.mapTo(LinkedHashSet()) { it.name }

    public fun definitions(): List<ToolDefinition> = functions.map { it.toDefinition() }

    /**
     * Snap a near-miss name the model emitted (`toggle_lampp`, `Toggle__Lamp`) to the catalog: exact match, then a
     * case- and punctuation-insensitive match, then edit distance <= 2 on the normalized form. `null` if nothing is close.
     */
    public fun snapName(raw: String?): String? {
        if (raw == null) return null
        if (raw in names) return raw
        val norm = normalize(raw)
        val byNorm = functions.associate { normalize(it.name) to it.name }
        byNorm[norm]?.let { return it }
        return byNorm.keys.map { it to levenshtein(it, norm) }.filter { it.second <= 2 }.minByOrNull { it.second }?.let { byNorm[it.first] }
    }

    public companion object {
        private val json = Json { ignoreUnknownKeys = false }

        public fun parse(text: String): ToolCatalog = json.decodeFromString(serializer(), text)

        public fun load(file: File): ToolCatalog = parse(file.readText())

        private fun normalize(s: String): String = s.lowercase().replace(Regex("[^a-z0-9_]"), "").replace(Regex("_+"), "_")

        private fun levenshtein(a: String, b: String): Int {
            val d = Array(a.length + 1) { IntArray(b.length + 1) }
            for (i in 0..a.length) d[i][0] = i
            for (j in 0..b.length) d[0][j] = j
            for (i in 1..a.length) for (j in 1..b.length) {
                d[i][j] = minOf(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + if (a[i - 1] == b[j - 1]) 0 else 1)
            }
            return d[a.length][b.length]
        }
    }
}
