package sk.ainet.cartridge.blueprint.core

import io.github.optimumcode.json.schema.ErrorCollector
import io.github.optimumcode.json.schema.JsonSchema
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.decodeFromJsonElement
import sk.ainet.cartridge.blueprint.model.Blueprint
import sk.ainet.cartridge.blueprint.model.MaterializationException
import sk.ainet.cartridge.blueprint.model.Profile
import java.io.File

/** The four spec schemas, bundled at the spec revision recorded in `schema/SOURCE.txt`. */
enum class SpecSchema(private val resource: String) {
    BLUEPRINT("cartridge-blueprint.schema.json"),
    PROFILE("materialization-profile.schema.json"),
    DESCRIPTOR("cartridge-descriptor.schema.json"),
    MANIFEST("cartridge-manifest.schema.json"),
    ;

    private val schema: JsonSchema by lazy {
        val text = SpecSchema::class.java.getResourceAsStream("/sk/ainet/cartridge/blueprint/schema/$resource")
            ?.reader()?.use { it.readText() }
            ?: error("Bundled schema $resource is missing from the plugin jar")
        JsonSchema.fromDefinition(text)
    }

    /** Returns one line per violation, `<json pointer>: <message>`; empty when valid. */
    fun violations(instance: JsonElement): List<String> {
        val errors = mutableListOf<String>()
        schema.validate(instance, ErrorCollector { e -> errors += "${e.objectPath}: ${e.message}" })
        return errors
    }

    fun require(instance: JsonElement, what: String) {
        val v = violations(instance)
        if (v.isNotEmpty()) {
            throw MaterializationException("$what does not conform to $resource:\n" + v.joinToString("\n") { "  - $it" })
        }
    }
}

/** A parsed, schema-valid document together with the digest of the file it came from. */
data class Loaded<T>(val value: T, val json: JsonElement, val file: File, val fileDigest: String)

object Documents {
    val json = Json { ignoreUnknownKeys = false; prettyPrint = true; prettyPrintIndent = "  " }

    fun blueprint(file: File): Loaded<Blueprint> = load(file, SpecSchema.BLUEPRINT, "Blueprint ${file.name}")

    fun profile(file: File): Loaded<Profile> = load(file, SpecSchema.PROFILE, "Profile ${file.name}")

    private inline fun <reified T> load(file: File, schema: SpecSchema, what: String): Loaded<T> {
        if (!file.isFile) throw MaterializationException("$what not found: ${file.absolutePath}")
        val element = try {
            json.parseToJsonElement(file.readText())
        } catch (e: Exception) {
            throw MaterializationException("$what is not valid JSON: ${e.message}", e)
        }
        schema.require(element, what)
        return Loaded(json.decodeFromJsonElement<T>(element), element, file, Digests.sha256(file))
    }

    /** The profile must name the blueprint in hand — by id and exact version, and by digest when it gives one. */
    fun requireMatch(blueprint: Loaded<Blueprint>, profile: Loaded<Profile>) {
        val ref = profile.value.blueprint
        val bp = blueprint.value
        if (ref.id != bp.id || ref.version != bp.version) {
            throw MaterializationException(
                "Profile ${profile.file.name} is for blueprint ${ref.id}@${ref.version}, " +
                    "but the blueprint in hand is ${bp.id}@${bp.version}.",
            )
        }
        if (ref.digest != null && ref.digest != blueprint.fileDigest) {
            throw MaterializationException(
                "Blueprint digest mismatch: profile pins ${ref.digest}, ${blueprint.file.name} is ${blueprint.fileDigest}. " +
                    "Same version, different recipe — refusing to materialize.",
            )
        }
    }
}
