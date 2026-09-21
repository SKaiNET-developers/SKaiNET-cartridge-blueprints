package sk.ainet.cartridge.blueprint.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonObject

/** `blueprint.json` — spec: blueprints.adoc, schema cartridge-blueprint.schema.json. */
@Serializable
data class Blueprint(
    @SerialName("spec_version") val specVersion: String,
    val kind: String,
    val id: String,
    val version: String,
    val description: String? = null,
    val license: String,
    @SerialName("descriptor_template") val descriptorTemplate: JsonObject,
    val sources: List<Source>,
    val toolchain: Map<String, Tool>,
    val targets: List<Target>,
    val flavors: List<Flavor> = emptyList(),
    val inputs: List<InputDecl> = emptyList(),
    val steps: List<Step>,
    val outputs: List<Output>,
    val verification: List<JsonObject> = emptyList(),
    @SerialName("reference_measurements") val referenceMeasurements: List<JsonObject> = emptyList(),
) {
    fun target(id: String): Target =
        targets.firstOrNull { it.id == id }
            ?: throw MaterializationException("Blueprint '${this.id}' has no target '$id' (known: ${targets.joinToString { it.id }})")

    fun source(name: String): Source? = sources.firstOrNull { it.name == name }
    fun input(name: String): InputDecl? = inputs.firstOrNull { it.name == name }
}

@Serializable
data class Source(
    val name: String,
    val role: String,
    val uri: String,
    val revision: String,
    val files: List<SourceFile>,
    val license: String,
    @SerialName("license_url") val licenseUrl: String? = null,
    val redistribution: Redistribution,
)

@Serializable
data class SourceFile(val path: String, val digest: String, val size: Long? = null)

@Serializable
enum class Redistribution {
    @SerialName("allowed") ALLOWED,
    @SerialName("restricted") RESTRICTED,
    @SerialName("acceptance-required") ACCEPTANCE_REQUIRED,
}

@Serializable
data class Tool(
    val version: String,
    val coordinates: String? = null,
    val image: String? = null,
    val description: String? = null,
)

@Serializable
data class Target(
    val id: String,
    @SerialName("cartridge_id") val cartridgeId: String? = null,
    val target: JsonObject,
    val requirements: JsonObject? = null,
    val params: JsonObject? = null,
)

@Serializable
data class Flavor(
    val id: String,
    val description: String? = null,
    val sources: List<String>,
    @SerialName("descriptor_overrides") val descriptorOverrides: JsonObject? = null,
)

@Serializable
data class InputDecl(
    val name: String,
    val description: String? = null,
    val role: String,
    val required: Boolean,
    val schema: String? = null,
)

@Serializable
data class Step(
    val id: String,
    val kind: String,
    val description: String? = null,
    val tool: String,
    val uses: List<String> = emptyList(),
    val produces: List<String>,
    @SerialName("per_target") val perTarget: Boolean = false,
    val params: JsonObject? = null,
)

@Serializable
data class Output(
    val from: String,
    val role: String,
    val path: String,
    @SerialName("derived_from") val derivedFrom: List<String>,
    val flavor: String? = null,
)

/** Every failure a materialization can end in; the message is what the person building reads. */
class MaterializationException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)
