package sk.ainet.cartridge.blueprint.core

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import sk.ainet.cartridge.blueprint.model.Blueprint
import sk.ainet.cartridge.blueprint.model.MaterializationException
import sk.ainet.cartridge.blueprint.model.Profile

/**
 * Step 7: template + what only a materialization knows = a schema-valid capability descriptor.
 * `performance` and `quality` come from the profile's measurements (or a `measure` step that wrote them
 * there) — never from the blueprint's `reference_measurements`.
 */
object DescriptorResolver {
    const val SPEC_VERSION = "0.4"

    fun cartridgeId(blueprint: Blueprint, profile: Profile): String {
        val target = blueprint.target(profile.target)
        return profile.cartridge.id ?: target.cartridgeId ?: "${blueprint.id}-${target.id}"
    }

    fun resolve(blueprint: Blueprint, profile: Profile, effectiveLicense: String): JsonObject {
        val target = blueprint.target(profile.target)
        val measurements = profile.measurements
        val performance = measurements?.performance ?: throw MaterializationException(
            "No `performance` for the descriptor. It must be MEASURED on the artifacts this materialization " +
                "produced, on the device `measured_on` names — supply `measurements.performance` in the profile " +
                "(or run the blueprint's `measure` step). A blueprint's reference_measurements are an expectation " +
                "and are never copied into a descriptor.",
        )

        val fields = LinkedHashMap<String, JsonElement>()
        fields["spec_version"] = JsonPrimitive(SPEC_VERSION)
        fields["id"] = JsonPrimitive(cartridgeId(blueprint, profile))
        fields["version"] = JsonPrimitive(profile.cartridge.version)
        fields.putAll(blueprint.descriptorTemplate)
        fields["license"] = JsonPrimitive(effectiveLicense)
        fields["target"] = target.target
        target.requirements?.let { fields["requirements"] = it }
        fields["performance"] = performance
        measurements.quality?.let { fields["quality"] = it }
        profile.descriptorOverrides?.forEach { (k, v) -> fields[k] = v }

        val flavors = blueprint.flavors.filter { it.id in profile.flavors }.map { flavor ->
            val m = measurements.flavors[flavor.id]
            buildJsonObject {
                put("id", JsonPrimitive("${fields.getValue("id").let { (it as JsonPrimitive).content }}-${flavor.id}"))
                flavor.descriptorOverrides?.forEach { (k, v) -> put(k, v) }
                m?.performance?.let { put("performance", it) }
                m?.quality?.let { put("quality", it) }
            }
        }
        if (flavors.isNotEmpty()) fields["flavors"] = JsonArray(flavors)

        val descriptor = JsonObject(fields)
        SpecSchema.DESCRIPTOR.require(descriptor, "The resolved descriptor for ${fields["id"]}")
        return descriptor
    }
}
