package sk.ainet.cartridge.blueprint.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonObject

/** Materialization profile — spec: blueprints.adoc, schema materialization-profile.schema.json. */
@Serializable
data class Profile(
    @SerialName("spec_version") val specVersion: String,
    val kind: String,
    val blueprint: BlueprintRef,
    val target: String,
    val flavors: List<String> = emptyList(),
    val cartridge: CartridgeIdentity,
    @SerialName("license_acceptance") val licenseAcceptance: List<LicenseAcceptance> = emptyList(),
    val mirrors: Map<String, String> = emptyMap(),
    val inputs: Map<String, SuppliedInput> = emptyMap(),
    val measurements: Measurements? = null,
    @SerialName("descriptor_overrides") val descriptorOverrides: JsonObject? = null,
    val signing: Signing? = null,
)

@Serializable
data class BlueprintRef(
    val id: String,
    val version: String,
    val digest: String? = null,
    val source: JsonObject? = null,
)

@Serializable
data class CartridgeIdentity(val id: String? = null, val version: String)

@Serializable
data class LicenseAcceptance(
    val source: String,
    val license: String,
    @SerialName("terms_url") val termsUrl: String? = null,
    @SerialName("accepted_by") val acceptedBy: String,
    @SerialName("accepted_at") val acceptedAt: String,
)

@Serializable
data class SuppliedInput(val path: String, val digest: String? = null, val license: String)

@Serializable
data class Measurements(
    val performance: JsonObject? = null,
    val quality: JsonObject? = null,
    val flavors: Map<String, FlavorMeasurements> = emptyMap(),
)

@Serializable
data class FlavorMeasurements(val performance: JsonObject? = null, val quality: JsonObject? = null)

@Serializable
data class Signing(
    val keyid: String,
    val signer: String,
    @SerialName("key_ref") val keyRef: String? = null,
)
