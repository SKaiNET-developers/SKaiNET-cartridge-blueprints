package sk.ainet.cartridge.blueprint.core

import sk.ainet.cartridge.blueprint.model.Blueprint
import sk.ainet.cartridge.blueprint.model.MaterializationException
import sk.ainet.cartridge.blueprint.model.Profile
import sk.ainet.cartridge.blueprint.model.Redistribution
import sk.ainet.cartridge.blueprint.model.Source

/** Which of a blueprint's sources a given profile actually needs: the base ones plus the selected flavors'. */
object Selection {
    fun sources(blueprint: Blueprint, profile: Profile): List<Source> {
        val unknown = profile.flavors - blueprint.flavors.map { it.id }.toSet()
        if (unknown.isNotEmpty()) {
            throw MaterializationException("Profile selects flavors the blueprint does not define: $unknown")
        }
        val flavorOnly = blueprint.flavors.flatMap { it.sources }.toSet()
        val selected = blueprint.flavors.filter { it.id in profile.flavors }.flatMap { it.sources }.toSet()
        // A source listed by some flavor belongs to flavors only; everything else is the base.
        return blueprint.sources.filter { it.name !in flavorOnly || it.name in selected }
    }
}

/**
 * Step 2 of materialization (blueprints.adoc): licenses are checked BEFORE anything is fetched.
 * For an `acceptance-required` source without a matching acceptance record the materializer MUST stop,
 * and MUST NOT offer to accept on anyone's behalf — so this only ever reports, it never prompts.
 */
object LicenseGate {
    fun check(blueprint: Blueprint, profile: Profile) {
        val missing = Selection.sources(blueprint, profile)
            .filter { it.redistribution == Redistribution.ACCEPTANCE_REQUIRED }
            .filter { src -> profile.licenseAcceptance.none { it.source == src.name && it.license == src.license } }
        if (missing.isEmpty()) return
        val lines = missing.joinToString("\n") { src ->
            "  - source '${src.name}' (${src.uri} @ ${src.revision}): ${src.license}" +
                (src.licenseUrl?.let { " — $it" } ?: "")
        }
        throw MaterializationException(
            "License acceptance required before anything is downloaded:\n$lines\n" +
                "Read the terms. If you (or your organization) accept them, record that in the profile under " +
                "`license_acceptance` — one entry per source, with `source`, `license` exactly as above, " +
                "`accepted_by` and `accepted_at`. This tool will not accept a license for you.",
        )
    }

    /** Sources whose conditions the materializing party has to check for itself — reported, never blocking. */
    fun restricted(blueprint: Blueprint, profile: Profile): List<Source> =
        Selection.sources(blueprint, profile).filter { it.redistribution == Redistribution.RESTRICTED }
}

/**
 * Step 8: the effective license is computed from what the outputs are actually derived from (ADR-004).
 * Deliberately simple and conservative: it states the full set of obligations, it does not try to decide
 * compatibility — that judgment stays with the materializing party.
 */
object EffectiveLicense {
    fun compute(blueprintLicense: String, upstream: Collection<String>): String {
        val all = (listOf(blueprintLicense) + upstream).map { it.trim() }.filter { it.isNotEmpty() }.distinct()
        return when {
            all.any { it.equals("NOASSERTION", ignoreCase = true) } -> "NOASSERTION"
            all.any { it.equals("proprietary", ignoreCase = true) } -> "proprietary"
            all.size == 1 -> all.single()
            else -> all.joinToString(" AND ") { if (SPDX_ID.matches(it)) it else "LicenseRef-" + it.replace(NOT_IDSTRING, "-").trim('-') }
        }
    }

    private val SPDX_ID = Regex("[A-Za-z0-9.+-]+")
    private val NOT_IDSTRING = Regex("[^A-Za-z0-9.-]+")
}
