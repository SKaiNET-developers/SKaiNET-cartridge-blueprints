package sk.ainet.cartridge.blueprint.core

import io.github.optimumcode.json.schema.ErrorCollector
import io.github.optimumcode.json.schema.JsonSchema
import sk.ainet.cartridge.blueprint.model.Blueprint
import sk.ainet.cartridge.blueprint.model.MaterializationException
import sk.ainet.cartridge.blueprint.model.Profile
import java.io.File

/** A declared input as the profile supplied it: the local file, its license, and the digest it has now. */
data class ResolvedInput(val file: File, val license: String, val digest: String)

/**
 * Step 4 of materialization (blueprints.adoc): the profile's `inputs` against the blueprint's declarations.
 *
 * One implementation for the two readers of an input: the [Materializer], which places it and records it in the
 * manifest, and a module's own step tasks (a `train` step reading a dataset, ADR-018), which get it through
 * `blueprint.inputFile(name)`. Both see the same file, checked the same way.
 */
object InputResolver {

    /**
     * Every supplied input by its declared name. Fails when the profile supplies an undeclared input, omits a
     * required one, points at a missing file, pins a digest the file does not have, or supplies a file that does
     * not conform to the schema the blueprint declares for it. Paths are relative to the profile file.
     */
    fun resolve(blueprint: Loaded<Blueprint>, profile: Loaded<Profile>): Map<String, ResolvedInput> {
        val bp = blueprint.value
        val pf = profile.value
        val undeclared = pf.inputs.keys - bp.inputs.map { it.name }.toSet()
        if (undeclared.isNotEmpty()) {
            throw MaterializationException("Profile supplies inputs the blueprint does not declare: $undeclared")
        }
        val resolved = LinkedHashMap<String, ResolvedInput>()
        for (decl in bp.inputs) {
            val supplied = pf.inputs[decl.name]
            if (supplied == null) {
                if (decl.required) {
                    throw MaterializationException(
                        "Required input '${decl.name}' is missing from the profile" +
                            (decl.description?.let { " — $it" } ?: "") + ". Supply it under `inputs.${decl.name}`.",
                    )
                }
                continue
            }
            val file = profile.file.parentFile.resolve(supplied.path).canonicalFile
            if (!file.isFile) throw MaterializationException("Input '${decl.name}': no such file ${file.path}")
            val digest = Digests.sha256(file)
            if (supplied.digest != null && supplied.digest != digest) {
                throw MaterializationException("Input '${decl.name}': profile pins ${supplied.digest}, file is $digest")
            }
            decl.schema?.let { validate(decl.name, file, blueprint.file.parentFile.resolve(it)) }
            resolved[decl.name] = ResolvedInput(file, supplied.license, digest)
        }
        return resolved
    }

    private fun validate(name: String, file: File, schemaFile: File) {
        if (!schemaFile.isFile) throw MaterializationException("Input '$name': declared schema not found: ${schemaFile.path}")
        val errors = mutableListOf<String>()
        JsonSchema.fromDefinition(schemaFile.readText())
            .validate(Documents.json.parseToJsonElement(file.readText()), ErrorCollector { e -> errors += "${e.objectPath}: ${e.message}" })
        if (errors.isNotEmpty()) {
            throw MaterializationException("Input '$name' (${file.name}) does not conform to ${schemaFile.name}:\n" + errors.joinToString("\n") { "  - $it" })
        }
    }
}
