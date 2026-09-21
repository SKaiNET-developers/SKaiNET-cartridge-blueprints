package sk.ainet.cartridge.blueprint.core

import io.github.optimumcode.json.schema.ErrorCollector
import io.github.optimumcode.json.schema.JsonSchema
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import sk.ainet.cartridge.blueprint.model.Blueprint
import sk.ainet.cartridge.blueprint.model.MaterializationException
import sk.ainet.cartridge.blueprint.model.Output
import sk.ainet.cartridge.blueprint.model.Profile
import java.io.File
import java.time.Instant
import java.time.temporal.ChronoUnit

/**
 * Steps 4 and 7–10 of materialization (blueprints.adoc): validate inputs, resolve the descriptor, compute
 * the effective license, pack the `pack_dir` (ADR-012) with a manifest that names the blueprint, and sign it.
 *
 * Everything between fetching and packing — convert, quantize, export, compile, build-runtime — is model- and
 * backend-specific and runs as ordinary build tasks; this class only sees their results as named [products].
 */
class Materializer(
    private val blueprint: Loaded<Blueprint>,
    private val profile: Loaded<Profile>,
    /** source name → (file path in the source → verified local file), as returned by [SourceFetcher]. */
    private val fetched: Map<String, Map<String, File>>,
    /** step product name → the file or directory a build task produced. */
    private val products: Map<String, File>,
    private val materializerId: String,
    private val toolchainUsed: Map<String, String> = blueprint.value.toolchain.mapValues { it.value.version },
    private val now: () -> Instant = Instant::now,
) {
    private val bp get() = blueprint.value
    private val pf get() = profile.value

    data class Result(val packDir: File, val descriptor: JsonObject, val manifest: JsonObject, val signed: Boolean)

    private data class Upstream(val label: String, val license: String)

    fun materialize(packDir: File, privateKeyPem: String?): Result {
        Documents.requireMatch(blueprint, profile)
        LicenseGate.check(bp, pf)
        val inputs = resolveInputs()

        val outputs = bp.outputs.filter { it.flavor == null || it.flavor in pf.flavors }
        val upstreamOf = outputs.associateWith { upstream(it, inputs) }
        val effective = EffectiveLicense.compute(bp.license, upstreamOf.values.flatten().map { it.license })
        val descriptor = DescriptorResolver.resolve(bp, pf, effective)

        if (packDir.exists()) packDir.deleteRecursively()
        packDir.mkdirs()

        val artifacts = mutableListOf<JsonObject>()
        val descriptorFile = File(packDir, "descriptor.json").apply { writeText(Documents.json.encodeToString(JsonObject.serializer(), descriptor) + "\n") }
        artifacts += artifact("descriptor", "descriptor.json", descriptorFile, bp.license, emptyList())

        for (output in outputs) {
            val up = upstreamOf.getValue(output)
            val license = if (up.isEmpty()) bp.license else EffectiveLicense.compute(up.first().license, up.drop(1).map { it.license })
            for ((relPath, src) in place(output, inputs)) {
                val dest = File(packDir, relPath).canonicalFile
                if (!dest.path.startsWith(packDir.canonicalPath + File.separator)) {
                    throw MaterializationException("Output path '$relPath' escapes the pack_dir")
                }
                dest.parentFile.mkdirs()
                src.copyTo(dest, overwrite = true)
                artifacts += artifact(output.role, relPath, dest, license, up)
            }
        }
        if (artifacts.none { it["role"] == JsonPrimitive("runtime") }) {
            throw MaterializationException(
                "The pack_dir has no artifact of role `runtime`. A cartridge bundles the native runtime that " +
                    "implements the Runtime ABI (ADR-012); the blueprint's outputs must map one.",
            )
        }

        val unsigned = manifest(descriptor, effective, artifacts, inputs)
        val signing = pf.signing
        val manifest = when {
            signing == null || privateKeyPem == null -> unsigned
            else -> ManifestSigner.sign(unsigned, signing, privateKeyPem, now())
        }
        val signed = "signatures" in manifest
        // The schema requires `signatures`; an unsigned manifest is still checked for everything else.
        val checkable = if (signed) manifest else JsonObject(manifest + ("signatures" to JsonArray(listOf(PLACEHOLDER_SIGNATURE))))
        SpecSchema.MANIFEST.require(checkable, "The manifest for ${descriptor["id"]}")
        File(packDir, "manifest.json").writeText(Documents.json.encodeToString(JsonObject.serializer(), manifest) + "\n")
        return Result(packDir, descriptor, manifest, signed)
    }

    // --- inputs ------------------------------------------------------------------------------------

    private data class Input(val file: File, val license: String, val digest: String)

    private fun resolveInputs(): Map<String, Input> {
        val undeclared = pf.inputs.keys - bp.inputs.map { it.name }.toSet()
        if (undeclared.isNotEmpty()) {
            throw MaterializationException("Profile supplies inputs the blueprint does not declare: $undeclared")
        }
        val resolved = LinkedHashMap<String, Input>()
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
            decl.schema?.let { validateInput(decl.name, file, blueprint.file.parentFile.resolve(it)) }
            resolved[decl.name] = Input(file, supplied.license, digest)
        }
        return resolved
    }

    private fun validateInput(name: String, file: File, schemaFile: File) {
        if (!schemaFile.isFile) throw MaterializationException("Input '$name': declared schema not found: ${schemaFile.path}")
        val errors = mutableListOf<String>()
        JsonSchema.fromDefinition(schemaFile.readText())
            .validate(Documents.json.parseToJsonElement(file.readText()), ErrorCollector { e -> errors += "${e.objectPath}: ${e.message}" })
        if (errors.isNotEmpty()) {
            throw MaterializationException("Input '$name' (${file.name}) does not conform to ${schemaFile.name}:\n" + errors.joinToString("\n") { "  - $it" })
        }
    }

    // --- outputs -----------------------------------------------------------------------------------

    private fun upstream(output: Output, inputs: Map<String, Input>): List<Upstream> = output.derivedFrom.map { name ->
        bp.source(name)?.let { Upstream("${it.uri}@${it.revision}", it.license) }
            ?: inputs[name]?.let { Upstream("input:$name", it.license) }
            ?: throw MaterializationException(
                "Output '${output.path}' is derived_from '$name', which is neither a source of this blueprint " +
                    "nor an input supplied by the profile.",
            )
    }

    /** What `from` names: a step product, else a supplied input, else (for a `fetch` product) the fetched source. */
    private fun locate(from: String, inputs: Map<String, Input>): List<File> {
        products[from]?.let { return listOf(it) }
        inputs[from]?.let { return listOf(it.file) }
        for (step in bp.steps.filter { it.kind == "fetch" && it.uses.size == it.produces.size }) {
            val i = step.produces.indexOf(from)
            if (i >= 0) return fetched[step.uses[i]]?.values?.toList()
                ?: throw MaterializationException("Product '$from' is the fetched source '${step.uses[i]}', which was not fetched")
        }
        throw MaterializationException(
            "Output maps product '$from', but no build task registered it and it is neither an input nor a fetch product. " +
                "Register it in the build script: blueprint { product(\"$from\", <file or directory>) }",
        )
    }

    private fun place(output: Output, inputs: Map<String, Input>): List<Pair<String, File>> {
        val files = locate(output.from, inputs).flatMap { f ->
            when {
                f.isDirectory -> f.walkTopDown().filter { it.isFile }.sortedBy { it.path }.map { it.relativeTo(f).invariantSeparatorsPath to it }.toList()
                f.isFile -> listOf(f.name to f)
                else -> throw MaterializationException("Product '${output.from}' does not exist: ${f.path}")
            }
        }
        if (files.isEmpty()) throw MaterializationException("Product '${output.from}' is empty")
        return if (output.path.endsWith("/")) {
            files.map { (rel, f) -> output.path + rel to f }
        } else {
            if (files.size != 1) {
                throw MaterializationException(
                    "Output path '${output.path}' names one file, but product '${output.from}' has ${files.size}. " +
                        "End the path with '/' to place a directory.",
                )
            }
            listOf(output.path to files.single().second)
        }
    }

    private fun artifact(role: String, path: String, file: File, license: String, upstream: List<Upstream>) = buildJsonObject {
        put("role", JsonPrimitive(role))
        put("path", JsonPrimitive(path))
        put("digest", JsonPrimitive(Digests.sha256(file)))
        put("size", JsonPrimitive(file.length()))
        put("license", JsonPrimitive(license))
        if (upstream.isNotEmpty()) {
            put("derived_from", buildJsonObject {
                put("source", JsonPrimitive(upstream.joinToString(", ") { it.label }))
                put("license", JsonPrimitive(upstream.map { it.license }.distinct().joinToString(" AND ")))
            })
        }
    }

    // --- manifest ----------------------------------------------------------------------------------

    private fun manifest(descriptor: JsonObject, effective: String, artifacts: List<JsonObject>, inputs: Map<String, Input>) = buildJsonObject {
        put("schema_version", JsonPrimitive("2"))
        put("kind", JsonPrimitive("cartridge"))
        put("id", descriptor.getValue("id"))
        put("version", descriptor.getValue("version"))
        put("effective_license", JsonPrimitive(effective))
        put("descriptor_digest", JsonPrimitive(CanonicalJson.sha256(descriptor)))
        put("digest_alg", JsonPrimitive("sha256"))
        put("artifacts", JsonArray(artifacts))
        put("provenance", buildJsonObject {
            put("producer", JsonPrimitive(materializerId))
            put("built_at", JsonPrimitive(now().truncatedTo(ChronoUnit.SECONDS).toString()))
            put("inputs", JsonArray(
                Selection.sources(bp, pf).flatMap { s -> s.files.map { f -> provenanceInput("${s.name}/${f.path}", f.digest) } } +
                    inputs.map { (name, i) -> provenanceInput("input:$name", i.digest) },
            ))
            put("blueprint", buildJsonObject {
                put("id", JsonPrimitive(bp.id))
                put("version", JsonPrimitive(bp.version))
                put("digest", JsonPrimitive(blueprint.fileDigest))
                pf.blueprint.source?.let { put("source", it) }
            })
            put("materialization", buildJsonObject {
                put("profile_digest", JsonPrimitive(profile.fileDigest))
                put("target", JsonPrimitive(pf.target))
                if (pf.flavors.isNotEmpty()) put("flavors", JsonArray(pf.flavors.map(::JsonPrimitive)))
                put("toolchain", JsonObject(toolchainUsed.mapValues { JsonPrimitive(it.value) }))
                put("materializer", JsonPrimitive(materializerId))
            })
        })
    }

    private companion object {
        val PLACEHOLDER_SIGNATURE = buildJsonObject {
            put("keyid", JsonPrimitive("unsigned")); put("alg", JsonPrimitive("ed25519")); put("signer", JsonPrimitive("unsigned"))
            put("signature", JsonPrimitive("")); put("signed_at", JsonPrimitive("1970-01-01T00:00:00Z"))
        }
    }

    private fun provenanceInput(name: String, digest: String) = buildJsonObject {
        put("name", JsonPrimitive(name))
        put("digest", JsonPrimitive(digest))
    }
}
