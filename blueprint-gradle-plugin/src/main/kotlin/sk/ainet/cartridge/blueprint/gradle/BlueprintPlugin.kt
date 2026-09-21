package sk.ainet.cartridge.blueprint.gradle

import org.gradle.api.DefaultTask
import org.gradle.api.GradleException
import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.api.file.ConfigurableFileCollection
import org.gradle.api.file.DirectoryProperty
import org.gradle.api.file.FileSystemLocation
import org.gradle.api.file.RegularFileProperty
import org.gradle.api.provider.MapProperty
import org.gradle.api.provider.Property
import org.gradle.api.provider.Provider
import org.gradle.api.tasks.Input
import org.gradle.api.tasks.InputFile
import org.gradle.api.tasks.InputFiles
import org.gradle.api.tasks.Internal
import org.gradle.api.tasks.Optional
import org.gradle.api.tasks.OutputDirectory
import org.gradle.api.tasks.PathSensitive
import org.gradle.api.tasks.PathSensitivity
import org.gradle.api.tasks.TaskAction
import org.gradle.work.DisableCachingByDefault
import sk.ainet.cartridge.blueprint.core.DescriptorResolver
import sk.ainet.cartridge.blueprint.core.Documents
import sk.ainet.cartridge.blueprint.core.LicenseGate
import sk.ainet.cartridge.blueprint.core.Materializer
import sk.ainet.cartridge.blueprint.core.Selection
import sk.ainet.cartridge.blueprint.core.SourceFetcher
import sk.ainet.cartridge.blueprint.model.MaterializationException
import sk.ainet.data.source.JvmDataSourceResolver
import java.io.File

/**
 * ```
 * plugins { id("sk.ainet.cartridge.blueprint") }
 *
 * blueprint {
 *     blueprintFile.set(layout.projectDirectory.file("blueprint.json"))   // default
 *     product("graphs-vmfb", compileGraphs.flatMap { it.outputDir })       // what a build task produced
 * }
 * ```
 * `./gradlew materializeCartridge -Pprofile=profiles/my-device.json`
 */
abstract class BlueprintExtension {
    /** The blueprint. Default: `blueprint.json` in the project directory. */
    abstract val blueprintFile: RegularFileProperty

    /** The materialization profile. Default: the `-Pprofile=<path>` Gradle property. */
    abstract val profileFile: RegularFileProperty

    /**
     * Where fetched, digest-verified sources are cached. Default: SKaiNET's shared data cache
     * (`~/.cache/skainet/data`), so a checkpoint is downloaded once per machine, not once per project.
     */
    abstract val sourcesDir: DirectoryProperty

    /** Parent of the produced `pack_dir`. Default: `build/cartridge`. */
    abstract val outputDir: DirectoryProperty

    internal abstract val productPaths: MapProperty<String, String>
    internal abstract val productFiles: ConfigurableFileCollection

    /**
     * Registers what a step of kind convert / quantize / export / compile / build-runtime produced, under the
     * name the blueprint's `steps[].produces` and `outputs[].from` use. Task dependencies travel with [location].
     */
    fun product(name: String, location: Provider<out FileSystemLocation>) {
        productPaths.put(name, location.map { it.asFile.absolutePath })
        productFiles.from(location)
    }

    fun product(name: String, location: File) {
        productPaths.put(name, location.absolutePath)
        productFiles.from(location)
    }
}

class BlueprintPlugin : Plugin<Project> {
    override fun apply(project: Project) {
        val ext = project.extensions.create("blueprint", BlueprintExtension::class.java)
        ext.blueprintFile.convention(project.layout.projectDirectory.file("blueprint.json"))
        ext.sourcesDir.convention(project.layout.dir(project.provider { JvmDataSourceResolver.defaultCacheDir() }))
        ext.outputDir.convention(project.layout.buildDirectory.dir("cartridge"))
        project.providers.gradleProperty("profile").orNull?.let { ext.profileFile.convention(project.layout.projectDirectory.file(it)) }

        val materializerId = "$PLUGIN_ID@${pluginVersion()}"

        project.tasks.register("blueprintValidate", BlueprintValidateTask::class.java) {
            group = GROUP
            description = "Validates blueprint.json (and the profile, when given) against the cartridge-spec schemas."
            blueprintFile.set(ext.blueprintFile)
            profileFile.set(ext.profileFile)
        }

        val fetch = project.tasks.register("blueprintFetch", BlueprintFetchTask::class.java) {
            group = GROUP
            description = "Checks license acceptance, then fetches the blueprint's sources and verifies every digest."
            blueprintFile.set(ext.blueprintFile)
            profileFile.set(ext.profileFile)
            sourcesDir.set(ext.sourcesDir)
            offline.set(project.gradle.startParameter.isOffline)
        }

        project.tasks.register("materializeCartridge", MaterializeCartridgeTask::class.java) {
            group = GROUP
            description = "Resolves the descriptor, packs the pack_dir with a manifest naming the blueprint, and signs it."
            dependsOn(fetch)
            blueprintFile.set(ext.blueprintFile)
            profileFile.set(ext.profileFile)
            sourcesDir.set(ext.sourcesDir)
            offline.set(project.gradle.startParameter.isOffline)
            outputDir.set(ext.outputDir)
            productPaths.set(ext.productPaths)
            productFiles.from(ext.productFiles)
            this.materializerId.set(materializerId)
            signingKeyFile.set(project.providers.gradleProperty("signingKeyFile").map { project.layout.projectDirectory.file(it) })
        }
    }

    private fun pluginVersion(): String =
        BlueprintPlugin::class.java.`package`?.implementationVersion ?: "dev"

    companion object {
        const val PLUGIN_ID = "sk.ainet.cartridge.blueprint"
        const val GROUP = "cartridge blueprint"
    }
}

internal fun <T> materializing(block: () -> T): T = try {
    block()
} catch (e: MaterializationException) {
    throw GradleException(e.message ?: "Materialization failed", e)
}

internal fun RegularFileProperty.requireProfile(): File =
    orNull?.asFile ?: throw GradleException(
        "No materialization profile. Pass one with -Pprofile=<path> (or set blueprint { profileFile }). " +
            "A profile is your half of the build: target, cartridge version, license acceptances, inputs, signing.",
    )

@DisableCachingByDefault(because = "Validation only")
abstract class BlueprintValidateTask : DefaultTask() {
    @get:InputFile @get:PathSensitive(PathSensitivity.NONE) abstract val blueprintFile: RegularFileProperty
    @get:InputFile @get:Optional @get:PathSensitive(PathSensitivity.NONE) abstract val profileFile: RegularFileProperty

    @TaskAction
    fun validate() = materializing {
        val blueprint = Documents.blueprint(blueprintFile.get().asFile)
        logger.lifecycle("blueprint ${blueprint.value.id}@${blueprint.value.version}: valid (${blueprint.fileDigest})")
        profileFile.orNull?.asFile?.let { file ->
            val profile = Documents.profile(file)
            Documents.requireMatch(blueprint, profile)
            blueprint.value.target(profile.value.target)
            Selection.sources(blueprint.value, profile.value)
            logger.lifecycle("profile ${file.name}: valid, target '${profile.value.target}' → cartridge " +
                "${DescriptorResolver.cartridgeId(blueprint.value, profile.value)}@${profile.value.cartridge.version}")
        }
    }
}

@DisableCachingByDefault(because = "Downloads are cached by SKaiNET's data-source cache, keyed and verified by digest")
abstract class BlueprintFetchTask : DefaultTask() {
    @get:InputFile @get:PathSensitive(PathSensitivity.NONE) abstract val blueprintFile: RegularFileProperty
    @get:InputFile @get:Optional @get:PathSensitive(PathSensitivity.NONE) abstract val profileFile: RegularFileProperty
    // A shared, machine-wide cache — not a task output: Gradle must neither own nor clean it.
    @get:Internal abstract val sourcesDir: DirectoryProperty
    @get:Input abstract val offline: Property<Boolean>

    init {
        outputs.upToDateWhen { false }   // cheap when cached; always re-verifies the digests
    }

    @TaskAction
    fun fetch() = materializing {
        val blueprint = Documents.blueprint(blueprintFile.get().asFile)
        val profile = Documents.profile(profileFile.requireProfile())
        Documents.requireMatch(blueprint, profile)
        LicenseGate.check(blueprint.value, profile.value)
        LicenseGate.restricted(blueprint.value, profile.value).forEach {
            logger.warn("source '${it.name}' is `restricted` (${it.license}): check that your use satisfies its conditions" +
                (it.licenseUrl?.let { url -> " — $url" } ?: ""))
        }
        // Tokens (HF_TOKEN / HUGGING_FACE_HUB_TOKEN) are read and scoped by SKaiNET's data-source module itself.
        val fetcher = SourceFetcher(
            cacheDir = sourcesDir.get().asFile,
            mirrors = profile.value.mirrors,
            offline = offline.get(),
            log = { logger.lifecycle(it) },
        )
        Selection.sources(blueprint.value, profile.value).forEach { source ->
            logger.lifecycle("source '${source.name}' (${source.license}, ${source.uri} @ ${source.revision})")
            fetcher.fetch(source)
        }
    }
}

@DisableCachingByDefault(because = "Signs with a key and stamps a build time")
abstract class MaterializeCartridgeTask : DefaultTask() {
    @get:InputFile @get:PathSensitive(PathSensitivity.NONE) abstract val blueprintFile: RegularFileProperty
    @get:InputFile @get:Optional @get:PathSensitive(PathSensitivity.NONE) abstract val profileFile: RegularFileProperty
    @get:Internal abstract val sourcesDir: DirectoryProperty
    @get:Input abstract val offline: Property<Boolean>
    @get:OutputDirectory abstract val outputDir: DirectoryProperty
    @get:Input abstract val productPaths: MapProperty<String, String>
    @get:InputFiles @get:PathSensitive(PathSensitivity.RELATIVE) abstract val productFiles: ConfigurableFileCollection
    @get:Input abstract val materializerId: Property<String>
    @get:InputFile @get:Optional @get:PathSensitive(PathSensitivity.NONE) abstract val signingKeyFile: RegularFileProperty

    @TaskAction
    fun materialize() = materializing {
        val blueprint = Documents.blueprint(blueprintFile.get().asFile)
        val profile = Documents.profile(profileFile.requireProfile())
        // Already fetched and verified by blueprintFetch; this re-reads the cache and re-checks the digests.
        val fetcher = SourceFetcher(cacheDir = sourcesDir.get().asFile, mirrors = profile.value.mirrors, offline = offline.get())
        val fetched = Selection.sources(blueprint.value, profile.value).associate { it.name to fetcher.fetch(it) }

        val key = signingKey(profile.value.signing?.keyRef)
        val id = DescriptorResolver.cartridgeId(blueprint.value, profile.value)
        val result = Materializer(
            blueprint = blueprint,
            profile = profile,
            fetched = fetched,
            products = productPaths.get().mapValues { File(it.value) },
            materializerId = materializerId.get(),
        ).materialize(File(outputDir.get().asFile, id), key)

        logger.lifecycle("cartridge $id@${profile.value.cartridge.version} → ${result.packDir}")
        logger.lifecycle("  effective license: ${(result.manifest["effective_license"])}")
        if (!result.signed) {
            logger.warn(
                "  UNSIGNED: no signing key was available, so manifest.json has no `signatures` and is not a valid " +
                    "cartridge manifest yet. Set `signing` in the profile and provide the key via the environment " +
                    "variable its `key_ref` names, or -PsigningKeyFile=<pem>.",
            )
        }
    }

    private fun signingKey(keyRef: String?): String? =
        signingKeyFile.orNull?.asFile?.readText()
            ?: keyRef?.let { System.getenv(it) }?.takeIf { it.isNotBlank() }
}
