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
import org.gradle.workers.WorkAction
import org.gradle.workers.WorkParameters
import org.gradle.workers.WorkerExecutor
import javax.inject.Inject
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
abstract class BlueprintExtension @javax.inject.Inject constructor(private val project: Project) {
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

    /**
     * Where `blueprintFetch` exposes each verified source file under a stable path,
     * `<dir>/<source name>/<file path>` — a link into the shared cache. Default: `build/blueprint/sources`.
     */
    abstract val sourceLinksDir: DirectoryProperty

    internal abstract val productPaths: MapProperty<String, String>
    internal abstract val productFiles: ConfigurableFileCollection
    internal lateinit var fetchTask: org.gradle.api.tasks.TaskProvider<BlueprintFetchTask>

    /**
     * The target the profile selected, with the blueprint's `params` for it as strings — what a module's
     * `compile` / `build-runtime` tasks are configured from. Resolved lazily; fails with the usual message when no
     * profile was given.
     */
    val target: Provider<SelectedTarget>
        get() = blueprintFile.zip(profileFile) { b, p ->
            materializing {
                val blueprint = Documents.blueprint(b.asFile).value
                val profile = Documents.profile(p.asFile).value
                val t = blueprint.target(profile.target)
                SelectedTarget(
                    id = t.id,
                    abi = (t.target["abi"] as? kotlinx.serialization.json.JsonPrimitive)?.content.orEmpty(),
                    cartridgeId = DescriptorResolver.cartridgeId(blueprint, profile),
                    params = t.params.orEmpty().mapValues { (_, v) -> (v as? kotlinx.serialization.json.JsonPrimitive)?.content ?: v.toString() },
                    toolchain = blueprint.toolchain.mapValues { it.value.version },
                    images = blueprint.toolchain.mapNotNull { (k, v) -> v.image?.let { k to it } }.toMap(),
                    flavors = profile.flavors,
                )
            }
        }

    /**
     * A fetched, digest-verified source file, for wiring into a model-specific step. Depends on `blueprintFetch`.
     * `source` and `path` are the names used in `blueprint.json`.
     */
    fun sourceFile(source: String, path: String): Provider<org.gradle.api.file.RegularFile> =
        fetchTask.flatMap { it.linksDir.file("$source/$path") }

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

    /**
     * A product that belongs to one [flavor]: registered — and its task chain run — only when the profile selects
     * that flavor. What the blueprint's `outputs[].flavor` says, applied to the recipe.
     */
    fun product(name: String, location: Provider<out FileSystemLocation>, flavor: String) {
        // flatMap, not zip: a zip carries `location`'s task dependency even when the flavor is off; a flatMap that
        // never touches `location` carries none, so an unselected flavor's export/compile chain is not scheduled.
        val selected = target.map { flavor in it.flavors }
        val gated: Provider<List<FileSystemLocation>> = selected.flatMap { on ->
            if (on) location.map { listOf(it) } else project.provider { emptyList<FileSystemLocation>() }
        }
        productPaths.putAll(gated.map { locs -> locs.associate { name to it.asFile.absolutePath } })
        productFiles.from(gated)
    }
}

/** See [BlueprintExtension.target]. Serializable so it can be a task input under the configuration cache. */
data class SelectedTarget(
    val id: String,
    val abi: String,
    val cartridgeId: String,
    val params: Map<String, String>,
    val toolchain: Map<String, String>,
    val images: Map<String, String>,
    /** The flavors the profile selected (empty when the blueprint has none). */
    val flavors: List<String> = emptyList(),
) : java.io.Serializable {
    fun param(name: String): String = params[name]
        ?: throw GradleException("Blueprint target '$id' has no param '$name' (has: ${params.keys})")

    fun image(tool: String): String = images[tool]
        ?: throw GradleException("Blueprint toolchain entry '$tool' has no container image")
}

class BlueprintPlugin : Plugin<Project> {
    override fun apply(project: Project) {
        val ext = project.extensions.create("blueprint", BlueprintExtension::class.java, project)
        ext.blueprintFile.convention(project.layout.projectDirectory.file("blueprint.json"))
        ext.sourcesDir.convention(project.layout.dir(project.provider { JvmDataSourceResolver.defaultCacheDir() }))
        ext.outputDir.convention(project.layout.buildDirectory.dir("cartridge"))
        ext.sourceLinksDir.convention(project.layout.buildDirectory.dir("blueprint/sources"))
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
            linksDir.set(ext.sourceLinksDir)
            offline.set(project.gradle.startParameter.isOffline)
        }
        ext.fetchTask = fetch

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
abstract class BlueprintFetchTask @Inject constructor(private val workers: WorkerExecutor) : DefaultTask() {
    @get:InputFile @get:PathSensitive(PathSensitivity.NONE) abstract val blueprintFile: RegularFileProperty
    @get:InputFile @get:Optional @get:PathSensitive(PathSensitivity.NONE) abstract val profileFile: RegularFileProperty
    // A shared, machine-wide cache — not a task output: Gradle must neither own nor clean it.
    @get:Internal abstract val sourcesDir: DirectoryProperty
    /** Stable, per-project paths to the verified files: `<linksDir>/<source>/<path>` → the cached file. */
    @get:OutputDirectory abstract val linksDir: DirectoryProperty
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
        Selection.sources(blueprint.value, profile.value).forEach { source ->
            logger.lifecycle("source '${source.name}' (${source.license}, ${source.uri} @ ${source.revision})")
        }
        // The download runs in a worker with its own classloader: SKaiNET's data-source module brings Ktor and a
        // kotlinx-coroutines newer than the one Gradle itself puts on every plugin's classpath, and the two must
        // not meet. Tokens (HF_TOKEN / HUGGING_FACE_HUB_TOKEN) are read and scoped by the data-source module.
        val queue = workers.classLoaderIsolation { classpath.from(FetchWork.pluginClasspath()) }
        queue.submit(FetchWork::class.java) {
            blueprintFile.set(this@BlueprintFetchTask.blueprintFile)
            profileFile.set(this@BlueprintFetchTask.profileFile)
            sourcesDir.set(this@BlueprintFetchTask.sourcesDir)
            linksDir.set(this@BlueprintFetchTask.linksDir)
            offline.set(this@BlueprintFetchTask.offline)
        }
        queue.await()
    }
}

/** The fetch itself, run under classloader isolation (see [BlueprintFetchTask]). */
abstract class FetchWork : WorkAction<FetchWork.Params> {
    interface Params : WorkParameters {
        val blueprintFile: RegularFileProperty
        val profileFile: RegularFileProperty
        val sourcesDir: DirectoryProperty
        val linksDir: DirectoryProperty
        val offline: Property<Boolean>
    }

    override fun execute() {
        val blueprint = Documents.blueprint(parameters.blueprintFile.get().asFile)
        val profile = Documents.profile(parameters.profileFile.get().asFile)
        val fetcher = SourceFetcher(
            cacheDir = parameters.sourcesDir.get().asFile,
            mirrors = profile.value.mirrors,
            offline = parameters.offline.get(),
            log = { println(it) },
        )
        Selection.sources(blueprint.value, profile.value).forEach { source ->
            fetcher.fetch(source).forEach { (path, cached) -> link(File(parameters.linksDir.get().asFile, "${source.name}/$path"), cached) }
        }
    }

    companion object {
        /** The plugin's own jars: the classloader Gradle built for this plugin holds exactly its declared dependencies. */
        fun pluginClasspath(): List<File> {
            val loader = FetchWork::class.java.classLoader
            val urls = generateSequence(loader) { it.parent }.takeWhile { it != null }
                .filterIsInstance<java.net.URLClassLoader>().firstOrNull()?.urLs
                ?: (loader as? java.net.URLClassLoader)?.urLs
            val fromLoader = urls?.map { File(it.toURI()) }.orEmpty()
            if (fromLoader.isNotEmpty()) return fromLoader
            // Fallback: the jar this class came from plus whatever sits next to it.
            val self = File(FetchWork::class.java.protectionDomain.codeSource.location.toURI())
            return listOf(self)
        }
    }
}

/** A symlink into the cache where the platform allows it, otherwise a copy — weights are large, links are free. */
internal fun link(at: File, target: File) {
    at.parentFile.mkdirs()
    java.nio.file.Files.deleteIfExists(at.toPath())
    try {
        java.nio.file.Files.createSymbolicLink(at.toPath(), target.toPath().toAbsolutePath())
    } catch (_: Exception) {
        target.copyTo(at, overwrite = true)
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
