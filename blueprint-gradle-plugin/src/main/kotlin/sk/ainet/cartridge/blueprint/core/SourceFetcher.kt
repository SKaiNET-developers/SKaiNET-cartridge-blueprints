package sk.ainet.cartridge.blueprint.core

import kotlinx.coroutines.runBlocking
import sk.ainet.cartridge.blueprint.model.MaterializationException
import sk.ainet.cartridge.blueprint.model.Source
import sk.ainet.cartridge.blueprint.model.SourceFile
import sk.ainet.data.source.CachePolicy
import sk.ainet.data.source.DataSourceAuthToken
import sk.ainet.data.source.DataSourceException
import sk.ainet.data.source.DataSourceRequest
import sk.ainet.data.source.DataSourceResolver
import sk.ainet.data.source.JvmDataSourceResolver
import java.io.File
import java.net.URI

/**
 * Step 3: fetch every file of a source and verify its digest — the only place a materialization touches the
 * network.
 *
 * The transport is SKaiNET's own `skainet-data-source`: `hf://`, https and local files, SHA-256 verified while
 * streaming and before anything is committed to its cache, a shared cache (`~/.cache/skainet/data` by default),
 * and Hugging Face tokens that are attached to hub requests only and not forwarded along redirects
 * (pinned by `SkainetDataSourceBehaviourTest`). What stays here is blueprint policy: how a source's
 * `uri` + `revision` + file path (or a profile's mirror) become one request, which schemes a blueprint may use,
 * and what a person building is told when something is wrong.
 */
class SourceFetcher(
    private val resolver: DataSourceResolver,
    private val mirrors: Map<String, String> = emptyMap(),
    private val offline: Boolean = false,
    private val log: (String) -> Unit = {},
) {
    constructor(
        cacheDir: File = JvmDataSourceResolver.defaultCacheDir(),
        mirrors: Map<String, String> = emptyMap(),
        offline: Boolean = false,
        huggingFaceToken: String? = null,
        log: (String) -> Unit = {},
    ) : this(
        resolver = JvmDataSourceResolver(
            cacheDir = cacheDir,
            huggingFaceToken = DataSourceAuthToken.fromOrNull(huggingFaceToken),
            // HF_TOKEN / HUGGING_FACE_HUB_TOKEN, read by SKaiNET itself when no token was handed in.
            useEnvironmentHuggingFaceToken = huggingFaceToken == null,
        ),
        mirrors = mirrors,
        offline = offline,
        log = log,
    )

    /** @return source file path → verified local file */
    fun fetch(source: Source): Map<String, File> =
        source.files.associate { f -> f.path to fetchFile(source, f) }

    private fun fetchFile(source: Source, file: SourceFile): File {
        val uri = requestUri(source, file)
        val expected = file.digest.removePrefix("sha256:")
        fun resolve(policy: CachePolicy) = runBlocking {
            resolver.resolve(DataSourceRequest(uri = uri, cachePolicy = policy, expectedSha256 = expected))
        }

        val artifact = try {
            try {
                resolve(if (offline) CachePolicy.Offline else CachePolicy.Use)
            } catch (e: DataSourceException) {
                // A cached copy that no longer matches (corrupted, or cached before the pin moved) is refreshed
                // once; if the publisher's bytes do not match either, that is fatal.
                if (offline || !e.isDigestMismatch()) throw e
                log("  ${source.name}/${file.path}: cached copy does not match the pinned digest, fetching again")
                resolve(CachePolicy.Refresh)
            }
        } catch (e: DataSourceException) {
            throw explain(source, file, uri, e)
        } catch (e: MaterializationException) {
            throw e
        } catch (e: Exception) {
            throw explain(source, file, uri, e)
        }

        val local = artifact.localPath?.let(::File)
            ?: throw MaterializationException("Source '${source.name}', file '${file.path}': the resolver returned no local file for $uri")
        log("  ${source.name}/${file.path}: ${if (artifact.cacheHit) "cached" else "fetched"}, digest ok")
        return local
    }

    /**
     * One request URI per file. `hf://org/repo` + revision + path → SKaiNET's `hf://org/repo@revision/path`;
     * a mirror replaces the base and keeps the path. A mirror changes where bytes come from, never which bytes.
     */
    internal fun requestUri(source: Source, file: SourceFile): String {
        val base = mirrors[source.name] ?: source.uri
        val path = file.path.trimStart('/')
        if (path.split('/').any { it == ".." }) {
            throw MaterializationException("Source '${source.name}': file path '${file.path}' must not contain '..'")
        }
        return when {
            base.startsWith("hf://") -> "hf://${base.removePrefix("hf://").trim('/')}@${source.revision}/$path"
            base.startsWith("https://") -> base.trimEnd('/') + "/" + path
            base.startsWith("file:") -> File(File(URI(base)), path).path
            else -> throw MaterializationException(
                "Source '${source.name}': unsupported uri scheme in '$base'. A blueprint source (or its mirror) is " +
                    "hf://<org>/<repo>, an https:// URL, or file: — plain http is not fetched.",
            )
        }
    }

    private fun DataSourceException.isDigestMismatch() = message?.contains("SHA-256 mismatch") == true

    private fun explain(source: Source, file: SourceFile, uri: String, e: Exception): MaterializationException {
        val msg = e.message.orEmpty()
        val text = when {
            e is DataSourceException && e.isDigestMismatch() ->
                "Digest mismatch for source '${source.name}', file '${file.path}' ($uri):\n  $msg\n" +
                    "The blueprint pins these bytes. A different file at the same revision means the publisher " +
                    "(or a mirror) changed it; do not proceed."
            msg.contains("offline", ignoreCase = true) ->
                "Source '${source.name}', file '${file.path}' is not in the local cache and the build is offline ($uri)."
            Regex("\\b40[13]\\b").containsMatchIn(msg) ->
                "Access denied for source '${source.name}', file '${file.path}' ($uri): the publisher requires " +
                    "authentication and/or accepted terms. For hf:// sources: accept the model's terms on its page " +
                    "and set HF_TOKEN.\n  $msg"
            Regex("\\b404\\b").containsMatchIn(msg) || msg.contains("not found", ignoreCase = true) ->
                "Source '${source.name}': no file '${file.path}' at revision '${source.revision}' ($uri).\n  $msg"
            else -> "Could not fetch source '${source.name}', file '${file.path}' ($uri): $msg"
        }
        return MaterializationException(text, e)
    }
}
