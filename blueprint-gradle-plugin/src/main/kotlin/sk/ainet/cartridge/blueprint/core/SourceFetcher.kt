package sk.ainet.cartridge.blueprint.core

import sk.ainet.cartridge.blueprint.model.MaterializationException
import sk.ainet.cartridge.blueprint.model.Source
import sk.ainet.cartridge.blueprint.model.SourceFile
import java.io.File
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.time.Duration

/**
 * Step 3: fetch every file of a source and verify its digest. The only place a materialization touches the
 * network. A mirror changes where bytes come from, never which bytes: the digests are the blueprint's.
 */
class SourceFetcher(
    private val cacheDir: File,
    private val mirrors: Map<String, String> = emptyMap(),
    private val bearerToken: (host: String) -> String? = { null },
    private val log: (String) -> Unit = {},
) {
    private val http: HttpClient by lazy {
        HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NEVER).connectTimeout(Duration.ofSeconds(30)).build()
    }

    /** @return source file path → verified local file */
    fun fetch(source: Source): Map<String, File> =
        source.files.associate { f -> f.path to fetchFile(source, f) }

    private fun fetchFile(source: Source, file: SourceFile): File {
        val dest = File(cacheDir, "${source.name}/${source.revision}/${file.path}").canonicalFile
        if (!dest.path.startsWith(cacheDir.canonicalPath + File.separator)) {
            throw MaterializationException("Source '${source.name}': file path '${file.path}' escapes the cache directory")
        }
        if (dest.isFile && Digests.sha256(dest) == file.digest) {
            log("  ${source.name}/${file.path}: cached, digest ok")
            return dest
        }
        val url = resolve(source, file)
        log("  ${source.name}/${file.path}: fetching $url")
        dest.parentFile.mkdirs()
        val tmp = File(dest.parentFile, dest.name + ".part")
        try {
            download(url, tmp)
            val actual = Digests.sha256(tmp)
            if (actual != file.digest) {
                throw MaterializationException(
                    "Digest mismatch for source '${source.name}', file '${file.path}':\n" +
                        "  expected ${file.digest}\n  actual   $actual\n  from     $url\n" +
                        "The blueprint pins these bytes. A different file at the same revision means the publisher " +
                        "(or a mirror) changed it; do not proceed.",
                )
            }
            Files.move(tmp.toPath(), dest.toPath(), StandardCopyOption.REPLACE_EXISTING)
        } finally {
            tmp.delete()
        }
        return dest
    }

    /** `hf://org/repo` → the hub's immutable-revision URL; a mirror replaces the base and keeps `<path>`. */
    internal fun resolve(source: Source, file: SourceFile): URI {
        mirrors[source.name]?.let { return URI(it.trimEnd('/') + "/" + encodePath(file.path)) }
        val uri = source.uri
        return when {
            uri.startsWith("hf://") ->
                URI("https://huggingface.co/${uri.removePrefix("hf://").trim('/')}/resolve/${source.revision}/${encodePath(file.path)}")
            uri.startsWith("https://") || uri.startsWith("file:") ->
                URI(uri.trimEnd('/') + "/" + encodePath(file.path))
            else -> throw MaterializationException("Source '${source.name}': unsupported uri scheme in '$uri' (hf://, https://, file:)")
        }
    }

    private companion object {
        const val MAX_REDIRECTS = 5
        val REDIRECTS = setOf(301, 302, 303, 307, 308)
    }

    private fun encodePath(path: String): String =
        path.split('/').joinToString("/") { URI(null, null, it, null).rawPath }

    private fun download(url: URI, dest: File) {
        if (url.scheme == "file") {
            val src = File(url)
            if (!src.isFile) throw MaterializationException("Not found: $url")
            Files.copy(src.toPath(), dest.toPath(), StandardCopyOption.REPLACE_EXISTING)
            return
        }
        // Redirects are followed by hand so that a credential is only ever sent to the host it was issued for:
        // model hubs redirect large files to a CDN, and an Authorization header must not travel with that hop.
        var current = url
        repeat(MAX_REDIRECTS + 1) {
            if (current.scheme != "https") throw MaterializationException("Refusing to fetch over '${current.scheme}': $current")
            val request = HttpRequest.newBuilder(current).timeout(Duration.ofMinutes(60)).GET().apply {
                if (current.host == url.host) bearerToken(current.host)?.let { header("Authorization", "Bearer $it") }
            }.build()
            val response = http.send(request, HttpResponse.BodyHandlers.ofFile(dest.toPath()))
            val status = response.statusCode()
            when {
                status in 200..299 -> return
                status in REDIRECTS -> {
                    dest.delete()
                    val location = response.headers().firstValue("Location").orElseThrow {
                        MaterializationException("HTTP $status without a Location header for $current")
                    }
                    current = current.resolve(location)
                }
                else -> {
                    dest.delete()
                    val hint = when (status) {
                        401, 403 -> " — the publisher requires authentication and/or accepted terms for this file " +
                            "(for hf:// sources: accept the model's terms on its page and set HF_TOKEN)"
                        404 -> " — no such file at this revision"
                        else -> ""
                    }
                    throw MaterializationException("HTTP $status for $current$hint")
                }
            }
        }
        throw MaterializationException("Too many redirects fetching $url")
    }
}
