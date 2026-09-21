package sk.ainet.cartridge.blueprint

import com.sun.net.httpserver.HttpServer
import kotlinx.coroutines.runBlocking
import kotlinx.io.readByteArray
import sk.ainet.data.source.KtorRemoteDataSourceFetcher
import java.net.InetSocketAddress
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

/**
 * Pins down behaviour of SKaiNET's data-source transport that the materializer relies on for credentials.
 * Model hubs redirect large files to a CDN on another host; a bearer token must not travel with that hop.
 */
class SkainetDataSourceBehaviourTest {

    @Test
    fun `authorization header is not forwarded to another authority on redirect`() {
        var seenByTarget: String? = "unset"
        var seenByOrigin: String? = "unset"
        val target = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0).apply {
            createContext("/file") { ex ->
                seenByTarget = ex.requestHeaders.getFirst("Authorization")
                val body = "payload".toByteArray()
                ex.sendResponseHeaders(200, body.size.toLong()); ex.responseBody.use { it.write(body) }
            }
            start()
        }
        val origin = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0).apply {
            createContext("/resolve") { ex ->
                seenByOrigin = ex.requestHeaders.getFirst("Authorization")
                ex.responseHeaders.add("Location", "http://127.0.0.1:${target.address.port}/file")
                ex.sendResponseHeaders(302, -1); ex.close()
            }
            start()
        }
        try {
            val fetcher = KtorRemoteDataSourceFetcher()
            val bytes = runBlocking {
                fetcher.fetch("http://127.0.0.1:${origin.address.port}/resolve", mapOf("Authorization" to "Bearer secret-token")).source.use { it.readByteArray() }
            }
            fetcher.close()
            assertEquals("payload", bytes.decodeToString())
            assertEquals("Bearer secret-token", seenByOrigin, "the origin must receive the credential")
            assertNull(seenByTarget, "the redirect target (another authority) must NOT receive the credential")
        } finally {
            origin.stop(0); target.stop(0)
        }
    }
}
