package sk.ainet.cartridge.blueprint.core

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import sk.ainet.cartridge.blueprint.model.MaterializationException
import sk.ainet.cartridge.blueprint.model.Signing
import java.security.KeyFactory
import java.security.PrivateKey
import java.security.Signature
import java.security.spec.PKCS8EncodedKeySpec
import java.time.Instant
import java.time.temporal.ChronoUnit
import java.util.Base64

/**
 * Step 10. Same construction as the spec's `sign_manifest.py`: Ed25519 over the ASCII hex digest of the
 * canonical manifest with `signatures` removed. The key never comes from the profile — only a reference to it.
 */
object ManifestSigner {

    fun sign(unsigned: JsonObject, signing: Signing, privateKeyPem: String, signedAt: Instant = Instant.now()): JsonObject {
        require("signatures" !in unsigned) { "Manifest is already signed" }
        val digestHex = CanonicalJson.sha256(unsigned).removePrefix("sha256:")
        val signature = Signature.getInstance("Ed25519").run {
            initSign(parsePrivateKey(privateKeyPem))
            update(digestHex.toByteArray(Charsets.US_ASCII))
            sign()
        }
        val entry = buildJsonObject {
            put("keyid", JsonPrimitive(signing.keyid))
            put("alg", JsonPrimitive("ed25519"))
            put("signer", JsonPrimitive(signing.signer))
            put("signature", JsonPrimitive(Base64.getEncoder().encodeToString(signature)))
            put("signed_at", JsonPrimitive(signedAt.truncatedTo(ChronoUnit.SECONDS).toString()))
        }
        return JsonObject(unsigned + ("signatures" to JsonArray(listOf(entry))))
    }

    internal fun parsePrivateKey(pem: String): PrivateKey {
        val body = pem.lineSequence().map { it.trim() }.filter { it.isNotEmpty() && !it.startsWith("-----") }.joinToString("")
        if (body.isEmpty() || !pem.contains("BEGIN PRIVATE KEY")) {
            throw MaterializationException(
                "The signing key is not an unencrypted PKCS#8 PEM (`-----BEGIN PRIVATE KEY-----`). " +
                    "Create one with: openssl genpkey -algorithm ed25519 -out key.pem",
            )
        }
        return try {
            KeyFactory.getInstance("Ed25519").generatePrivate(PKCS8EncodedKeySpec(Base64.getDecoder().decode(body)))
        } catch (e: Exception) {
            throw MaterializationException("The signing key could not be read as an Ed25519 private key: ${e.message}", e)
        }
    }
}
