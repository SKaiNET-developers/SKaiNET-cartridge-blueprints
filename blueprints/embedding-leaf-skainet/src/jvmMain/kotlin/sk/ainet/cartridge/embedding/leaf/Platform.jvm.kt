package sk.ainet.cartridge.embedding.leaf

import java.io.File

internal actual fun readTextFile(path: String): String? =
    File(path).takeIf { it.isFile }?.readText()

internal actual fun envVar(name: String): String? = System.getenv(name)

internal actual fun homeDir(): String? = System.getProperty("user.home")
