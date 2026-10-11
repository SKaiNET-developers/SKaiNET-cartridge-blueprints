package sk.ainet.cartridge.embedding.leaf

/** Whole-file text read; null when the file does not exist. Snapshot files are small text. */
internal expect fun readTextFile(path: String): String?

internal expect fun envVar(name: String): String?

internal expect fun homeDir(): String?
