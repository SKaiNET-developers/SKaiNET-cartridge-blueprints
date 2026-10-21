package sk.ainet.cartridge.generator.smollm2

/** Whole-file text read; null when the file does not exist. Snapshot sidecars are small text. */
internal expect fun readTextFile(path: String): String?

internal expect fun envVar(name: String): String?

internal expect fun homeDir(): String?
