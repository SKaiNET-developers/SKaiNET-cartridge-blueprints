package sk.ainet.cartridge.generator.llama32

/** Byte-count of a file; null when it does not exist. The GGUF is pinned by size + SHA-256. */
internal expect fun fileSize(path: String): Long?

/**
 * Install the engine's optimized kernel packs once per process. The packed Q4_K_M weights this
 * cartridge serves (MAPPED residency) fall to the decoding reference kernel without this —
 * correct, but dramatically slower per matmul (the KLlamaJava facade carries the same install
 * for the same reason). JVM adds the FFM row-major pack on top of the common best-provider set.
 */
internal expect fun installKernelPacks()

internal expect fun envVar(name: String): String?

internal expect fun homeDir(): String?
