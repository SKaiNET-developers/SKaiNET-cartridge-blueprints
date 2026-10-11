package sk.ainet.cartridge.embedding.leaf

/**
 * The fixed inputs whose vectors define `mdbr-leaf-ir-v1`. The JVM reference values live in
 * `samples/golden-vectors.json` and inline in [LeafGoldenTest]; `GoldenRegenerator` (jvmTest)
 * rewrites both when the space is deliberately re-pinned.
 */
internal object GoldenCases {
    const val PREFIX_DIMS: Int = 8

    val TEXTS: List<String> = listOf(
        "which JEP introduced virtual threads",
        "Virtual Threads is an API change in Java 21, specified by JEP 444.",
        "Class files compiled for Java 8 carry major version 52.0.",
    )
}
