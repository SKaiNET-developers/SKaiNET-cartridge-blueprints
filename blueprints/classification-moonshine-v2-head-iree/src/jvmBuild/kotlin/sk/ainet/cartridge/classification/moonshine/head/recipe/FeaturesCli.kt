package sk.ainet.cartridge.classification.moonshine.head.recipe

import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

/**
 * Step `features`: every recording of the dataset manifest through the frozen encoder ([EncoderRows]), cached.
 *
 *   --manifest <dataset.json> --encoder <dir with config.json, model.safetensors> --out <dir>
 *   [--max-rows 256] [--workers <n>]
 *
 * Resumable: a finished shard is never recomputed, so an interrupted run continues where it stopped.
 */
public fun main(args: Array<String>) {
    val a = Args(args)
    val manifestFile = a.file("manifest")
    val encoderDir = a.file("encoder")
    val maxRows = a.int("max-rows", 256)
    val workers = a.int("workers", maxOf(1, Runtime.getRuntime().availableProcessors() / 2))
    val manifest = DatasetManifest.read(manifestFile)
    val store = FeatureStore(a.file("out"))

    val dim = EncoderRows(encoderDir).use { it.dim }
    store.open(
        FeatureStore.Header(
            manifest = sha256(manifestFile),
            encoder = sha256(File(encoderDir, "model.safetensors")),
            dim = dim, maxRows = maxRows, entries = manifest.entries.size,
        ),
    )
    val shards = manifest.entries.chunked(FeatureStore.SHARD)
    val todo = shards.indices.filter { !store.has(it) }
    println("features: ${manifest.entries.size} recordings, ${shards.size} shards, ${todo.size} to compute, $workers workers")

    val encoders = ThreadLocal.withInitial { EncoderRows(encoderDir) }
    val pool = Executors.newFixedThreadPool(workers)
    val done = AtomicInteger(shards.size - todo.size)
    val started = System.nanoTime()
    var seconds = 0.0
    val lock = Any()
    try {
        todo.map { shard ->
            pool.submit {
                val encoder = encoders.get()
                var audio = 0.0
                val rows = ArrayList<FloatArray>()
                val items = shards[shard].map { entry ->
                    val pcm = Wav.read(manifestFile.absoluteFile.parentFile.resolve(entry.audio))
                    audio += pcm.size.toDouble() / EncoderRows.SAMPLE_RATE
                    val (r, n) = encoder.rows(pcm, maxRows)
                    rows += r
                    FeatureStore.Item(entry.audio, entry.label, entry.split, n)
                }
                store.write(shard, items, rows, dim)
                synchronized(lock) {
                    seconds += audio
                    val elapsed = (System.nanoTime() - started) / 1e9
                    println("features: shard ${done.incrementAndGet()}/${shards.size}, ${"%.1f".format(seconds / elapsed)}x real time")
                }
            }
        }.forEach { it.get() }
    } finally {
        pool.shutdown()
    }
}
