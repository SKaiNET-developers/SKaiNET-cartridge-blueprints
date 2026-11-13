package sk.ainet.cartridge.classification.moonshine.head.recipe

import kotlinx.io.asSink
import kotlinx.io.buffered
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import sk.ainet.cartridge.classification.moonshine.head.LabelSet
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.safetensors.SafeTensorsWriter
import sk.ainet.io.safetensors.StreamingSafeTensorsReader
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

@OptIn(kotlinx.serialization.ExperimentalSerializationApi::class)
internal val json = Json { prettyPrint = true; ignoreUnknownKeys = true; encodeDefaults = true; explicitNulls = false }

/** `--name value` arguments of a step tool. */
internal class Args(args: Array<String>) {
    private val values: Map<String, String> = buildMap {
        require(args.size % 2 == 0) { "arguments come as --name value pairs: ${args.toList()}" }
        for (i in args.indices step 2) {
            require(args[i].startsWith("--")) { "expected --name, got '${args[i]}'" }
            put(args[i].removePrefix("--"), args[i + 1])
        }
    }

    fun optional(name: String): String? = values[name]
    fun string(name: String): String = values[name] ?: error("missing --$name")
    fun file(name: String): File = File(string(name))
    fun int(name: String, default: Int): Int = values[name]?.toInt() ?: default
    fun long(name: String, default: Long): Long = values[name]?.toLong() ?: default
    fun float(name: String, default: Float): Float = values[name]?.toFloat() ?: default
}

internal fun sha256(file: File): String {
    val md = MessageDigest.getInstance("SHA-256")
    file.inputStream().use { input ->
        val buffer = ByteArray(1 shl 16)
        while (true) { val n = input.read(buffer); if (n < 0) break; md.update(buffer, 0, n) }
    }
    return "sha256:" + md.digest().joinToString("") { "%02x".format(it) }
}

/** The `dataset` input (schema/dataset-manifest.schema.json). */
@Serializable
internal data class DatasetManifest(val name: String? = null, val entries: List<Entry>) {
    @Serializable
    data class Entry(val audio: String, val label: String, val split: String, val speaker: String? = null)

    companion object {
        fun read(file: File): DatasetManifest = json.decodeFromString(serializer(), file.readText())
    }
}

/** The `label-set` input (schema/label-set.schema.json): the pack's label set plus what to hold out. */
@Serializable
internal data class LabelSetInput(
    val labels: List<String>,
    @SerialName("reject_label") val rejectLabel: String? = null,
    @SerialName("held_out") val heldOut: List<String> = emptyList(),
) {
    val labelSet: LabelSet get() = LabelSet(labels, rejectLabel)

    init {
        val overlap = heldOut.intersect(labels.toSet())
        require(overlap.isEmpty()) { "held_out labels cannot also be labels: $overlap" }
    }

    companion object {
        fun read(file: File): LabelSetInput = json.decodeFromString(serializer(), file.readText())
    }
}

/** A recording as the head sees it: its encoder rows, and what the data says it is. */
internal class Example(val rows: FloatArray, val count: Int, val label: String) {
    /** Output index the head should give, or [OUT_OF_SET] when no class stands for this label. */
    var target: Int = OUT_OF_SET

    /** True when the label was held out: the head has never trained on this class, not even as "everything else". */
    var unseen: Boolean = false

    companion object {
        const val OUT_OF_SET = -1
    }
}

/**
 * The examples of one split with their targets under a label set.
 *
 * A label of the set is its index. Any other label is the reject class when the set has one, and out of set
 * otherwise. Held-out labels are dropped from `train` and `dev`; in `test` they are out-of-set inputs the head
 * has never heard.
 */
internal fun FeatureStore.examples(split: String, labels: LabelSetInput): List<Example> {
    val index = labels.labels.withIndex().associate { (i, l) -> l to i }
    val reject = labels.labelSet.rejectIndex
    return load(split).mapNotNull { e ->
        val held = e.label in labels.heldOut
        if (held && split != "test") return@mapNotNull null
        e.target = index[e.label] ?: if (reject >= 0) reject else Example.OUT_OF_SET
        e.unseen = held
        e
    }
}

/** 16 kHz mono PCM from a WAV file: 16-bit integer or 32-bit float samples, as floats in [-1, 1]. */
internal object Wav {
    fun read(file: File): FloatArray = decode(file.readBytes(), file.path, EncoderRows.SAMPLE_RATE).second

    /** The samples (channels averaged) and the rate; [requireRate], when set, must be the file's rate. */
    fun decode(bytes: ByteArray, name: String, requireRate: Int? = null): Pair<Int, FloatArray> {
        val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        require(bytes.size >= 12 && String(bytes, 0, 4) == "RIFF" && String(bytes, 8, 4) == "WAVE") { "$name is not a WAV file" }
        var at = 12
        var format = 0; var channels = 0; var rate = 0; var bits = 0
        while (at + 8 <= bytes.size) {
            val id = String(bytes, at, 4)
            val size = b.getInt(at + 4)
            val body = at + 8
            if (id == "fmt ") {
                format = b.getShort(body).toInt() and 0xffff
                channels = b.getShort(body + 2).toInt()
                rate = b.getInt(body + 4)
                bits = b.getShort(body + 14).toInt()
                if (format == 0xfffe) format = b.getShort(body + 24).toInt() and 0xffff   // WAVE_FORMAT_EXTENSIBLE
            } else if (id == "data") {
                require(channels >= 1) { "$name: the data chunk comes before the format" }
                require((format == 1 && bits == 16) || (format == 3 && bits == 32)) {
                    "$name: format $format with $bits bits; expected 16-bit PCM or 32-bit float"
                }
                require(requireRate == null || rate == requireRate) { "$name is $rate Hz; expected $requireRate Hz mono. Resample it first." }
                val bytesPerSample = bits / 8
                val frames = minOf(size, bytes.size - body) / (bytesPerSample * channels)
                val out = FloatArray(frames)
                for (i in 0 until frames) {
                    var sum = 0f
                    for (c in 0 until channels) {
                        val p = body + (i * channels + c) * bytesPerSample
                        sum += if (format == 1) b.getShort(p) / 32768f else b.getFloat(p)
                    }
                    out[i] = sum / channels
                }
                return rate to out
            }
            at = body + size + (size and 1)
        }
        error("$name has no data chunk")
    }

    fun write(file: File, pcm: FloatArray, rate: Int) {
        val data = ByteBuffer.allocate(44 + pcm.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        data.put("RIFF".toByteArray()).putInt(36 + pcm.size * 2).put("WAVE".toByteArray())
        data.put("fmt ".toByteArray()).putInt(16).putShort(1).putShort(1).putInt(rate).putInt(rate * 2).putShort(2).putShort(16)
        data.put("data".toByteArray()).putInt(pcm.size * 2)
        for (s in pcm) data.putShort((s.coerceIn(-1f, 1f) * 32767f).toInt().toShort())
        file.writeBytes(data.array())
    }
}

/**
 * The `features` product: the encoder rows of every manifest entry, in shards of [SHARD] entries.
 * `shard-NNNNN.safetensors` holds `rows` `[total, dim]`; `shard-NNNNN.json` says which entries, with how many rows
 * each. A shard is complete when its json exists (written last), which is what makes the step resumable.
 */
internal class FeatureStore(private val dir: File) {
    @Serializable
    data class Header(val manifest: String, val encoder: String, val dim: Int, @SerialName("max_rows") val maxRows: Int, val entries: Int)

    @Serializable
    data class Shard(val entries: List<Item>)

    @Serializable
    data class Item(val audio: String, val label: String, val split: String, val rows: Int)

    val headerFile = File(dir, "features.json")

    fun header(): Header? = headerFile.takeIf { it.isFile }?.let { json.decodeFromString(Header.serializer(), it.readText()) }

    /** Starts over unless the directory already holds features for exactly this [header]. */
    fun open(header: Header) {
        if (header() != header) {
            dir.deleteRecursively()
            dir.mkdirs()
            headerFile.writeText(json.encodeToString(Header.serializer(), header))
        }
    }

    fun has(shard: Int): Boolean = meta(shard).isFile

    fun write(shard: Int, items: List<Item>, rows: List<FloatArray>, dim: Int) {
        val total = items.sumOf { it.rows }
        val all = FloatArray(total * dim)
        var at = 0
        for ((item, r) in items.zip(rows)) { r.copyInto(all, at, 0, item.rows * dim); at += item.rows * dim }
        val tmp = File(dir, "shard-%05d.tmp".format(shard))
        tmp.outputStream().asSink().buffered().use { sink -> SafeTensorsWriter.write(sink) { tensorF32("rows", intArrayOf(total, dim), all) } }
        check(tmp.renameTo(data(shard))) { "could not move ${tmp.name} into place" }
        meta(shard).writeText(json.encodeToString(Shard.serializer(), Shard(items)))
    }

    /** Every example of [split], in manifest order. */
    fun load(split: String): List<Example> {
        val header = header() ?: error("no features in $dir")
        val out = ArrayList<Example>()
        var shard = 0
        var seen = 0
        while (seen < header.entries) {
            check(has(shard)) { "features are incomplete: shard $shard is missing in $dir" }
            val items = json.decodeFromString(Shard.serializer(), meta(shard).readText()).entries
            seen += items.size
            if (items.any { it.split == split }) {
                val all = StreamingSafeTensorsReader.open(JvmRandomAccessSource.open(data(shard))).use { reader ->
                    HeadWeights.littleEndianFloats(reader.loadTensorData(reader.tensors.single()))
                }
                var at = 0
                for (item in items) {
                    val n = item.rows * header.dim
                    if (item.split == split && item.rows > 0) out += Example(all.copyOfRange(at, at + n), item.rows, item.label)
                    at += n
                }
            }
            shard++
        }
        return out
    }

    private fun data(shard: Int) = File(dir, "shard-%05d.safetensors".format(shard))
    private fun meta(shard: Int) = File(dir, "shard-%05d.json".format(shard))

    companion object {
        const val SHARD = 64
    }
}
