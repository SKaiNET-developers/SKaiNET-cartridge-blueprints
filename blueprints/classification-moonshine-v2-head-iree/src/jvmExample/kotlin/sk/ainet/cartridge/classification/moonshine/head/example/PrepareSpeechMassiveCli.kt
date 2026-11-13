package sk.ainet.cartridge.classification.moonshine.head.example

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import sk.ainet.cartridge.classification.moonshine.head.recipe.DatasetManifest
import sk.ainet.cartridge.classification.moonshine.head.recipe.Wav
import sk.ainet.cartridge.classification.moonshine.head.recipe.sha256
import java.io.File
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.sql.DriverManager
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/** `samples/speech-massive-de.source.json`: where the example data is published, pinned. */
@Serializable
internal data class ExampleSource(val name: String, val url: String, val revision: String, val license: String, val terms: String, val files: List<SourceFile>) {
    @Serializable
    data class SourceFile(val path: String, val digest: String, val size: Long, val split: String)
}

/**
 * Not a recipe step: the example's data adapter. Turns the published Speech-MASSIVE parquet files into what the
 * recipe's `dataset` input is: a manifest and 16 kHz mono WAV files. The only code here that knows parquet (DuckDB);
 * nothing of it reaches a cartridge.
 *
 *   --source <speech-massive-de.source.json> --out <dir> --cache <dir> --accept-license <the source's license>
 *   [--only <regex over file paths>] [--dev-speakers 10]
 *
 * The data has a license. Nothing is downloaded until `--accept-license` names it exactly: the choice is yours,
 * and it goes into your profile as the `dataset` input's license.
 *
 * `train` files become `train` and `dev` (every `--dev-speakers`-th speaker goes to dev, whole); `test` files
 * become `test`.
 */
public fun main(args: Array<String>) {
    val a = args.toList().chunked(2).associate { it[0].removePrefix("--") to it[1] }
    val source = Json { ignoreUnknownKeys = true }.decodeFromString(ExampleSource.serializer(), File(a.getValue("source")).readText())
    val out = File(a.getValue("out")).apply { mkdirs() }
    val cache = File(a.getValue("cache")).apply { mkdirs() }
    require(a["accept-license"] == source.license) {
        "${source.name} is published under ${source.license} (${source.terms}). Read the terms; if you accept them, " +
            "pass --accept-license ${source.license} (Gradle: -PacceptDatasetLicense=${source.license})."
    }
    val only = a["only"]?.let(::Regex)
    val devEvery = a["dev-speakers"]?.toInt() ?: 10
    val files = source.files.filter { only == null || only.containsMatchIn(it.path) }
    require(files.isNotEmpty()) { "--only matches none of ${source.files.map { it.path }}" }

    val http = HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL).build()
    val local = files.associateWith { f ->
        val file = File(cache, f.path.substringAfterLast('/'))
        if (!file.isFile || sha256(file) != f.digest) {
            println("prepare: fetching ${f.path} (${f.size / 1_000_000} MB)")
            val request = HttpRequest.newBuilder(URI("${source.url}/resolve/${source.revision}/${f.path}"))
            System.getenv("HF_TOKEN")?.let { request.header("Authorization", "Bearer $it") }
            val response = http.send(request.build(), HttpResponse.BodyHandlers.ofFile(file.toPath()))
            check(response.statusCode() == 200) { "${f.path}: HTTP ${response.statusCode()}" }
            check(sha256(file) == f.digest) { "${f.path}: digest mismatch, expected ${f.digest}. The publisher changed a pinned file; do not proceed." }
        }
        file
    }

    val entries = ArrayList<DatasetManifest.Entry>()
    DriverManager.getConnection("jdbc:duckdb:").use { db ->
        // Speakers of the train files, in a fixed order, so the dev speakers are the same on every run.
        val trainFiles = local.filterKeys { it.split == "train" }.values.joinToString(", ") { "'${it.path}'" }
        val devSpeakers = HashSet<String>()
        if (trainFiles.isNotEmpty()) {
            db.createStatement().use { s ->
                s.executeQuery("SELECT DISTINCT speaker_id FROM read_parquet([$trainFiles]) ORDER BY speaker_id").use { r ->
                    var i = 0
                    while (r.next()) if (i++ % devEvery == devEvery - 1) devSpeakers += r.getString(1)
                }
            }
        }
        for ((f, file) in local) {
            db.createStatement().use { s ->
                s.executeQuery("SELECT id, intent_str, speaker_id, audio.bytes FROM read_parquet('${file.path}')").use { r ->
                    while (r.next()) {
                        val id = r.getString(1); val label = r.getString(2); val speaker = r.getString(3)
                        val split = if (f.split == "train" && speaker in devSpeakers) "dev" else f.split
                        val relative = "wav/${f.split}/$id.wav"
                        val wav = File(out, relative)
                        if (!wav.isFile) {
                            val blob = r.getBlob(4)
                            val (rate, pcm) = Wav.decode(blob.getBytes(1, blob.length().toInt()), "$id in ${f.path}")
                            wav.parentFile.mkdirs()
                            Wav.write(wav, resample(pcm, rate, 16_000), 16_000)
                        }
                        entries += DatasetManifest.Entry(relative, label, split, speaker)
                    }
                }
            }
            println("prepare: ${f.path} done, ${entries.size} recordings so far")
        }
    }
    val manifest = File(out, "dataset.json")
    manifest.writeText(Json.encodeToString(DatasetManifest.serializer(), DatasetManifest("${source.name} @ ${source.revision.take(8)}", entries)) + "\n")
    println("prepare: ${manifest.path}: " + entries.groupingBy { it.split }.eachCount() + ", license ${source.license}")
}

/**
 * Band-limited resampling: a Hann-windowed sinc, cut off at the lower of the two Nyquist frequencies. The rates'
 * ratio is rational, so the kernel repeats: one row of taps per phase, computed once.
 */
internal fun resample(pcm: FloatArray, from: Int, to: Int): FloatArray {
    if (from == to) return pcm
    var g = from; var r = to
    while (r != 0) { val t = g % r; g = r; r = t }
    val phases = to / g          // an output sample sits at input position i * (from / g) / phases
    val advance = from / g
    val cutoff = minOf(1.0, to.toDouble() / from)
    val half = (16 / cutoff).toInt()
    val taps = Array(phases) { phase ->
        FloatArray(2 * half + 1) { k ->
            val distance = (k - half) - phase.toDouble() / phases
            val x = distance * cutoff
            val sinc = if (x == 0.0) 1.0 else sin(PI * x) / (PI * x)
            val window = if (kotlin.math.abs(distance) > half + 1) 0.0 else 0.5 + 0.5 * cos(PI * distance / (half + 1))
            (sinc * window * cutoff).toFloat()
        }
    }
    val out = FloatArray((pcm.size.toLong() * to / from).toInt())
    for (i in out.indices) {
        val position = i.toLong() * advance
        val base = (position / phases).toInt()
        val kernel = taps[(position % phases).toInt()]
        var sum = 0f
        for (k in maxOf(0, half - base)..minOf(2 * half, pcm.size - 1 - base + half)) sum += pcm[base - half + k] * kernel[k]
        out[i] = sum
    }
    return out
}
