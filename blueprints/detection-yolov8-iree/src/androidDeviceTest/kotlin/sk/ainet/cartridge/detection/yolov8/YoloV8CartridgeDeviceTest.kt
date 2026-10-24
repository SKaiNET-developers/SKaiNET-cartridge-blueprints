package sk.ainet.cartridge.detection.yolov8

import android.graphics.BitmapFactory
import android.os.Debug
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import sk.ainet.cartridge.profile.yolo.v1.YoloImage
import sk.ainet.cartridge.profile.yolo.v1.YoloResult
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/**
 * The E2 golden run: every materialized pack served by the host, opened through the public API
 * ([YoloV8Cartridge.open]) on this device, run on a real image — bus.jpg must give 4 person + 1 bus, the known
 * ONNX-Runtime-verified result. Measures the app-level latency (letterbox → head → NMS → source pixels), not the
 * bare head like the runtime's `IreeModuleRunnerDeviceTest`.
 *
 * ```
 * adb reverse tcp:8766 tcp:8766
 * (cd blueprints/detection-yolov8-iree/build/cartridge && python3 -m http.server 8766) &   # with packs.json + bus.jpg
 * ./gradlew :blueprints:detection-yolov8-iree:connectedAndroidTest
 * ```
 *
 * Served: `packs.json` (a JSON array of pack_dir names), each pack's `descriptor.json`, `manifest.json` and the
 * artifact files the manifest lists, and `bus.jpg`. Skips (does not fail) when no server is reachable.
 * Instrumentation arg `packServerUrl` overrides the base URL. Per pack it logs detections, create/first/p50/p95 ms
 * and PSS under tag `YoloCartridgeDevice`.
 */
@RunWith(AndroidJUnit4::class)
class YoloV8CartridgeDeviceTest {

    private companion object {
        const val TAG = "YoloCartridgeDevice"
        const val DEFAULT_URL = "http://localhost:8766"
        const val RUNS = 20
    }

    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val baseUrl = InstrumentationRegistry.getArguments().getString("packServerUrl") ?: DEFAULT_URL
    private val packsDir = File(instrumentation.targetContext.filesDir, "yolo-packs").apply { mkdirs() }

    private fun log(msg: String) {
        Log.i(TAG, msg)
        println(msg)
    }

    private fun fetch(name: String): File? {
        val dest = File(packsDir, name)
        return try {
            val conn = URL("$baseUrl/$name").openConnection() as HttpURLConnection
            conn.connectTimeout = 3_000
            conn.readTimeout = 60_000
            dest.parentFile?.mkdirs()
            conn.inputStream.use { input -> dest.outputStream().use { input.copyTo(it) } }
            dest
        } catch (e: Exception) {
            log("cannot fetch $baseUrl/$name: ${e.message}")
            null
        }
    }

    /** bus.jpg decoded to the profile's RGB888, row-major. */
    private fun image(file: File): YoloImage {
        val bitmap = BitmapFactory.decodeFile(file.path) ?: error("cannot decode ${file.name}")
        val argb = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(argb, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        val rgb = ByteArray(argb.size * 3)
        for (i in argb.indices) {
            val p = argb[i]
            rgb[i * 3] = (p shr 16 and 0xFF).toByte()
            rgb[i * 3 + 1] = (p shr 8 and 0xFF).toByte()
            rgb[i * 3 + 2] = (p and 0xFF).toByte()
        }
        return YoloImage(rgb, bitmap.width, bitmap.height)
    }

    @Test
    fun everyServedPackDetectsTheBus() {
        val index = fetch("packs.json")
        assumeTrue("no pack server at $baseUrl (see the class doc)", index != null)
        val packs = JSONArray(index!!.readText())
        assertTrue("packs.json is empty", packs.length() > 0)
        val bus = fetch("bus.jpg") ?: error("bus.jpg not served")
        val img = image(bus)
        val failures = mutableListOf<String>()
        for (i in 0 until packs.length()) runPack(packs.getString(i), img)?.let { failures += it }
        assertTrue("failed packs:\n" + failures.joinToString("\n"), failures.isEmpty())
    }

    /** Returns a failure line, or null when the pack passes. */
    private fun runPack(id: String, img: YoloImage): String? {
        // The pack, exactly as materialized: descriptor + manifest, then every artifact the manifest lists.
        val descriptor = fetch("$id/descriptor.json") ?: return "$id: descriptor.json not served"
        // The reference image's golden counts depend on the pack's resolution: at 640 ONNX Runtime
        // finds 4 person + 1 bus, at 320 the smallest person drops below the confidence threshold
        // (host parity at 320: 3 person + 1 bus, identical to ORT).
        val resolution = JSONObject(descriptor.readText()).getJSONObject("attributes").optInt("input_resolution", 640)
        val expected = mapOf("person" to if (resolution <= 320) 3 else 4, "bus" to 1)
        val manifest = fetch("$id/manifest.json") ?: return "$id: manifest.json not served"
        val artifacts = JSONObject(manifest.readText()).getJSONArray("artifacts")
        for (i in 0 until artifacts.length()) {
            val path = artifacts.getJSONObject(i).getString("path")
            if (path != "descriptor.json") fetch("$id/$path") ?: return "$id: $path not served"
        }

        val createStart = System.nanoTime()
        val engine = try {
            YoloV8Cartridge.open(PackDir(File(packsDir, id).absolutePath))
        } catch (e: Exception) {
            return "$id: open failed: ${e.message}"
        }
        val createMs = (System.nanoTime() - createStart) / 1e6
        engine.use { e ->
            val times = DoubleArray(RUNS)
            var last: YoloResult? = null
            for (k in 0 until RUNS) {
                val t0 = System.nanoTime()
                last = runBlocking { e.infer(img) }
                times[k] = (System.nanoTime() - t0) / 1e6
            }
            val result = last ?: return "$id: no result"
            if (result is YoloResult.Failed) return "$id: ${result.error.code}: ${result.error.message}"
            val boxes = (result as YoloResult.Detections).boxes
            val counts = boxes.groupingBy { it.label ?: "class-${it.classId}" }.eachCount()
            val inBounds = boxes.all { b ->
                b.box.x1 >= 0f && b.box.y1 >= 0f && b.box.x2 <= img.width && b.box.y2 <= img.height && b.box.x1 < b.box.x2 && b.box.y1 < b.box.y2
            }
            times.sort()
            fun pct(p: Double) = times[((times.size - 1) * p).toInt()]
            log(
                "$id: ${boxes.size} detections $counts scores=${boxes.map { "%.3f".format(it.score) }} " +
                    "create=%.0f ms first=%.1f ms p50=%.1f ms p95=%.1f ms pss=%d MB (runs=$RUNS)"
                        .format(createMs, times.first(), pct(0.5), pct(0.95), Debug.getPss() / 1024),
            )
            if (!inBounds) return "$id: a box lies outside the ${img.width}x${img.height} source image"
            return if (counts == expected) null
            else "$id: expected $expected, got $counts"
        }
    }
}
