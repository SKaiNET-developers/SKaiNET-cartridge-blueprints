package sk.ainet.cartridge.demo.yolo

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.net.Uri
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import kotlinx.coroutines.runBlocking
import sk.ainet.cartridge.detection.yolov8.PackDir
import sk.ainet.cartridge.detection.yolov8.YoloV8Cartridge
import sk.ainet.cartridge.profile.yolo.v1.Detection
import sk.ainet.cartridge.profile.yolo.v1.YoloEngine
import sk.ainet.cartridge.profile.yolo.v1.YoloImage
import sk.ainet.cartridge.profile.yolo.v1.YoloResult
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Live `yolo/v1` on this phone's camera: the analysis stream is drawn as the preview, the engine
 * runs on whatever pack you select, and the newest detections ride on top — name and probability
 * per box. Packs live under this app's external-files dir (`files/packs/<pack_dir>/`), pushed once
 * over adb — the app ships no weights (see the module's build script header). Every pack found
 * gets a button; the pack decides its device (`vulkan-arm64` → Vulkan, `cpu-arm64` → CPU), so
 * switching buttons is the live CPU/GPU comparison.
 *
 * There is no CameraX `Preview` use case on purpose: frames shown and frames analyzed are the same
 * bitmaps, so the overlay maps 1:1 with none of the crop/rotation mismatch a separate preview
 * surface invites. *Pick image* pauses the live feed and runs the engine once on a gallery photo;
 * *Live* resumes. All engine work runs on one worker thread — [YoloEngine] is single-flight.
 */
class MainActivity : ComponentActivity() {

    private companion object {
        const val PICK_IMAGE = 1
        const val REQ_CAMERA = 2
        /** Decode bound for picked photos: detectors see 640² after letterboxing. */
        const val MAX_DECODE_EDGE = 2048
    }

    private val worker = Executors.newSingleThreadExecutor()
    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val detecting = AtomicBoolean(false)

    private var engine: YoloEngine? = null
    private var packName: String? = null
    private var stillBitmap: Bitmap? = null

    /** True while the camera feed drives the preview; false in picked-image mode. */
    @Volatile private var live = true
    private var cameraRunning = false

    private lateinit var status: TextView
    private lateinit var packRow: LinearLayout
    private lateinit var preview: OverlayView
    private lateinit var liveButton: Button
    private lateinit var runAgain: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val dp = resources.displayMetrics.density

        status = TextView(this).apply {
            textSize = 13f
            setPadding(0, 0, 0, (6 * dp).toInt())
        }
        packRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        liveButton = Button(this).apply {
            text = "Live"
            isEnabled = false // live is the initial mode
            setOnClickListener {
                live = true
                isEnabled = false
                runAgain.isEnabled = false
                status.text = statusPrefix()
            }
        }
        val pick = Button(this).apply {
            text = "Pick image"
            setOnClickListener {
                startActivityForResult(
                    Intent(Intent.ACTION_GET_CONTENT).setType("image/*"),
                    PICK_IMAGE,
                )
            }
        }
        runAgain = Button(this).apply {
            text = "Run again"
            isEnabled = false
            setOnClickListener { inferStill() }
        }
        val buttonRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            addView(liveButton)
            addView(pick)
            addView(runAgain)
        }
        preview = OverlayView(this)
        val banner = TextView(this).apply {
            text = "⚡ powered by SKaiNET"
            gravity = Gravity.CENTER
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.rgb(16, 18, 28))
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            textSize = 14f
            letterSpacing = 0.06f
            setPadding(0, (10 * dp).toInt(), 0, (10 * dp).toInt())
        }

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            val pad = (12 * dp).toInt()
            setPadding(pad, (24 * dp).toInt(), pad, 0)
            addView(status)
            addView(packRow)
            addView(buttonRow)
            addView(preview, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))
            addView(banner, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT))
        }
        setContentView(root)
        status.text = "Loading packs…"
        worker.execute {
            // Packs bundled into the APK (assets/packs/<pack_dir>/…) unpack into files/packs once;
            // packs loaded over adb land there too. The IREE runtime opens files by path, so assets
            // must reach the file system first.
            try {
                for (name in assets.list("packs").orEmpty()) {
                    YoloV8Cartridge.packFromAssets(this, "packs/$name")
                }
            } catch (_: Exception) {
                // no bundled packs — adb-loaded ones may still exist
            }
            runOnUiThread { loadPackButtons() }
        }

        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), REQ_CAMERA)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQ_CAMERA) {
            if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) startCamera()
            else status.text = "No camera permission — live preview off; Pick image still works."
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        worker.execute { engine?.close() }
        worker.shutdown()
        cameraExecutor.shutdown()
    }

    // ---- camera ----

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(cameraExecutor) { proxy -> onFrame(proxy) }
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
            cameraRunning = true
        }, mainExecutor)
    }

    /** Camera thread: every frame becomes the preview; the newest idle moment becomes a detection. */
    private fun onFrame(proxy: ImageProxy) {
        if (!live) {
            proxy.close()
            return
        }
        val rotation = proxy.imageInfo.rotationDegrees
        val bitmap = proxy.toBitmap()
        proxy.close()
        val upright = if (rotation == 0) bitmap else {
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, Matrix().apply { postRotate(rotation.toFloat()) }, true)
        }
        runOnUiThread { if (live) preview.setFrame(upright) }

        val e = engine ?: return
        if (!detecting.compareAndSet(false, true)) return
        worker.execute {
            try {
                if (e !== engine || !live) return@execute // pack switched or mode left meanwhile
                val img = toYoloImage(upright)
                val t0 = System.nanoTime()
                val result = runBlocking { e.infer(img) }
                val ms = (System.nanoTime() - t0) / 1e6
                runOnUiThread {
                    if (!live) return@runOnUiThread
                    when (result) {
                        is YoloResult.Failed -> status.text = "${result.error.code}: ${result.error.message}"
                        is YoloResult.Detections -> {
                            preview.setBoxes(result.boxes)
                            status.text = "%s · %.0f ms · %s".format(statusPrefix(), ms, countsOf(result.boxes))
                        }
                    }
                }
            } catch (ex: Exception) {
                runOnUiThread { status.text = "infer failed: ${ex.message}" }
            } finally {
                detecting.set(false)
            }
        }
    }

    // ---- packs ----

    /** Internal first (survives scoped storage: shell-created dirs under the external root are
     *  invisible to the app, so `adb push` + `run-as cp` lands them internally — see the README),
     *  external second for anything the app itself produced. */
    private fun packRoots() = listOf(File(filesDir, "packs"), File(getExternalFilesDir(null), "packs"))

    private fun loadPackButtons() {
        val packs = packRoots()
            .flatMap { root -> root.listFiles { f: File -> File(f, "descriptor.json").isFile }?.toList().orEmpty() }
            .distinctBy { it.name }
            .sortedBy { it.name }
        packRow.removeAllViews()
        if (packs.isEmpty()) {
            status.text = "No packs. Load pack_dirs into\n${packRoots().first()}\n(adb push to /data/local/tmp, then run-as cp — see README) and reopen."
            return
        }
        status.text = "Select a pack."
        for (dir in packs) {
            packRow.addView(
                Button(this).apply {
                    // detection-yolov8-iree-vulkan-arm64 → "vulkan-arm64"
                    text = dir.name.removePrefix("detection-yolov8-iree-")
                    setOnClickListener { openPack(dir, this) }
                },
            )
        }
    }

    private fun statusPrefix(): String = packName?.removePrefix("detection-yolov8-iree-") ?: "no pack"

    private fun countsOf(boxes: List<Detection>): String =
        if (boxes.isEmpty()) "nothing detected"
        else boxes.groupingBy { it.label ?: "class-${it.classId}" }.eachCount()
            .entries.joinToString { "${it.value} ${it.key}" }

    private fun setPackButtonsEnabled(enabled: Boolean) {
        for (i in 0 until packRow.childCount) packRow.getChildAt(i).isEnabled = enabled
    }

    private fun openPack(dir: File, button: Button) {
        status.text = "Opening ${dir.name}…"
        setPackButtonsEnabled(false)
        worker.execute {
            try {
                engine?.close()
                engine = null
                val t0 = System.nanoTime()
                val opened = YoloV8Cartridge.open(PackDir(dir.absolutePath))
                val createMs = (System.nanoTime() - t0) / 1e6
                engine = opened
                packName = dir.name
                runOnUiThread {
                    setPackButtonsEnabled(true)
                    button.isEnabled = false // the selected pack
                    status.text = "%s · %s v%s · open %.0f ms".format(
                        statusPrefix(), opened.info.name, opened.info.version, createMs,
                    )
                    if (!live && stillBitmap != null) inferStill()
                }
            } catch (e: Exception) {
                runOnUiThread {
                    setPackButtonsEnabled(true)
                    status.text = "open failed: ${e.message}"
                }
            }
        }
    }

    // ---- picked-image mode ----

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != PICK_IMAGE || resultCode != RESULT_OK) return
        val uri = data?.data ?: return
        live = false
        liveButton.isEnabled = cameraRunning
        worker.execute {
            val decoded = decode(uri)
            runOnUiThread {
                if (decoded == null) {
                    status.text = "cannot decode that image"
                } else {
                    stillBitmap = decoded
                    preview.set(decoded, emptyList())
                    runAgain.isEnabled = true
                    if (engine != null) inferStill() else status.text = "Image ready — select a pack."
                }
            }
        }
    }

    /** Bounded decode: big camera photos add nothing past the letterbox, only conversion time. */
    private fun decode(uri: Uri): Bitmap? = try {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
        var sample = 1
        while (maxOf(bounds.outWidth, bounds.outHeight) / (sample * 2) >= MAX_DECODE_EDGE / 2) sample *= 2
        val opts = BitmapFactory.Options().apply { inSampleSize = sample }
        contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) }
    } catch (e: Exception) {
        null
    }

    private fun inferStill() {
        val bmp = stillBitmap ?: return
        val e = engine ?: return
        status.text = "Running ${statusPrefix()}…"
        runAgain.isEnabled = false
        worker.execute {
            try {
                val img = toYoloImage(bmp)
                val t0 = System.nanoTime()
                val result = runBlocking { e.infer(img) }
                val ms = (System.nanoTime() - t0) / 1e6
                runOnUiThread {
                    runAgain.isEnabled = true
                    when (result) {
                        is YoloResult.Failed -> status.text = "${result.error.code}: ${result.error.message}"
                        is YoloResult.Detections -> {
                            preview.set(bmp, result.boxes)
                            status.text = "%s · %.0f ms · %d detections: %s".format(
                                statusPrefix(), ms, result.boxes.size, countsOf(result.boxes),
                            )
                        }
                    }
                }
            } catch (ex: Exception) {
                runOnUiThread {
                    runAgain.isEnabled = true
                    status.text = "infer failed: ${ex.message}"
                }
            }
        }
    }

    private fun toYoloImage(bmp: Bitmap): YoloImage {
        val argb = IntArray(bmp.width * bmp.height)
        bmp.getPixels(argb, 0, bmp.width, 0, 0, bmp.width, bmp.height)
        val rgb = ByteArray(argb.size * 3)
        for (i in argb.indices) {
            val p = argb[i]
            rgb[i * 3] = (p shr 16 and 0xFF).toByte()
            rgb[i * 3 + 1] = (p shr 8 and 0xFF).toByte()
            rgb[i * 3 + 2] = (p and 0xFF).toByte()
        }
        return YoloImage(rgb, bmp.width, bmp.height)
    }
}

/**
 * The current frame (live) or picked image, fit-centered, with detections drawn in source-pixel
 * coordinates mapped by the same fit — name and probability on every box.
 */
private class OverlayView(activity: Activity) : View(activity) {

    private var bitmap: Bitmap? = null
    private var boxes: List<Detection> = emptyList()

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2.5f * resources.displayMetrics.density
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 13f * resources.displayMetrics.scaledDensity
    }
    private val textBg = Paint()
    private val dst = RectF()

    // Distinct, stable per-class colors (golden-angle hues).
    private fun classColor(classId: Int): Int =
        Color.HSVToColor(floatArrayOf((classId * 137.5f) % 360f, 0.8f, 0.95f))

    /** Live path: new frame, keep the latest boxes riding on top. */
    fun setFrame(bitmap: Bitmap) {
        this.bitmap = bitmap
        invalidate()
    }

    /** Live path: newest detections for the stream. */
    fun setBoxes(boxes: List<Detection>) {
        this.boxes = boxes
        invalidate()
    }

    /** Still path: image and its detections together. */
    fun set(bitmap: Bitmap, boxes: List<Detection>) {
        this.bitmap = bitmap
        this.boxes = boxes
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        val bmp = bitmap ?: return
        val scale = minOf(width.toFloat() / bmp.width, height.toFloat() / bmp.height)
        val ox = (width - bmp.width * scale) / 2f
        val oy = (height - bmp.height * scale) / 2f
        dst.set(ox, oy, ox + bmp.width * scale, oy + bmp.height * scale)
        canvas.drawBitmap(bmp, null, dst, null)

        for (d in boxes) {
            val color = classColor(d.classId)
            boxPaint.color = color
            canvas.drawRect(
                ox + d.box.x1 * scale, oy + d.box.y1 * scale,
                ox + d.box.x2 * scale, oy + d.box.y2 * scale,
                boxPaint,
            )
            val label = "%s %.2f".format(d.label ?: "class-${d.classId}", d.score)
            val tw = textPaint.measureText(label)
            val th = textPaint.fontMetrics.let { it.descent - it.ascent }
            val tx = (ox + d.box.x1 * scale).coerceIn(0f, width - tw)
            val ty = (oy + d.box.y1 * scale - 4f).coerceAtLeast(th)
            textBg.color = color
            canvas.drawRect(tx - 2f, ty - th, tx + tw + 2f, ty + textPaint.fontMetrics.descent, textBg)
            canvas.drawText(label, tx, ty, textPaint)
        }
    }
}
