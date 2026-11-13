package sk.ainet.cartridge.classification.moonshine.head.recipe

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import sk.ainet.models.moonshine.MoonshineV2Checkpoint
import sk.ainet.models.moonshine.MoonshineV2HfWeightMap
import sk.ainet.models.moonshine.moonshineV2Encoder
import sk.ainet.models.moonshine.moonshineV2Frontend
import java.io.File

/**
 * The frozen encoder as the head sees it: 16 kHz mono PCM in, the encoder's rows out — computed the way a
 * streaming recognizer computes them, so the head is trained on the rows it will be fed.
 *
 * The audio is cut into windows of [CHUNK] feature frames that advance by [HOP]; each window gives the rows up
 * to [LOOKAHEAD] frames before its end, the last window gives all of its rows, and a row is taken once, from the
 * first window that finalizes it. A window that does not start the utterance carries [FRONTEND_CONTEXT] frames of
 * real audio as the frontend's left context; the first window starts at sample 0 with none. The rows are the
 * encoder's output *before* the recognizer's adapter (no positional table): `[rows, dim]` at 50 rows per second.
 *
 * Eager SKaiNET on the CPU, weights from the publisher's checkpoint (`config.json`, `model.safetensors`).
 */
internal class EncoderRows(modelDir: File) : AutoCloseable {
    private val checkpoint = MoonshineV2Checkpoint(modelDir)
    private val ctx = DirectCpuExecutionContext.create()
    private val frontend: Module<FP32, Float> = moonshineV2Frontend<FP32, Float>(FP32::class, checkpoint.config.dim)
        .also { checkpoint.bake(it, MoonshineV2HfWeightMap::frontend, FP32::class, ctx) }
    private val encoder: Module<FP32, Float> = moonshineV2Encoder<FP32, Float>(checkpoint.config, FP32::class)
        .also { checkpoint.bake(it, MoonshineV2HfWeightMap::encoder, FP32::class, ctx) }

    val dim: Int get() = checkpoint.config.dim

    /** The first [maxRows] rows of [pcm] (16 kHz, mono, [-1, 1]), row-major, and how many there are. */
    fun rows(pcm: FloatArray, maxRows: Int): Pair<FloatArray, Int> {
        val frames = pcm.size / FRAME_SAMPLES   // a trailing partial frame is dropped, as a stream would hold it back
        val out = FloatArray(maxRows * dim)
        var done = 0
        var window = 0
        while (done < frames && done < maxRows) {
            val start = window * HOP
            val count = minOf(CHUNK, frames - start)
            val last = start + CHUNK >= frames
            val context = if (start == 0) 0 else FRONTEND_CONTEXT
            val samples = pcm.copyOfRange((start - context) * FRAME_SAMPLES, (start + count) * FRAME_SAMPLES)
            var features = frontend.forward(ctx.fromFloatArray(Shape(1, samples.size), FP32::class, samples), ctx)
            if (context > 0) features = ctx.ops.narrow(features, 1, context, count)
            val encoded = encoder.forward(features, ctx)   // [1, count, dim]
            val end = if (last) start + count else start + CHUNK - LOOKAHEAD
            for (row in done until minOf(end, maxRows)) {
                for (d in 0 until dim) out[row * dim + d] = encoded.data.get(0, row - start, d)
            }
            done = minOf(end, maxRows)
            if (last) break
            window++
        }
        return out to done
    }

    override fun close() = checkpoint.close()

    companion object {
        const val SAMPLE_RATE = 16_000
        /** Samples per feature frame: 50 rows per second. */
        const val FRAME_SAMPLES = 320
        const val CHUNK = 64
        const val LOOKAHEAD = 4
        const val LEFT = 16
        const val HOP = CHUNK - LEFT - LOOKAHEAD   // 44
        const val FRONTEND_CONTEXT = 4
    }
}
