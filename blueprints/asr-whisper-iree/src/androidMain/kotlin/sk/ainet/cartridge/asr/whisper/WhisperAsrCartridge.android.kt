package sk.ainet.cartridge.asr.whisper

import android.content.Context
import kotlinx.io.files.Path
import sk.ainet.audio.mel.MelConfig
import sk.ainet.audio.mel.MelFilterbanks
import sk.ainet.audio.mel.MelSpectrogram
import sk.ainet.transformers.iree.android.IreeWhisperSession
import java.io.File

/**
 * The Android binding: [AsrCartridge] over SKaiNET-transformers' released Whisper runtime (`IreeWhisperSession`,
 * `libskainet_whisper_session.so`) and the model files of one language flavor of a materialized [PackDir].
 *
 * Per utterance: Whisper's 80-bin log-mel of the 16 kHz audio (SKaiNET-audio, padded or cut to the window), the
 * encoder on [encoderDevice], prefill and greedy steps on [decoderDevice], then `tokenizer.json`. The pack compiles
 * prefill and step for the CPU on every target, so the decoder device is the CPU; the encoder runs on the GPU when the
 * pack was built for one.
 *
 * Calls are synchronized. The session is created lazily on the first [transcribe] and lives until [close] — creating
 * it loads the three graphs and binds the parameter archive.
 */
public class WhisperAsrCartridge(
    public val pack: PackDir,
    /** BCP-47 tag; the flavor is chosen with [Language.select] against what the pack holds. */
    languageTag: String = "en",
    /** Where the encoder runs. Defaults to what the pack was built for. */
    private val encoderDevice: String = if (pack.targetsAccelerator) IreeWhisperSession.VULKAN_DEVICE else IreeWhisperSession.CPU_DEVICE,
    /** Where prefill and step run: the CPU, which the pack compiles them for. */
    private val decoderDevice: String = IreeWhisperSession.CPU_DEVICE,
    /** Repetition guard of the greedy decode (see `IreeWhisperSession.decode`); 0 disables it. */
    private val repeatNgram: Int = IreeWhisperSession.DEFAULT_REPEAT_NGRAM,
) : AsrCartridge {

    override val id: String = pack.id
    override val language: String = Language.select(languageTag, pack.languages)
        ?: throw IllegalArgumentException("pack ${pack.id} holds languages ${pack.languages}, none serves '$languageTag'")

    private val files = pack.model(language)
    private val contract = IreeWhisperSession.Contract.fromManifest(PackDir.readText(files.contractJson))
    private val tokenizer by lazy { WhisperTokenizer.fromFile(files.tokenizerJson) }
    private val mel by lazy {
        MelSpectrogram(
            MelFilterbanks.slaney(contract.nMels, N_FFT, sampleRateHz),
            MelConfig.whisper(maxFrames = contract.melFrames),
        )
    }
    private var session: IreeWhisperSession? = null

    /** `melFrames` hops of 10 ms: 4 s for the window this blueprint exports. */
    override val maxUtteranceSeconds: Float = contract.melFrames * HOP / sampleRateHz.toFloat()

    @Synchronized
    override fun transcribe(pcm: FloatArray): String {
        val s = session ?: IreeWhisperSession(
            encoderDevice,
            IreeWhisperSession.Files(files.encoderVmfb.toString(), files.prefillVmfb.toString(), files.stepVmfb.toString(), files.paramsIrpa.toString()),
            contract,
            decoderDevice,
        ).also { session = it }
        // pad or cut the AUDIO to the window, as Whisper's reference does: zero-padded mel columns would differ from
        // the log-mel of silence the model was trained on
        val features = mel.logMel(pcm.copyOf(maxUtteranceSamples())).data
        val text = tokenizer.decode(s.decode(features, repeatNgram = repeatNgram)).trim()
        return if (s.stats()["stoppedOnRepeat"] == "1") Transcript.dropRepeat(text) else text
    }

    /** What the last [transcribe] cost, by stage (`IreeWhisperSession.stats`). */
    @Synchronized
    public fun stats(): Map<String, String> = session?.stats().orEmpty()

    @Synchronized
    override fun close() {
        session?.close(); session = null
    }

    private fun maxUtteranceSamples(): Int = contract.melFrames * HOP

    public companion object {
        private const val N_FFT = 400
        private const val HOP = 160

        /**
         * Copies a pack shipped under `assets/<assetPath>/` into [into] (once; existing non-empty files are kept) and
         * returns it as a [PackDir]. The IREE runtime opens files by path, so assets must be on the file system.
         */
        public fun packFromAssets(context: Context, assetPath: String, into: File = File(context.filesDir, assetPath)): PackDir {
            val assets = context.assets
            fun copy(rel: String) {
                val children = assets.list(rel).orEmpty()
                if (children.isEmpty()) {
                    val out = File(into, rel.removePrefix(assetPath).trimStart('/'))
                    if (out.exists() && out.length() > 0) return
                    out.parentFile?.mkdirs()
                    assets.open(rel).use { i -> out.outputStream().use { i.copyTo(it) } }
                } else children.forEach { copy("$rel/$it") }
            }
            copy(assetPath)
            return PackDir(Path(into.absolutePath))
        }
    }
}
