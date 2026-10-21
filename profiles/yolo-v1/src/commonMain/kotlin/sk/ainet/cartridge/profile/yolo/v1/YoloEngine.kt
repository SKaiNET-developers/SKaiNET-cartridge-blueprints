package sk.ainet.cartridge.profile.yolo.v1

/**
 * The `yolo/v1` core contract (spec: docs/modules/ROOT/pages/task-profiles.adoc, ADR-008).
 * Implemented by cartridge adapters (over the C Runtime ABI or a language-native cartridge
 * API) and consumed by applications' closed adapters — never the other way around: this
 * artifact depends on no application and no cartridge.
 *
 * v1 is single-shot/batch only, mirroring the C ABI's `ctg_infer` exactly: one whole image
 * in, one terminal result out. Unlike `asr/v1` (which must serve both batch and streaming
 * engines from day one), no known cartridge behind this profile streams, so there is no
 * session/Flow layer to hide that capability behind — see ADR-008. A streaming/video variant,
 * if one is ever needed, is a new profile version, not a flag on this one.
 */
public interface YoloEngine : AutoCloseable {

    public val info: YoloEngineInfo

    /**
     * Run detection on one image. Suspends for the duration of one inference call.
     * Concurrent calls on one engine are an engine/cartridge property (descriptor
     * `requirements`), not a profile guarantee — same rule `asr/v1` states for sessions.
     */
    public suspend fun infer(image: YoloImage): YoloResult
}
