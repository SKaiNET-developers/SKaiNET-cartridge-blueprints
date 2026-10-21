package sk.ainet.cartridge.generator.smollm2

import sk.ainet.cartridge.profiles.llm.LlmMessage
import sk.ainet.cartridge.profiles.llm.LlmRole

/**
 * The golden decode that defines this cartridge's behavioral identity: greedy (temperature 0)
 * is deterministic, so the fixed conversation below must produce exactly [GOLDEN_TOKENS] /
 * [GOLDEN_TEXT] on every compiled target — token-for-token, which is a stricter cross-backend
 * claim than the embedding cartridges' 2e-3 budget (an argmax has no tolerance; a flipped
 * near-tie IS a space change).
 *
 * Regenerate deliberately (jvmTest GoldenRegenerator, -PregenerateGoldens=true); changed
 * values mean a new checkpoint or a changed decode path, both of which are a new cartridge
 * version, not a test update.
 */
internal object GoldenCases {

    val MESSAGES: List<LlmMessage> = listOf(
        LlmMessage(LlmRole.USER, "What is the capital of France?"),
    )

    const val MAX_TOKENS: Int = 24

    /** Greedy transcript for [MESSAGES] (the 135M model's honest greedy answer, stop-token
     * terminated — small-model quality is exactly what hosts gate on, never hidden here). */
    const val GOLDEN_TEXT: String = "The capital of France"

    /** Generated token ids for [MESSAGES]. */
    val GOLDEN_TOKENS: List<Int> = listOf(504, 3575, 282, 4649)
}
