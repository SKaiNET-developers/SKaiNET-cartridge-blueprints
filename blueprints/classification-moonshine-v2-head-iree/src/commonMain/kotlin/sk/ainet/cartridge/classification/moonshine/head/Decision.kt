package sk.ainet.cartridge.classification.moonshine.head

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlin.math.exp
import kotlin.math.ln

/**
 * `artifacts/other/label-set.json`: the classes, in output order. [rejectLabel], when set, is the class trained on
 * out-of-set examples; it is one of [labels] and is never an accepted result.
 */
@Serializable
public data class LabelSet(
    val labels: List<String>,
    @SerialName("reject_label") val rejectLabel: String? = null,
) {
    init {
        require(labels.size >= 2) { "a label set needs at least two labels" }
        require(labels.toSet().size == labels.size) { "labels must be unique" }
        require(rejectLabel == null || rejectLabel in labels) { "reject_label '$rejectLabel' is not one of the labels" }
    }

    /** Index of the reject class, or -1. */
    val rejectIndex: Int get() = rejectLabel?.let(labels::indexOf) ?: -1
}

/** `score` of a calibration: how the logits become one number where lower means "more surely in the set". */
@Serializable
public enum class Score {
    /** E = −T · log Σ exp(z_i / T) over the classes other than the reject class. */
    @SerialName("energy") ENERGY,

    /** 1 − max softmax(z / T) over all classes. */
    @SerialName("max_softmax") MAX_SOFTMAX,
}

@Serializable
public data class Thresholds(val accept: Float, val reject: Float) {
    init { require(accept <= reject) { "thresholds.accept ($accept) must not exceed thresholds.reject ($reject)" } }
}

/**
 * `artifacts/calibration/calibration.json`: the fitted numbers that turn logits into a decision. [fittedOn] and
 * [targetRisk] say what "accepted" was calibrated to mean: on that data, at most [targetRisk] of the accepted
 * inputs were wrong — a wrong label, or an out-of-set input.
 */
@Serializable
public data class Calibration(
    val temperature: Float,
    val score: Score = Score.ENERGY,
    val thresholds: Thresholds,
    @SerialName("target_risk") val targetRisk: Float? = null,
    @SerialName("fitted_on") val fittedOn: String? = null,
) {
    init { require(temperature > 0f) { "temperature must be positive, got $temperature" } }
}

/** What the logits of one input say, under a label set and a calibration. */
public sealed interface Outcome {
    /** The top label, with a score at or below `thresholds.accept`: act on it. */
    public data class Accepted(val label: String, val index: Int, val score: Float) : Outcome

    /** Between the thresholds: neither sure enough to act nor sure enough to refuse. Use another path. */
    public data class Undecided(val label: String, val index: Int, val score: Float) : Outcome

    /** Out of set: the score is above `thresholds.reject`, or the top class is the reject class. */
    public data class Rejected(val score: Float) : Outcome
}

/** The `classification/v1` decision: host-side arithmetic over the raw logits, the same for every backend. */
public class Decision(private val labels: LabelSet, private val calibration: Calibration) {

    /** Lower is more confident. See [Score]. */
    public fun score(logits: FloatArray): Float {
        require(logits.size == labels.labels.size) { "${logits.size} logits for ${labels.labels.size} labels" }
        val t = calibration.temperature
        return when (calibration.score) {
            Score.ENERGY -> -t * logSumExp(logits, t, skip = labels.rejectIndex)
            Score.MAX_SOFTMAX -> 1f - exp(logits.max() / t - logSumExp(logits, t, skip = -1))
        }
    }

    public fun decide(logits: FloatArray): Outcome {
        val s = score(logits)
        val top = argmax(logits)
        return when {
            top == labels.rejectIndex || s > calibration.thresholds.reject -> Outcome.Rejected(s)
            s <= calibration.thresholds.accept -> Outcome.Accepted(labels.labels[top], top, s)
            else -> Outcome.Undecided(labels.labels[top], top, s)
        }
    }

    public companion object {
        public fun argmax(values: FloatArray): Int {
            var best = 0
            for (i in 1 until values.size) if (values[i] > values[best]) best = i
            return best
        }

        /** log Σ exp(z_i / t), max-shifted so large logits do not overflow; [skip] leaves one class out. */
        public fun logSumExp(logits: FloatArray, t: Float, skip: Int): Float {
            var max = Float.NEGATIVE_INFINITY
            for (i in logits.indices) if (i != skip && logits[i] > max) max = logits[i]
            var sum = 0.0
            for (i in logits.indices) if (i != skip) sum += exp(((logits[i] - max) / t).toDouble())
            return max / t + ln(sum).toFloat()
        }
    }
}
