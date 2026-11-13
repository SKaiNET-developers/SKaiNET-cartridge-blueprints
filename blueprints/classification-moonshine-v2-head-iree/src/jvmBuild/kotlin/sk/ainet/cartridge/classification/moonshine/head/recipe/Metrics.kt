package sk.ainet.cartridge.classification.moonshine.head.recipe

import sk.ainet.cartridge.classification.moonshine.head.Decision
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt

/** The numbers the `calibrate` and `measure` steps compute. Plain arithmetic over logits, no tensors. */
internal object Metrics {

    /** Mean negative log-likelihood of [targets] under softmax([logits] / [t]). */
    fun nll(logits: List<FloatArray>, targets: IntArray, t: Float): Double {
        var sum = 0.0
        for (i in logits.indices) sum += Decision.logSumExp(logits[i], t, skip = -1) - logits[i][targets[i]] / t
        return sum / logits.size
    }

    /** The temperature in [0.05, 20] that minimizes [nll]: golden-section search over log T. */
    fun fitTemperature(logits: List<FloatArray>, targets: IntArray): Float {
        val ratio = (sqrt(5.0) - 1) / 2
        var lo = ln(0.05); var hi = ln(20.0)
        var a = hi - ratio * (hi - lo); var b = lo + ratio * (hi - lo)
        var fa = nll(logits, targets, exp(a).toFloat()); var fb = nll(logits, targets, exp(b).toFloat())
        repeat(60) {
            if (fa < fb) { hi = b; b = a; fb = fa; a = hi - ratio * (hi - lo); fa = nll(logits, targets, exp(a).toFloat()) }
            else { lo = a; a = b; fa = fb; b = lo + ratio * (hi - lo); fb = nll(logits, targets, exp(b).toFloat()) }
        }
        return exp((lo + hi) / 2).toFloat()
    }

    /**
     * The loosest accept threshold at which at most [targetRisk] of the accepted inputs are wrong.
     * [scores] lower is more confident; [acceptable] says whether an input could be accepted at all (its top class
     * is not the reject class); [correct] whether accepting it would be right. With no threshold that meets the
     * risk, nothing is accepted: the result is below every score.
     */
    fun acceptThreshold(scores: FloatArray, acceptable: BooleanArray, correct: BooleanArray, targetRisk: Float): Float {
        val order = scores.indices.filter { acceptable[it] }.sortedBy { scores[it] }
        if (order.isEmpty()) return (scores.minOrNull() ?: 0f) - 1f
        var wrong = 0
        var best = -1
        for ((k, i) in order.withIndex()) {
            if (!correct[i]) wrong++
            val tie = k + 1 < order.size && scores[order[k + 1]] == scores[i]
            if (!tie && wrong.toDouble() / (k + 1) <= targetRisk) best = k
        }
        return if (best < 0) scores[order.first()] - 1f else scores[order[best]]
    }

    /** The score at or below which [keep] of [scores] lie. */
    fun quantile(scores: FloatArray, keep: Float): Float {
        require(scores.isNotEmpty())
        val sorted = scores.sorted()
        return sorted[(kotlin.math.ceil(keep.toDouble() * sorted.size).toInt() - 1).coerceIn(0, sorted.size - 1)]
    }

    /** Area under the ROC curve for "in set" = low score: the chance a random in-set input scores below a random out-of-set one. */
    fun auroc(inSet: FloatArray, outOfSet: FloatArray): Double {
        if (inSet.isEmpty() || outOfSet.isEmpty()) return Double.NaN
        val out = outOfSet.sorted()
        var wins = 0.0
        for (s in inSet) {
            val below = lowerBound(out, s)          // out-of-set scores strictly below s
            val upTo = upperBound(out, s)           // ... at or below s
            wins += (out.size - upTo) + 0.5 * (upTo - below)
        }
        return wins / (inSet.size.toDouble() * out.size)
    }

    /** Share of out-of-set inputs that pass the threshold keeping [tpr] of the in-set inputs. */
    fun fprAtTpr(inSet: FloatArray, outOfSet: FloatArray, tpr: Float = 0.95f): Double {
        if (inSet.isEmpty() || outOfSet.isEmpty()) return Double.NaN
        val threshold = quantile(inSet, tpr)
        return outOfSet.count { it <= threshold }.toDouble() / outOfSet.size
    }

    /** Expected calibration error of the top probability under softmax(z / [t]), [bins] equal-width bins. */
    fun ece(logits: List<FloatArray>, targets: IntArray, t: Float, bins: Int = 15): Double {
        val n = IntArray(bins); val confidence = DoubleArray(bins); val hits = DoubleArray(bins)
        for (i in logits.indices) {
            val top = Decision.argmax(logits[i])
            val p = exp((logits[i][top] / t - Decision.logSumExp(logits[i], t, skip = -1)).toDouble())
            val bin = minOf(bins - 1, (p * bins).toInt())
            n[bin]++; confidence[bin] += p; if (top == targets[i]) hits[bin] += 1.0
        }
        var sum = 0.0
        for (b in 0 until bins) if (n[b] > 0) sum += abs(hits[b] - confidence[b])
        return sum / logits.size
    }

    /** Unweighted mean of the per-class F1 over the classes that occur as a target or a prediction. */
    fun macroF1(predicted: IntArray, targets: IntArray, classes: Int): Double {
        val tp = IntArray(classes); val fp = IntArray(classes); val fn = IntArray(classes)
        for (i in predicted.indices) {
            if (predicted[i] == targets[i]) tp[targets[i]]++ else { fp[predicted[i]]++; fn[targets[i]]++ }
        }
        val f1 = (0 until classes).filter { tp[it] + fp[it] + fn[it] > 0 }.map { 2.0 * tp[it] / (2 * tp[it] + fp[it] + fn[it]) }
        return if (f1.isEmpty()) Double.NaN else f1.average()
    }

    private fun lowerBound(sorted: List<Float>, v: Float): Int { var lo = 0; var hi = sorted.size; while (lo < hi) { val m = (lo + hi) / 2; if (sorted[m] < v) lo = m + 1 else hi = m }; return lo }
    private fun upperBound(sorted: List<Float>, v: Float): Int { var lo = 0; var hi = sorted.size; while (lo < hi) { val m = (lo + hi) / 2; if (sorted[m] <= v) lo = m + 1 else hi = m }; return lo }
}
