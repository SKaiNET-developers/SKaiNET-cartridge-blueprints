package sk.ainet.cartridge.classification.moonshine.head.recipe

import sk.ainet.cartridge.classification.moonshine.head.Calibration
import sk.ainet.cartridge.classification.moonshine.head.Decision
import sk.ainet.cartridge.classification.moonshine.head.Score
import sk.ainet.cartridge.classification.moonshine.head.Thresholds

/**
 * Step `calibrate` (ADR-018): the numbers that turn logits into accept / undecided / reject, fitted on `dev`.
 *
 *   --weights <head.safetensors> --features <dir> --label-set <label-set.json> --out <calibration.json> --seed <n>
 *   [--score energy|max_softmax] [--target-risk 0.01] [--reject-keep 0.95]
 *
 * 1. The temperature that minimizes the negative log-likelihood of the dev labels.
 * 2. `thresholds.accept`: the loosest score at which at most `target-risk` of the accepted dev inputs are wrong
 *    (a wrong label, or an out-of-set input accepted).
 * 3. `thresholds.reject`: the score that keeps `reject-keep` of the in-set dev inputs, never below `accept`.
 *
 * Deterministic: `--seed` is recorded, not used.
 */
public fun main(args: Array<String>) {
    val a = Args(args)
    a.long("seed", -1).also { require(it >= 0) { "missing --seed: a calibrate step states its seed (ADR-018)" } }
    val labels = LabelSetInput.read(a.file("label-set"))
    val score = when (val s = a.optional("score") ?: "energy") {
        "energy" -> Score.ENERGY
        "max_softmax" -> Score.MAX_SOFTMAX
        else -> error("--score must be energy or max_softmax, got '$s'")
    }
    val targetRisk = a.float("target-risk", 0.01f)
    val dev = FeatureStore(a.file("features")).examples("dev", labels)
    require(dev.isNotEmpty()) { "the dataset has no `dev` recordings to calibrate on" }
    val logits = EagerHead.of(a.file("weights")).logits(dev)

    val known = dev.indices.filter { dev[it].target >= 0 }
    val temperature = Metrics.fitTemperature(known.map { logits[it] }, IntArray(known.size) { dev[known[it]].target })

    val reject = labels.labelSet.rejectIndex
    val provisional = Decision(labels.labelSet, Calibration(temperature, score, Thresholds(0f, 0f)))
    val scores = FloatArray(dev.size) { provisional.score(logits[it]) }
    val top = IntArray(dev.size) { Decision.argmax(logits[it]) }
    val accept = Metrics.acceptThreshold(
        scores,
        acceptable = BooleanArray(dev.size) { top[it] != reject },
        correct = BooleanArray(dev.size) { top[it] == dev[it].target },
        targetRisk = targetRisk,
    )
    val inSet = dev.indices.filter { dev[it].target >= 0 && dev[it].target != reject }
    require(inSet.isNotEmpty()) { "no in-set `dev` recordings to calibrate on" }
    val keep = Metrics.quantile(FloatArray(inSet.size) { scores[inSet[it]] }, a.float("reject-keep", 0.95f))

    val calibration = Calibration(temperature, score, Thresholds(accept, maxOf(accept, keep)), targetRisk, "dev (${dev.size} recordings)")
    a.file("out").apply { absoluteFile.parentFile.mkdirs() }.writeText(json.encodeToString(Calibration.serializer(), calibration) + "\n")
    val accepted = dev.indices.count { top[it] != reject && scores[it] <= accept }
    println("calibrate: temperature ${"%.3f".format(temperature)}, accept <= ${"%.3f".format(accept)} ($accepted of ${dev.size} dev inputs), reject > ${"%.3f".format(calibration.thresholds.reject)}")
    if (accepted == 0) println("calibrate: WARNING no threshold meets a risk of $targetRisk on dev: nothing will be accepted")
}
