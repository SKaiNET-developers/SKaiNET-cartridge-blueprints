package sk.ainet.cartridge.classification.moonshine.head.recipe

import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.addJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import kotlinx.serialization.json.putJsonObject
import sk.ainet.cartridge.classification.moonshine.head.Calibration
import sk.ainet.cartridge.classification.moonshine.head.Decision
import sk.ainet.cartridge.classification.moonshine.head.Outcome

/**
 * Step `measure`: the `test` split, once, with the head and the calibration fixed.
 *
 *   --weights <head.safetensors> --calibration <calibration.json> --features <dir> --label-set <label-set.json>
 *   --manifest <dataset.json> --out <quality-report.json>
 *
 * `quality` in the report has the shape of a descriptor's `quality` (metric `top1`): copy it into the profile's
 * `measurements.quality`. The materializer never copies it for you.
 */
public fun main(args: Array<String>) {
    val a = Args(args)
    val labels = LabelSetInput.read(a.file("label-set"))
    val calibration = json.decodeFromString(Calibration.serializer(), a.file("calibration").readText())
    val manifestFile = a.file("manifest")
    val test = FeatureStore(a.file("features")).examples("test", labels)
    require(test.isNotEmpty()) { "the dataset has no `test` recordings to measure" }
    val logits = EagerHead.of(a.file("weights")).logits(test)
    val decision = Decision(labels.labelSet, calibration)
    val reject = labels.labelSet.rejectIndex
    val classes = labels.labels.size

    val known = test.indices.filter { test[it].target >= 0 }
    val predicted = IntArray(known.size) { Decision.argmax(logits[known[it]]) }
    val targets = IntArray(known.size) { test[known[it]].target }
    // The score as the decision uses it: an input whose top class is the reject class is refused whatever its
    // score, so it ranks last.
    val scores = FloatArray(test.size) { if (Decision.argmax(logits[it]) == reject) Float.POSITIVE_INFINITY else decision.score(logits[it]) }
    fun inSet(i: Int) = test[i].target >= 0 && test[i].target != reject
    val inScores = test.indices.filter(::inSet).map { scores[it] }.toFloatArray()
    val outScores = test.indices.filter { !inSet(it) }.map { scores[it] }.toFloatArray()
    val unseenScores = test.indices.filter { test[it].unseen }.map { scores[it] }.toFloatArray()

    val outcomes = test.indices.map { decision.decide(logits[it]) }
    val accepted = test.indices.filter { outcomes[it] is Outcome.Accepted }
    val wrong = accepted.count { (outcomes[it] as Outcome.Accepted).index != test[it].target }
    fun share(part: Int, of: Int): Double? = if (of == 0) null else part.toDouble() / of
    fun count(inSet: Boolean, kind: (Outcome) -> Boolean) = test.indices.count { inSet(it) == inSet && kind(outcomes[it]) }

    val report = buildJsonObject {
        putJsonObject("quality") {
            put("metric", "top1")
            put("value", predicted.indices.count { predicted[it] == targets[it] }.toDouble() / known.size)
            put("dataset", "${DatasetManifest.read(manifestFile).name ?: manifestFile.name}, test split")
            put("dataset_digest", sha256(manifestFile))
            put("harness", "MeasureCli (classification-moonshine-v2-head-iree), eager SKaiNET forward on the host")
            put("samples", known.size)
            put("macro_f1", Metrics.macroF1(predicted, targets, classes))
            put("ece", Metrics.ece(known.map { logits[it] }, targets, calibration.temperature))
            number("auroc", Metrics.auroc(inScores, outScores))
            number("fpr_at_95tpr", Metrics.fprAtTpr(inScores, outScores))
            number("auroc_unseen", Metrics.auroc(inScores, unseenScores))
            number("coverage", share(accepted.count(::inSet), inScores.size))
            number("risk", share(wrong, accepted.size))
            calibration.targetRisk?.let { put("target_risk", it) }
        }
        putJsonObject("outcomes") {
            putJsonObject("in_set") {
                put("total", inScores.size)
                put("accepted", count(true) { it is Outcome.Accepted })
                put("undecided", count(true) { it is Outcome.Undecided })
                put("rejected", count(true) { it is Outcome.Rejected })
            }
            putJsonObject("out_of_set") {
                put("total", outScores.size)
                put("unseen_classes", unseenScores.size)
                put("accepted", count(false) { it is Outcome.Accepted })
                put("undecided", count(false) { it is Outcome.Undecided })
                put("rejected", count(false) { it is Outcome.Rejected })
            }
            put("accepted_wrong", wrong)
        }
        putJsonArray("risk_coverage") {
            // What other target risks would have given: each threshold fitted on dev, as `calibrate` fits it,
            // then applied to test.
            val dev = FeatureStore(a.file("features")).examples("dev", labels)
            val devLogits = EagerHead.of(a.file("weights")).logits(dev)
            val devTop = IntArray(dev.size) { Decision.argmax(devLogits[it]) }
            val devScores = FloatArray(dev.size) { decision.score(devLogits[it]) }
            val top = IntArray(test.size) { Decision.argmax(logits[it]) }
            for (target in listOf(0.01f, 0.02f, 0.05f, 0.1f)) {
                val threshold = Metrics.acceptThreshold(
                    devScores,
                    acceptable = BooleanArray(dev.size) { devTop[it] != reject },
                    correct = BooleanArray(dev.size) { devTop[it] == dev[it].target },
                    targetRisk = target,
                )
                val taken = test.indices.filter { top[it] != reject && scores[it] <= threshold }
                addJsonObject {
                    put("target_risk", target)
                    put("accepted", taken.size)
                    number("coverage", share(taken.count(::inSet), inScores.size))
                    number("risk", share(taken.count { top[it] != test[it].target }, taken.size))
                }
            }
        }
        putJsonObject("definitions") {
            put("top1", "share of test recordings whose top class is their class; a recording outside the label set belongs to the reject class when the set has one, and is left out otherwise")
            put("auroc", "in-set vs out-of-set by the calibration's score, an input whose top class is the reject class ranking last; out-of-set = reject-class and held-out recordings")
            put("auroc_unseen", "the same against held-out classes only: classes the head never trained on")
            put("coverage", "share of in-set recordings that are accepted")
            put("risk", "share of accepted recordings that are wrong: a wrong label, or an out-of-set recording accepted")
            put("risk_coverage", "coverage and risk on test for other target risks, each threshold fitted on dev")
        }
    }
    a.file("out").apply { absoluteFile.parentFile.mkdirs() }.writeText(json.encodeToString(kotlinx.serialization.json.JsonObject.serializer(), report) + "\n")
    println("measure: " + report.getValue("quality"))
    println("measure: " + report.getValue("outcomes"))
}

private fun kotlinx.serialization.json.JsonObjectBuilder.number(name: String, value: Double?) {
    if (value != null && !value.isNaN()) put(name, JsonPrimitive(value))
}
