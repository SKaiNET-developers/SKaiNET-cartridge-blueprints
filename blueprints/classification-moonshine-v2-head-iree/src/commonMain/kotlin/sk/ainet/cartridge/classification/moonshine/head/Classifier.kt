package sk.ainet.cartridge.classification.moonshine.head

/**
 * Runs the cartridge's one compiled function: the head's input `[1, maxRows, dim + 1]` in, the logits
 * `[1, classes]` out. The platform binding implements it over the IREE runtime; tests implement it over a fixture.
 */
public interface HeadRunner : AutoCloseable {
    public fun run(input: FloatArray): FloatArray
}

/**
 * A materialized pack, ready to classify: encoder rows in, a `classification/v1` [Outcome] out. The rows come
 * from the encoder the pack names in [PackDir.features]; a recognizer that already runs that encoder hands over
 * what it has computed.
 */
public class Classifier(pack: PackDir, private val runner: HeadRunner) : AutoCloseable {
    private val features = pack.features
    private val decision = pack.decision()
    private val classes = pack.labels.labels.size

    /** The raw logits for [count] encoder rows ([rows] is row-major `[count, dim]`). */
    public fun logits(rows: FloatArray, count: Int): FloatArray {
        require(count > 0) { "no rows to classify" }
        val out = runner.run(features.input(rows, count))
        check(out.size == classes) { "the module returned ${out.size} logits for $classes labels" }
        return out
    }

    /** Accepted, undecided or rejected, under the pack's calibration. */
    public fun classify(rows: FloatArray, count: Int): Outcome = decision.decide(logits(rows, count))

    override fun close(): Unit = runner.close()
}
