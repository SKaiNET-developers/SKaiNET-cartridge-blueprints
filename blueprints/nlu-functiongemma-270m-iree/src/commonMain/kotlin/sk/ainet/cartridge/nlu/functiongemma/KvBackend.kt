package sk.ainet.cartridge.nlu.functiongemma

/**
 * The stateful KV-cache contract the engine decodes against (`functiongemma-kv-v1`): prefill a long, constant
 * prefix once and snapshot the cache; per utterance restore the snapshot, feed the new tokens in fixed-size
 * chunks, then step token by token. Every call returns the id of the next token the model predicts.
 *
 * Platform source sets bind this to a runtime: on Android to SKaiNET-transformers' `IreeKvSession`
 * (`libskainet_iree_kv.so`). Tests bind it to a script.
 */
public interface KvBackend : AutoCloseable {
    /** Opaque copy of the cache state. Owned by the backend that made it; closed by the engine. */
    public interface Snapshot : AutoCloseable

    /** Runs the prefill graph over [tokens] (padded to the graph's sequence length); [n] are real. */
    public fun prefill(tokens: IntArray, n: Int): Int

    /** Frees what only prefill needed; called once after [snapshot]. */
    public fun releasePrefill()

    public fun snapshot(): Snapshot

    public fun restore(snapshot: Snapshot)

    /** Feeds one chunk of [tokens] (padded to the chunk size); [n] are real. */
    public fun chunk(tokens: IntArray, n: Int): Int

    public fun step(token: Int): Int
}

/** Minimal tokenizer surface the engine needs; platform bindings adapt SKaiNET's tokenizers to it. */
public interface NluTokenizer {
    public fun encode(text: String): IntArray
    public fun decode(tokens: IntArray): String
}

/** Remembers the tokenized catalog prefix between runs — encoding a multi-kilobyte prefix is slow on small devices. */
public interface PrefixIdCache {
    public fun get(key: String): IntArray?
    public fun put(key: String, ids: IntArray)

    public object None : PrefixIdCache {
        override fun get(key: String): IntArray? = null
        override fun put(key: String, ids: IntArray) {}
    }
}
