package sk.ainet.cartridge.embedding.leaf

import sk.ainet.cartridge.profiles.embedding.EmbeddingDescriptor
import sk.ainet.cartridge.profiles.embedding.EmbeddingEngine
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.io.openRandomAccessSource
import sk.ainet.io.safetensors.SafeTensorsParametersLoader
import sk.ainet.lang.types.FP32
import sk.ainet.llm.api.Embedding
import sk.ainet.llm.api.EmbeddingModel
import sk.ainet.llm.api.EmbeddingRequest
import sk.ainet.llm.api.EmbeddingResponse
import sk.ainet.llm.api.Usage
import sk.ainet.models.bert.BertConfigParser
import sk.ainet.models.bert.BertEncoderRuntime
import sk.ainet.models.bert.BertExecutionMode
import sk.ainet.models.bert.BertNetworkLoader
import sk.ainet.models.bert.HuggingFaceTokenizer
import sk.ainet.models.bert.PooledExecutionContext
import sk.ainet.models.bert.createBertEncoderRuntime

/**
 * The LEAF embedding cartridge: MongoDB's `mdbr-leaf-ir` retrieval encoder running entirely on
 * SKaiNET-transformers' commonMain BERT path, eager mode — tokenize (WordPiece) → 6-layer BERT
 * (384 hidden) → mean pooling → sentence-transformers `2_Dense` projection 384→768 → L2
 * normalization. That is exactly the checkpoint's `modules.json` stack, and the path upstream
 * verifies against sentence-transformers.
 *
 * Being a real-weights cartridge, opening it needs the model snapshot on disk
 * (the blueprint's fetch step; ~92 MB). The descriptor stays intrinsic: dimensions, fingerprint
 * and prefixes are properties of the pinned checkpoint, asserted against the loaded network at
 * open time, never read from a file a host could drift from.
 *
 * LEAF is a *symmetric* retrieval model: no query/document instruction prefixes
 * (upstream's `EmbeddingModelProfiles` maps it to `EmbeddingPrefixes.NONE`).
 */
public object LeafEmbeddingCartridge {

    /** Hugging Face checkpoint this space is pinned to. */
    public const val MODEL_REPO: String = "MongoDB/mdbr-leaf-ir"
    public const val MODEL_REVISION: String = "4262131b32c3182bd06e67e92ae69d7bd66e0c5c"

    /**
     * The behavioral identity of the vector space: this checkpoint, this module stack. A new
     * revision with different weights is a new fingerprint and a full corpus re-embed.
     */
    public const val SPACE_FINGERPRINT: String = "mdbr-leaf-ir-v1"

    public val descriptor: EmbeddingDescriptor = EmbeddingDescriptor(
        spaceFingerprint = SPACE_FINGERPRINT,
        dimensions = 768,
        normalized = true,
        queryPrefix = "",
        documentPrefix = "",
    )

    /**
     * The default snapshot location: `LEAF_MODEL_DIR` when set, else the shared SKaiNET model
     * cache (`~/.cache/skainet/models/MongoDB_mdbr-leaf-ir`) that
     * `BertEmbeddingModel.fromHuggingFace` also populates on JVM.
     */
    public fun defaultModelDir(): String? {
        val explicit = envVar("LEAF_MODEL_DIR")
        if (!explicit.isNullOrBlank()) return explicit
        val home = homeDir() ?: return null
        return "$home/.cache/skainet/models/${MODEL_REPO.replace('/', '_')}"
    }

    /**
     * Load the snapshot and build the engine. Suspend because weight loading is real I/O
     * (~92 MB of f32 tensors).
     *
     * @param modelDir snapshot directory holding `config.json`, `model.safetensors`,
     *   `vocab.txt`, `1_Pooling/config.json` and `2_Dense/{config.json,model.safetensors}`;
     *   defaults to [defaultModelDir]
     * @param ctx execution context; the default pooled CPU context reuses scratch buffers
     *   across [EmbeddingEngine.embed] calls
     */
    public suspend fun open(
        modelDir: String? = null,
        ctx: ExecutionContext = PooledExecutionContext(DirectCpuExecutionContext()),
    ): LeafEmbeddingEngine {
        val dir = modelDir ?: defaultModelDir()
            ?: error("No model directory: set LEAF_MODEL_DIR or pass modelDir explicitly")

        val configJson = readTextFile("$dir/config.json")
            ?: error("config.json not found in $dir — run the blueprint's fetch step first")
        val denseConfigJson = readTextFile("$dir/2_Dense/config.json")
        val poolingJson = readTextFile("$dir/1_Pooling/config.json")
        val vocab = readTextFile("$dir/vocab.txt")
            ?: error("vocab.txt not found in $dir")

        val config = BertConfigParser.parse(configJson, denseConfigJson)
        val pooling = BertConfigParser.parsePooling(poolingJson)
        val tokenizer = HuggingFaceTokenizer.fromVocabTxt(vocab)

        val loaders = buildList {
            add(loaderFor("$dir/model.safetensors"))
            if (denseConfigJson != null) add(loaderFor("$dir/2_Dense/model.safetensors"))
        }
        val tensors = BertNetworkLoader.loadWeightTensors(loaders, ctx, FP32::class)
        val runtime = createBertEncoderRuntime<FP32>(config, tensors, ctx, pooling = pooling, mode = BertExecutionMode.DIRECT)

        check(runtime.dimensions == descriptor.dimensions) {
            "Loaded network produces ${runtime.dimensions}-d vectors but the $SPACE_FINGERPRINT " +
                "descriptor pins ${descriptor.dimensions} — wrong snapshot in $dir?"
        }
        return LeafEmbeddingEngine(runtime, tokenizer, ctx)
    }

    private fun loaderFor(path: String) = SafeTensorsParametersLoader(
        sourceProvider = {
            openRandomAccessSource(path) ?: error("cannot open $path")
        },
    )
}

/**
 * One engine, two contracts: the cartridge-side `embedding/v1` [EmbeddingEngine] and
 * SKaiNET-transformers' provider-neutral [EmbeddingModel] SPI. They were designed to converge —
 * `embed(text): FloatArray`, `dimensions`, `AutoCloseable` — so a single `embed` serves both,
 * and the SPI's `embedQuery`/`embedDocument` defaults are already correct for a symmetric
 * model like LEAF.
 *
 * **Threading:** not thread-safe (the pooled execution context reuses scratch buffers);
 * serialize calls or open one engine per worker.
 */
public class LeafEmbeddingEngine internal constructor(
    private val runtime: BertEncoderRuntime<FP32>,
    private val tokenizer: HuggingFaceTokenizer,
    private val ctx: ExecutionContext,
) : EmbeddingEngine, EmbeddingModel {

    override val dimensions: Int get() = runtime.dimensions

    override fun embed(text: String): FloatArray = runtime.encode(tokenizer.encode(text))

    override fun call(request: EmbeddingRequest): EmbeddingResponse {
        var promptTokens = 0
        val embeddings = request.inputs.mapIndexed { index, text ->
            val tokens = tokenizer.encode(text)
            promptTokens += tokens.size
            Embedding(index = index, vector = runtime.encode(tokens))
        }
        return EmbeddingResponse(
            embeddings = embeddings,
            usage = Usage(promptTokens = promptTokens, completionTokens = 0),
            modelId = LeafEmbeddingCartridge.MODEL_REPO,
        )
    }

    override fun close() {
        (ctx as? AutoCloseable)?.close()
    }
}
