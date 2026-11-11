package sk.ainet.cartridge.runtime.lfm2.torq.iree

import cnames.structs.skainet_iree_runtime
import cnames.structs.skainet_iree_session
import cnames.structs.skainet_iree_view
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.CPointerVar
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.LongVar
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.allocArrayOf
import kotlinx.cinterop.convert
import kotlinx.cinterop.cstr
import kotlinx.cinterop.get
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.set
import kotlinx.cinterop.toKString
import kotlinx.cinterop.usePinned
import sk.ainet.cartridge.runtime.lfm2.torq.native.SKAINET_IREE_BF16
import sk.ainet.cartridge.runtime.lfm2.torq.native.SKAINET_IREE_F32
import sk.ainet.cartridge.runtime.lfm2.torq.native.skainet_iree_call
import sk.ainet.cartridge.runtime.lfm2.torq.native.skainet_iree_last_error
import sk.ainet.cartridge.runtime.lfm2.torq.native.skainet_iree_runtime_create
import sk.ainet.cartridge.runtime.lfm2.torq.native.skainet_iree_runtime_release
import sk.ainet.cartridge.runtime.lfm2.torq.native.skainet_iree_session_create
import sk.ainet.cartridge.runtime.lfm2.torq.native.skainet_iree_session_release
import sk.ainet.cartridge.runtime.lfm2.torq.native.skainet_iree_view_bytes
import sk.ainet.cartridge.runtime.lfm2.torq.native.skainet_iree_view_dim
import sk.ainet.cartridge.runtime.lfm2.torq.native.skainet_iree_view_element_type
import sk.ainet.cartridge.runtime.lfm2.torq.native.skainet_iree_view_from_host
import sk.ainet.cartridge.runtime.lfm2.torq.native.skainet_iree_view_rank
import sk.ainet.cartridge.runtime.lfm2.torq.native.skainet_iree_view_release
import sk.ainet.cartridge.runtime.lfm2.torq.native.skainet_iree_view_to_host

// Derived from the generic-session part of SKaiNET-embedded's Kotlin IREE wrapper (MIT), with the bf16 and f32 host
// tensors only; the C shim it calls is native/iree-session (native/upstream.json).

/** A failure reported by the native runtime; [message] is the formatted IREE status. */
public class IreeException(message: String) : IllegalStateException(message)

@OptIn(ExperimentalForeignApi::class)
private fun failure(what: String): Nothing = throw IreeException("$what: ${skainet_iree_last_error()?.toKString().orEmpty()}")

@OptIn(ExperimentalForeignApi::class)
public enum class ElementType(public val code: Int, public val bytes: Int) {
    F32(SKAINET_IREE_F32.toInt(), 4), BF16(SKAINET_IREE_BF16.toInt(), 2);

    public companion object {
        public fun of(code: Int): ElementType = entries.firstOrNull { it.code == code } ?: throw IreeException("unsupported element type 0x${code.toString(16)}")
    }
}

/** A dense row-major tensor in host memory: what goes into and comes out of a [IreeSession.call]. */
public class HostTensor(public val type: ElementType, public val shape: LongArray, public val bytes: ByteArray) {
    init {
        val count = shape.fold(1L) { a, d -> a * d }
        require(bytes.size.toLong() == count * type.bytes) { "tensor ${shape.toList()} ${type.name} needs ${count * type.bytes} bytes, got ${bytes.size}" }
    }

    public val elementCount: Int get() = bytes.size / type.bytes

    public fun toFloats(): FloatArray = when (type) {
        ElementType.F32 -> FloatArray(elementCount) { Float.fromBits(uint16(it * 4) or (uint16(it * 4 + 2) shl 16)) }
        ElementType.BF16 -> FloatArray(elementCount) { Float.fromBits(uint16(it * 2) shl 16) }
    }

    private fun uint16(o: Int): Int = (bytes[o].toInt() and 0xff) or ((bytes[o + 1].toInt() and 0xff) shl 8)

    public companion object {
        /** f32 values rounded to bf16 (round to nearest even). */
        public fun bf16(shape: LongArray, values: FloatArray): HostTensor =
            HostTensor(ElementType.BF16, shape, ByteArray(values.size * 2).also { b ->
                values.forEachIndexed { i, v ->
                    val u = v.toRawBits()
                    val h = (u + 0x7FFF + ((u ushr 16) and 1)) ushr 16
                    b[i * 2] = h.toByte(); b[i * 2 + 1] = (h ushr 8).toByte()
                }
            })
    }
}

/**
 * One IREE instance for the process. Devices ("local-task" = CPU, "torq" = NPU) are created on first use and shared
 * by every session on them. [flags] are IREE runtime flags, parsed once per process.
 */
@OptIn(ExperimentalForeignApi::class)
public class IreeRuntime(flags: List<String> = DEFAULT_FLAGS) : AutoCloseable {
    private var handle: CPointer<skainet_iree_runtime>? = memScoped {
        val argv = allocArrayOf(flags.map { it.cstr.ptr })
        skainet_iree_runtime_create(argv, flags.size) ?: failure("runtime")
    }

    private val rt: CPointer<skainet_iree_runtime> get() = handle ?: throw IreeException("runtime closed")

    /** Loads [vmfb] on [device]. */
    public fun session(device: String, vmfb: String): IreeSession =
        IreeSession(skainet_iree_session_create(rt, device, vmfb, null, null) ?: failure("session $vmfb on $device"))

    override fun close() {
        handle?.let { skainet_iree_runtime_release(it) }
        handle = null
    }

    public companion object {
        public val DEFAULT_FLAGS: List<String> = listOf("--torq_hw_type=astra_machina")
    }
}

/** One loaded module on one device. Not thread-safe: callers serialize calls per session. */
@OptIn(ExperimentalForeignApi::class)
public class IreeSession internal constructor(private var handle: CPointer<skainet_iree_session>?) : AutoCloseable {
    private val s: CPointer<skainet_iree_session> get() = handle ?: throw IreeException("session closed")

    /** Calls [function] (e.g. `module.decode_step`) with host tensors and returns its outputs copied back to the host. */
    public fun call(function: String, inputs: List<HostTensor>, maxOutputs: Int = 64): List<HostTensor> = memScoped {
        val ins = allocArray<CPointerVar<skainet_iree_view>>(inputs.size.coerceAtLeast(1))
        val outs = allocArray<CPointerVar<skainet_iree_view>>(maxOutputs)
        var produced = 0
        try {
            inputs.forEachIndexed { i, t ->
                val dims = allocArray<LongVar>(t.shape.size.coerceAtLeast(1))
                t.shape.forEachIndexed { d, v -> dims[d] = v }
                ins[i] = t.bytes.usePinned { pinned ->
                    skainet_iree_view_from_host(s, t.type.code, t.shape.size, dims, pinned.addressOf(0), t.bytes.size.convert())
                } ?: failure("input $i of $function")
            }
            produced = skainet_iree_call(s, function, ins, inputs.size, outs, maxOutputs)
            if (produced < 0) failure("call $function")
            List(produced) { i -> toHost(outs[i]!!) }
        } finally {
            for (i in inputs.indices) ins[i]?.let { skainet_iree_view_release(it) }
            for (i in 0 until produced) outs[i]?.let { skainet_iree_view_release(it) }
        }
    }

    private fun toHost(v: CPointer<skainet_iree_view>): HostTensor {
        val type = ElementType.of(skainet_iree_view_element_type(v))
        val shape = LongArray(skainet_iree_view_rank(v)) { skainet_iree_view_dim(v, it) }
        val bytes = ByteArray(skainet_iree_view_bytes(v).toInt())
        if (bytes.isNotEmpty()) {
            val rc = bytes.usePinned { skainet_iree_view_to_host(s, v, it.addressOf(0), bytes.size.convert()) }
            if (rc != 0) failure("read output")
        }
        return HostTensor(type, shape, bytes)
    }

    override fun close() {
        handle?.let { skainet_iree_session_release(it) }
        handle = null
    }
}
