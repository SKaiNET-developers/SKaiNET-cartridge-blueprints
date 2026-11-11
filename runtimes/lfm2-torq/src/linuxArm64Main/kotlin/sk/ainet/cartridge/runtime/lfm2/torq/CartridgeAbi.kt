@file:OptIn(ExperimentalForeignApi::class, kotlin.experimental.ExperimentalNativeApi::class)

package sk.ainet.cartridge.runtime.lfm2.torq

import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.COpaquePointer
import kotlinx.cinterop.COpaquePointerVar
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.StableRef
import kotlinx.cinterop.UByteVar
import kotlinx.cinterop.alloc
import kotlinx.cinterop.asStableRef
import kotlinx.cinterop.convert
import kotlinx.cinterop.get
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.nativeHeap
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.pointed
import kotlinx.cinterop.ptr
import kotlinx.cinterop.readBytes
import kotlinx.cinterop.reinterpret
import kotlinx.cinterop.set
import kotlinx.cinterop.sizeOf
import kotlinx.cinterop.toKString
import kotlinx.cinterop.value
import platform.posix.free
import platform.posix.size_tVar
import platform.posix.strdup
import sk.ainet.cartridge.runtime.lfm2.torq.abi.ctg_buffer
import sk.ainet.cartridge.runtime.lfm2.torq.abi.ctg_open_options
import sk.ainet.cartridge.runtime.lfm2.torq.abi.ctg_preflight_report
import sk.ainet.cartridge.runtime.lfm2.torq.abi.ctg_session_options
import sk.ainet.cartridge.runtime.lfm2.torq.abi.lfm2_torq_descriptor_canonical
import sk.ainet.cartridge.runtime.lfm2.torq.abi.lfm2_torq_descriptor_pack
import kotlin.experimental.ExperimentalNativeApi

// Cartridge Runtime ABI 0.1 (native/include/cartridge_abi.h), exported from liblfm2_torq_cartridge.so. Every entry
// point catches everything: no Kotlin exception crosses the C boundary. Handles are StableRefs.

internal object Status {
    const val OK = 0
    const val INVALID_ARG = -1
    const val UNSUPPORTED = -2
    const val PREFLIGHT = -3
    const val LOAD = -4
    const val BACKEND = -5
    const val CANCELLED = -6
    const val STATE = -7
    const val ABI_MISMATCH = -8
    const val INTERNAL = -99
}

/** A per-handle error message in C storage, stable until the next failure on the same handle. */
internal class ErrorSlot {
    var text: CPointer<ByteVar>? = null
        private set

    fun set(message: String) {
        val old = text
        text = strdup(message)
        old?.let { free(it) }
    }

    fun clear() {
        text?.let { free(it) }
        text = null
    }
}

/** Failures with no handle: ctg_open, ctg_preflight, ctg_session_end's consumed handle, a NULL argument. */
private val processErrors = ErrorSlot()

/** ctg_preflight's report detail: static storage until the next preflight. */
private val preflightDetail = ErrorSlot()

internal class Result(val kind: Int, tokens: IntArray) {
    val length: Int = tokens.size * 4
    val bytes: CPointer<UByteVar> = nativeHeap.allocArray<UByteVar>(length.coerceAtLeast(1)).also { b ->
        tokens.forEachIndexed { i, t ->
            for (k in 0 until 4) b[i * 4 + k] = ((t ushr (8 * k)) and 0xFF).toUByte()
        }
    }

    fun free() = nativeHeap.free(bytes.rawValue)
}

private inline fun guarded(errors: ErrorSlot, block: () -> Int): Int = try {
    block()
} catch (e: CartridgeException) {
    errors.set(e.message ?: "failed"); e.status
} catch (e: Throwable) {
    errors.set("${e::class.simpleName}: ${e.message}"); Status.INTERNAL
}

private fun <T : Any> handle(p: COpaquePointer?): T? = p?.asStableRef<Any>()?.get()?.let {
    @Suppress("UNCHECKED_CAST")
    it as? T
}

private fun cartridgeOf(p: COpaquePointer?): Lfm2TorqCartridge? = p?.let { handle<Any>(it) as? Lfm2TorqCartridge }
private fun sessionOf(p: COpaquePointer?): GenerationSession? = p?.let { handle<Any>(it) as? GenerationSession }

/** The pack's descriptor.json must be the exact bytes this library was built with. */
private fun checkDescriptor(files: PackFiles, status: Int) {
    val actual = readBytes(files.descriptor)
    val expected = memScoped {
        val n = alloc<size_tVar>()
        val p = lfm2_torq_descriptor_pack(n.ptr) ?: throw CartridgeException(Status.INTERNAL, "no embedded descriptor")
        p.readBytes(n.value.toInt())
    }
    if (!actual.contentEquals(expected)) {
        throw CartridgeException(status, "${files.descriptor} is not the descriptor this runtime was built for (rebuild the runtime with the pack)")
    }
}

private fun checkFiles(files: PackFiles, status: Int) {
    val missing = files.required.filterNot(::readable)
    if (missing.isNotEmpty()) throw CartridgeException(status, "pack_dir is missing ${missing.joinToString()}")
    checkDescriptor(files, status)
}

@CName("ctg_abi_version")
public fun ctgAbiVersion(): UInt = (0u shl 16) or 1u

@CName("ctg_descriptor_json")
public fun ctgDescriptorJson(): CPointer<ByteVar>? = lfm2_torq_descriptor_canonical()?.reinterpret()

@CName("ctg_preflight")
public fun ctgPreflight(packDir: CPointer<ByteVar>?, report: CPointer<ctg_preflight_report>?): Int {
    if (report == null || packDir == null) {
        processErrors.set("ctg_preflight: NULL argument"); return Status.INVALID_ARG
    }
    if (report.pointed.size < sizeOf<ctg_preflight_report>().convert()) {
        processErrors.set("ctg_preflight: report size ${report.pointed.size}"); return Status.ABI_MISMATCH
    }
    // Device-side facts this runtime can check without loading the model: the pack's files and descriptor identity.
    // The torq device itself is opened (and checked) by ctg_open; the Torq runtime is linked in, so there is no
    // separate driver library to probe.
    val status = guarded(processErrors) {
        checkFiles(PackFiles(packDir.toKString()), Status.PREFLIGHT); Status.OK
    }
    report.pointed.ok = if (status == Status.OK) 1 else 0
    if (status == Status.OK) {
        preflightDetail.clear(); report.pointed.detail = null
    } else {
        preflightDetail.set(processErrors.text?.toKString() ?: "preflight failed"); report.pointed.detail = preflightDetail.text
    }
    return status
}

@CName("ctg_open")
public fun ctgOpen(packDir: CPointer<ByteVar>?, opts: CPointer<ctg_open_options>?, out: CPointer<COpaquePointerVar>?): Int {
    if (packDir == null || out == null) {
        processErrors.set("ctg_open: NULL argument"); return Status.INVALID_ARG
    }
    if (opts != null && opts.pointed.size < sizeOf<ctg_open_options>().convert()) {
        processErrors.set("ctg_open: options size ${opts.pointed.size}"); return Status.ABI_MISMATCH
    }
    out.pointed.value = null
    return guarded(processErrors) {
        val files = PackFiles(packDir.toKString())
        checkFiles(files, Status.LOAD)
        val cartridge = Lfm2TorqCartridge(files)
        out.pointed.value = StableRef.create<Any>(cartridge).asCPointer()
        Status.OK
    }
}

@CName("ctg_close")
public fun ctgClose(handle: COpaquePointer?): Int {
    val c = cartridgeOf(handle) ?: run { processErrors.set("ctg_close: not a cartridge"); return Status.INVALID_ARG }
    return guarded(c.errors) {
        if (c.openSessions > 0) throw CartridgeException(Status.STATE, "ctg_close with ${c.openSessions} open session(s)")
        c.close()
        handle!!.asStableRef<Any>().dispose()
        Status.OK
    }
}

@CName("ctg_session_begin")
public fun ctgSessionBegin(handle: COpaquePointer?, opts: CPointer<ctg_session_options>?, out: CPointer<COpaquePointerVar>?): Int {
    val c = cartridgeOf(handle) ?: run { processErrors.set("ctg_session_begin: not a cartridge"); return Status.INVALID_ARG }
    if (out == null) {
        c.errors.set("ctg_session_begin: NULL out"); return Status.INVALID_ARG
    }
    if (opts != null && opts.pointed.size < sizeOf<ctg_session_options>().convert()) {
        c.errors.set("ctg_session_begin: options size ${opts.pointed.size}"); return Status.ABI_MISMATCH
    }
    out.pointed.value = null
    return guarded(c.errors) {
        val s = GenerationSession(c)
        c.openSessions++
        out.pointed.value = StableRef.create<Any>(s).asCPointer()
        Status.OK
    }
}

@CName("ctg_session_end")
public fun ctgSessionEnd(handle: COpaquePointer?): Int {
    val s = sessionOf(handle) ?: run { processErrors.set("ctg_session_end: not a session"); return Status.INVALID_ARG }
    s.cartridge.openSessions--
    s.errors.clear()
    handle!!.asStableRef<Any>().dispose()
    return Status.OK
}

@CName("ctg_infer")
public fun ctgInfer(handle: COpaquePointer?, input: CPointer<ctg_buffer>?, out: CPointer<COpaquePointerVar>?): Int {
    val s = sessionOf(handle) ?: run { processErrors.set("ctg_infer: not a session"); return Status.INVALID_ARG }
    s.errors.set("ctg_infer: this cartridge streams (io.mode streaming): push, flush, pull")
    return Status.UNSUPPORTED
}

@CName("ctg_push")
public fun ctgPush(handle: COpaquePointer?, chunk: CPointer<ctg_buffer>?): Int {
    val s = sessionOf(handle) ?: run { processErrors.set("ctg_push: not a session"); return Status.INVALID_ARG }
    return guarded(s.errors) {
        val b = chunk?.pointed ?: throw CartridgeException(Status.INVALID_ARG, "ctg_push: NULL chunk")
        if (b.size < sizeOf<ctg_buffer>().convert()) throw CartridgeException(Status.ABI_MISMATCH, "ctg_push: buffer size ${b.size}")
        val len = b.len.toLong()
        if (len % 4 != 0L) throw CartridgeException(Status.INVALID_ARG, "ctg_push: $len bytes is not a whole number of int32 token ids")
        val data = b.data?.reinterpret<UByteVar>()
        if (data == null && len > 0) throw CartridgeException(Status.INVALID_ARG, "ctg_push: NULL data")
        val tokens = IntArray((len / 4).toInt()) { i ->
            var v = 0
            for (k in 0 until 4) v = v or (data!![i * 4 + k].toInt() shl (8 * k))
            v
        }
        s.push(tokens)
        Status.OK
    }
}

@CName("ctg_flush")
public fun ctgFlush(handle: COpaquePointer?): Int {
    val s = sessionOf(handle) ?: run { processErrors.set("ctg_flush: not a session"); return Status.INVALID_ARG }
    return guarded(s.errors) { s.flush(); Status.OK }
}

@CName("ctg_pull")
public fun ctgPull(handle: COpaquePointer?, out: CPointer<COpaquePointerVar>?): Int {
    val s = sessionOf(handle) ?: run { processErrors.set("ctg_pull: not a session"); return Status.INVALID_ARG }
    if (out == null) {
        s.errors.set("ctg_pull: NULL out"); return Status.INVALID_ARG
    }
    out.pointed.value = null
    return guarded(s.errors) {
        val next = try {
            s.pull()
        } catch (e: CartridgeException) {
            throw e
        } catch (e: Throwable) {
            throw CartridgeException(Status.BACKEND, "decode step: ${e.message}")
        }
        if (next != null) out.pointed.value = StableRef.create<Any>(Result(next.first, next.second)).asCPointer()
        Status.OK
    }
}

@CName("ctg_cancel")
public fun ctgCancel(handle: COpaquePointer?): Int {
    val s = sessionOf(handle) ?: return Status.INVALID_ARG
    s.cancelRequested.value = 1
    return Status.OK
}

@CName("ctg_result_get_kind")
public fun ctgResultGetKind(result: COpaquePointer?): Int = handle<Any>(result)?.let { (it as? Result)?.kind } ?: 0

@CName("ctg_result_get_payload")
public fun ctgResultGetPayload(result: COpaquePointer?, view: CPointer<ctg_buffer>?): Int {
    val r = handle<Any>(result) as? Result ?: return Status.INVALID_ARG
    val v = view?.pointed ?: return Status.INVALID_ARG
    if (v.size < sizeOf<ctg_buffer>().convert()) return Status.ABI_MISMATCH
    v.data = r.bytes
    v.len = r.length.convert()
    return Status.OK
}

@CName("ctg_result_release")
public fun ctgResultRelease(result: COpaquePointer?) {
    val ref = result?.asStableRef<Any>() ?: return
    (ref.get() as? Result)?.let { it.free(); ref.dispose() }
}

@CName("ctg_last_error")
public fun ctgLastError(handle: COpaquePointer?): CPointer<ByteVar>? = when (val h = handle?.let { handle<Any>(it) }) {
    null -> processErrors.text
    is Lfm2TorqCartridge -> h.errors.text
    is GenerationSession -> h.errors.text
    else -> null
}
