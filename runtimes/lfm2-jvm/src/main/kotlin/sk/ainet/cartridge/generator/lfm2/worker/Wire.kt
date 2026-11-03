package sk.ainet.cartridge.generator.lfm2.worker

import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.EOFException
import java.io.InputStream
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.CodingErrorAction

internal const val MAX_PAYLOAD = 4 * 1024 * 1024
internal object Type {
    const val RUN = 1
    const val CANCEL = 2
    const val STOP = 3
    const val LOADED = 16
    const val PARTIAL = 17
    const val FINAL = 18
    const val ERROR = 19
    const val DONE = 20
}
internal object Status {
    const val INVALID = -1
    const val LOAD = -4
    const val BACKEND = -5
    const val CANCELLED = -6
    const val BAD_STATE = -7
}

/** requestId preserves all uint64 bit patterns, including those represented as negative Long. */
internal data class Frame(val type: Int, val requestId: Long, val payload: ByteArray = byteArrayOf())
internal class FrameException(val requestId: Long, message: String) : IllegalArgumentException(message)

internal class FrameReader(input: InputStream) {
    private val stream = DataInputStream(input)
    fun read(): Frame? {
        val type = stream.read()
        if (type == -1) return null
        var id = 0L
        try {
            id = stream.readLong()
            val size = stream.readInt()
            if (size < 0 || size > MAX_PAYLOAD) throw FrameException(id, "Frame payload exceeds 4 MiB")
            return Frame(type, id, ByteArray(size).also(stream::readFully))
        } catch (failure: EOFException) {
            throw FrameException(id, "Truncated frame")
        }
    }
}

internal class FrameWriter(output: OutputStream) {
    private val stream = DataOutputStream(output)
    @Synchronized fun write(frame: Frame) {
        require(frame.type in 0..255 && frame.payload.size <= MAX_PAYLOAD)
        stream.writeByte(frame.type)
        stream.writeLong(frame.requestId)
        stream.writeInt(frame.payload.size)
        stream.write(frame.payload)
        stream.flush()
    }
    @Synchronized fun finish(frame: Frame) {
        write(frame)
        write(Frame(Type.DONE, frame.requestId))
    }
    fun error(id: Long, status: Int, message: String, done: Boolean = true) {
        val detail = message.take(2000).toByteArray(Charsets.UTF_8)
        val payload = ByteBuffer.allocate(4 + detail.size).order(ByteOrder.BIG_ENDIAN).putInt(status).put(detail).array()
        val frame = Frame(Type.ERROR, id, payload)
        if (done) finish(frame) else write(frame)
    }
}

internal fun utf8(bytes: ByteArray): String = Charsets.UTF_8.newDecoder()
    .onMalformedInput(CodingErrorAction.REPORT)
    .onUnmappableCharacter(CodingErrorAction.REPORT)
    .decode(ByteBuffer.wrap(bytes)).toString()
