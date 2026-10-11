package sk.ainet.cartridge.generator.smollm2

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.toKString
import kotlinx.cinterop.usePinned
import platform.posix.SEEK_END
import platform.posix.SEEK_SET
import platform.posix.fclose
import platform.posix.fopen
import platform.posix.fread
import platform.posix.fseek
import platform.posix.ftell
import platform.posix.getenv

// One POSIX actual for all Apple targets (macosArm64 + both iOS) — unlike embedding-leaf,
// this cartridge compiles for iOS, so the file read lives in appleMain, not macosArm64Main.

@OptIn(ExperimentalForeignApi::class)
internal actual fun readTextFile(path: String): String? {
    val file = fopen(path, "rb") ?: return null
    try {
        fseek(file, 0, SEEK_END)
        val size = ftell(file)
        if (size < 0) return null
        fseek(file, 0, SEEK_SET)
        val bytes = ByteArray(size.toInt())
        if (bytes.isNotEmpty()) {
            val read = bytes.usePinned { fread(it.addressOf(0), 1u, size.toULong(), file) }
            if (read.toLong() != size) return null
        }
        return bytes.decodeToString()
    } finally {
        fclose(file)
    }
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun envVar(name: String): String? = getenv(name)?.toKString()

@OptIn(ExperimentalForeignApi::class)
internal actual fun homeDir(): String? = getenv("HOME")?.toKString()
