package sk.ainet.cartridge.generator.llama32

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.toKString
import sk.ainet.backend.api.kernel.KernelPacks
import kotlin.concurrent.AtomicInt
import platform.posix.SEEK_END
import platform.posix.SEEK_SET
import platform.posix.fclose
import platform.posix.fopen
import platform.posix.fseek
import platform.posix.ftell
import platform.posix.getenv

// One POSIX actual for all Apple targets (macosArm64 + both iOS), as in the sibling blueprints.

@OptIn(ExperimentalForeignApi::class)
internal actual fun fileSize(path: String): Long? {
    val file = fopen(path, "rb") ?: return null
    try {
        fseek(file, 0, SEEK_END)
        val size = ftell(file)
        return if (size < 0) null else size
    } finally {
        fclose(file)
    }
}

private val kernelsInstalled = AtomicInt(0)

internal actual fun installKernelPacks() {
    if (!kernelsInstalled.compareAndSet(0, 1)) return
    KernelPacks.install()
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun envVar(name: String): String? = getenv(name)?.toKString()

@OptIn(ExperimentalForeignApi::class)
internal actual fun homeDir(): String? = getenv("HOME")?.toKString()
