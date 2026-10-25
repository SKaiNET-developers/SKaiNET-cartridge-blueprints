package sk.ainet.cartridge.generator.llama32

import sk.ainet.backend.api.kernel.KernelPacks
import sk.ainet.exec.kernel.FfmRowMajorKernelPack
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

internal actual fun fileSize(path: String): Long? =
    File(path).takeIf { it.isFile }?.length()

private val kernelsInstalled = AtomicBoolean(false)

internal actual fun installKernelPacks() {
    if (!kernelsInstalled.compareAndSet(false, true)) return
    KernelPacks.install()
    FfmRowMajorKernelPack.install()
}

internal actual fun envVar(name: String): String? = System.getenv(name)

internal actual fun homeDir(): String? = System.getProperty("user.home")
