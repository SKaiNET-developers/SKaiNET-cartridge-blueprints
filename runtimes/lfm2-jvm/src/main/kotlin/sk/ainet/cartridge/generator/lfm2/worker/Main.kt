package sk.ainet.cartridge.generator.lfm2.worker

import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.FileDescriptor
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.file.Path
import kotlin.system.exitProcess

fun main(args: Array<String>) {
    val output = BufferedOutputStream(FileOutputStream(FileDescriptor.out))
    // Dependency diagnostics may print to System.out; only the saved raw descriptor
    // is allowed to carry protocol bytes. Ordinary printing is routed to stderr.
    System.setOut(System.err)
    if (args.size == 3 && args[0] == "--preflight" && args[1] == "--pack-dir") {
        try {
            val report = preflightRuntime(Path.of(args[2]))
            output.write((report.toString() + "\n").toByteArray(Charsets.UTF_8)); output.flush()
            exitProcess(0)
        } catch (error: Exception) {
            System.err.println("Worker preflight failed: ${error.message}")
            exitProcess(1)
        }
    }
    val runtime = try {
        require(args.size == 2 && args[0] == "--pack-dir") { "Usage: --pack-dir <package-directory>" }
        loadRuntime(Path.of(args[1]))
    } catch (error: Exception) {
        try { FrameWriter(output).error(0, Status.LOAD, error.message ?: "Model load failed", done = false) }
        catch (_: Exception) { /* The parent may already have closed its pipe. */ }
        System.err.println("Worker startup failed: ${error.message}")
        exitProcess(1)
    }
    val status = WorkerService(runtime, BufferedInputStream(FileInputStream(FileDescriptor.`in`)), output).run()
    exitProcess(status)
}
