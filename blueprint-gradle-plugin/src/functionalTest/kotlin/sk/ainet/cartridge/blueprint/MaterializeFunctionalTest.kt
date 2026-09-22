package sk.ainet.cartridge.blueprint

import org.gradle.testkit.runner.GradleRunner
import org.gradle.testkit.runner.TaskOutcome
import java.io.File
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** Real Gradle builds against the synthetic blueprint: the plugin as a blueprint module would apply it. */
class MaterializeFunctionalTest {

    private fun project(f: Fixture): File {
        val dir = f.blueprintDir
        File(dir, "settings.gradle.kts").writeText("""rootProject.name = "toy-blueprint"""" + "\n")
        File(dir, "build.gradle.kts").writeText(
            """
            plugins { id("sk.ainet.cartridge.blueprint") }

            // Stand-ins for the model-specific steps (export / compile / build-runtime) a real blueprint module has.
            val compileGraphs = tasks.register("compileGraphs") {
                val out = layout.buildDirectory.dir("graphs")
                outputs.dir(out)
                doLast {
                    out.get().asFile.mkdirs()
                    out.get().file("prefill.vmfb").asFile.writeBytes(byteArrayOf(1, 2, 3))
                    out.get().file("decode.vmfb").asFile.writeBytes(byteArrayOf(4, 5))
                }
            }
            val buildRuntime = tasks.register("buildRuntime") {
                val out = layout.buildDirectory.file("runtime/libtoy-ctg.so")
                outputs.file(out)
                doLast { out.get().asFile.apply { parentFile.mkdirs() }.writeBytes(byteArrayOf(0x7f, 0x45, 0x4c, 0x46)) }
            }

            // A flavor-scoped product: its task must run only when the profile selects the flavor.
            val bakeDe = tasks.register("bakeDe") {
                val out = layout.buildDirectory.file("de/extra.bin")
                outputs.file(out)
                doLast { out.get().asFile.apply { parentFile.mkdirs() }.writeBytes(byteArrayOf(9)) }
            }

            blueprint {
                product("graphs", compileGraphs.map { layout.buildDirectory.dir("graphs").get() })
                product("runtime-so", buildRuntime.map { layout.buildDirectory.file("runtime/libtoy-ctg.so").get() })
                product("de-extra", bakeDe.map { layout.buildDirectory.file("de/extra.bin").get() }, flavor = "de")
            }
            """.trimIndent(),
        )
        return dir
    }

    private fun runner(dir: File, vararg args: String) =
        GradleRunner.create().withProjectDir(dir).withPluginClasspath().withArguments(*args, "--stacktrace")

    @Test
    fun `materializeCartridge runs the step tasks, fetches, packs and signs`() {
        val f = Fixture()
        val dir = project(f)
        val (privatePem, publicKey) = Fixture.keyPair()
        val keyFile = File(f.root, "signing-key.pem").apply { writeText(privatePem) }
        File(f.root, "signing-key.pub").writeText(Fixture.publicPem(publicKey))

        val result = runner(dir, "materializeCartridge", "-Pprofile=${f.profileFile.path}", "-PsigningKeyFile=${keyFile.path}").build()

        assertEquals(TaskOutcome.SUCCESS, result.task(":blueprintFetch")?.outcome)
        assertEquals(TaskOutcome.SUCCESS, result.task(":compileGraphs")?.outcome, "product providers must carry their task dependency")
        assertEquals(TaskOutcome.SUCCESS, result.task(":buildRuntime")?.outcome)
        assertEquals(TaskOutcome.SUCCESS, result.task(":materializeCartridge")?.outcome)
        assertFalse(result.output.contains("UNSIGNED"))

        val pack = File(dir, "build/cartridge/nlu-toy-iree-cpu-arm64")
        for (path in listOf("descriptor.json", "manifest.json", "artifacts/runtime/libtoy-ctg.so", "artifacts/model/prefill.vmfb",
            "artifacts/weights/model.bin", "artifacts/tokenizer/tokenizer.json", "artifacts/other/tool-catalog.json")) {
            assertTrue(File(pack, path).isFile, "missing $path")
        }
        assertContains(File(pack, "manifest.json").readText(), "\"signatures\"")
        // Left on disk on purpose: CI verifies this manifest with the spec's verify_manifest.py.
        println("PACK_DIR=${pack.path}")
        println("PUBLIC_KEY=${File(f.root, "signing-key.pub").path}")
    }

    @Test
    fun `a flavor's step tasks are scheduled only when the profile selects the flavor`() {
        val fOff = Fixture(withGermanFlavor = true)
        val off = runner(project(fOff), "materializeCartridge", "-Pprofile=${fOff.profileFile.path}").build()
        assertTrue(off.task(":bakeDe") == null, "unselected flavor: its task must not be in the graph")
        val fOn = Fixture(withGermanFlavor = true, selectFlavors = listOf("de"))
        val on = runner(project(fOn), "materializeCartridge", "-Pprofile=${fOn.profileFile.path}").build()
        assertEquals(TaskOutcome.SUCCESS, on.task(":bakeDe")!!.outcome)
    }

    @Test
    fun `missing license acceptance fails the build before anything is fetched`() {
        val f = Fixture(accept = false)
        val dir = project(f)
        val result = runner(dir, "materializeCartridge", "-Pprofile=${f.profileFile.path}").buildAndFail()
        assertContains(result.output, "License acceptance required before anything is downloaded")
        assertContains(result.output, "Toy Terms of Use")
        assertFalse(File(dir, "build/blueprint/sources/weights").exists())
        assertFalse(File(dir, "build/cartridge").exists())
    }

    @Test
    fun `no profile is a clear error, and validate works without one`() {
        val f = Fixture()
        val dir = project(f)
        assertContains(runner(dir, "blueprintFetch").buildAndFail().output, "No materialization profile")
        assertContains(runner(dir, "blueprintValidate").build().output, "blueprint nlu-toy-iree@0.1.0: valid")
    }

    @Test
    fun `without a key the build succeeds and warns loudly that the manifest is unsigned`() {
        val f = Fixture()
        val dir = project(f)
        val result = runner(dir, "materializeCartridge", "-Pprofile=${f.profileFile.path}").build()
        assertContains(result.output, "UNSIGNED")
    }
}
