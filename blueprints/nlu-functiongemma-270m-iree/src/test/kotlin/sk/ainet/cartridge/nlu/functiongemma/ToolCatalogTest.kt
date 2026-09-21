package sk.ainet.cartridge.nlu.functiongemma

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

class ToolCatalogTest {
    private val dir = File(System.getProperty("blueprint.dir"))
    private val toy = ToolCatalog.load(File(dir, "samples/toy-catalog.json"))

    @Test
    fun `the toy catalog loads and becomes tool definitions`() {
        assertEquals("toy-tools.v1", toy.id)
        assertEquals(setOf("set_timer", "toggle_lamp", "roll_dice"), toy.names)
        val lamp = toy.definitions().single { it.name == "toggle_lamp" }
        assertTrue(lamp.parameters.toString().contains("\"enum\":[\"on\",\"off\"]"), lamp.parameters.toString())
        assertTrue(lamp.parameters.toString().contains("\"required\":[\"state\"]"))
    }

    @Test
    fun `near-miss names snap to the catalog, far ones do not`() {
        assertEquals("toggle_lamp", toy.snapName("toggle_lamp"))
        assertEquals("toggle_lamp", toy.snapName("Toggle__Lamp"))
        assertEquals("toggle_lamp", toy.snapName("toggle_lampp"))
        assertEquals("set_timer", toy.snapName("set-timer"))
        assertNull(toy.snapName("open_garage_door"))
        assertNull(toy.snapName(null))
    }

    @Test
    fun `an ambiguous or empty catalog is refused`() {
        assertFailsWith<IllegalArgumentException> { ToolCatalog.parse("""{"id":"x","functions":[]}""") }
        assertFailsWith<IllegalArgumentException> {
            ToolCatalog.parse("""{"id":"x","functions":[{"name":"a","description":"1"},{"name":"a","description":"2"}]}""")
        }
        assertFailsWith<Exception> { ToolCatalog.parse("""{"id":"x","functions":[{"name":"a","description":"1"}],"unknown":true}""") }
    }
}
