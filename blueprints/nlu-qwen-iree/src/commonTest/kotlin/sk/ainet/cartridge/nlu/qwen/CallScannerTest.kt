package sk.ainet.cartridge.nlu.qwen

import kotlin.test.Test
import kotlin.test.assertEquals

class CallScannerTest {
    @Test
    fun `the call ends at the brace that closes the forced object`() {
        val text = """roll_dice", "arguments": {}}"""
        assertEquals(text.length, CallScanner.completeLength(text))
        assertEquals(text.length, CallScanner.completeLength(text + "\n</tool_call>\n<tool_call>{\"name\": \"x\"}"))
    }

    @Test
    fun `braces and escaped quotes inside strings are not structure`() {
        val text = """a}b", "arguments": {"s": "x\"}{y"}}"""
        assertEquals(text.length, CallScanner.completeLength(text))
    }

    @Test
    fun `an unfinished call reports -1`() {
        assertEquals(-1, CallScanner.completeLength(""))
        assertEquals(-1, CallScanner.completeLength("""toggle_lamp", "arguments": {"state": "on"}"""))
    }
}
