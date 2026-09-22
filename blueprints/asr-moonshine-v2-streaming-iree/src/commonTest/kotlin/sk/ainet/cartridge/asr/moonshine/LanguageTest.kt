package sk.ainet.cartridge.asr.moonshine

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class LanguageTest {
    @Test
    fun theSubtagIsWhatCounts() {
        assertEquals("de", Language.subtag("de-DE")); assertEquals("de", Language.subtag("de_AT")); assertEquals("en", Language.subtag("EN"))
    }

    @Test
    fun selectsThePackFlavorOrTheFallback() {
        assertEquals("de", Language.select("de-DE", listOf("en", "de")))
        assertEquals("en", Language.select("fr-FR", listOf("en", "de")), "unknown language falls back to English when the pack has it")
        assertEquals("de", Language.select("fr-FR", listOf("de"), fallback = "de"))
        assertNull(Language.select("fr-FR", listOf("de")), "no English in the pack and no French: nothing to load")
        assertNull(Language.select("en", emptyList()))
    }
}
