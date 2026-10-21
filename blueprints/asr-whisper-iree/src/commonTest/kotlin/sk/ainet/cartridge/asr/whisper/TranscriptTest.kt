package sk.ainet.cartridge.asr.whisper

import kotlin.test.Test
import kotlin.test.assertEquals

class TranscriptTest {
    @Test
    fun theFragmentOfARestartedSentenceIsDropped() {
        // what the guard leaves when the repeat starts with a different token than the sentence (`F` vs ` F`)
        assertEquals("Fern sie ihr lauter.", Transcript.dropRepeat("Fern sie ihr lauter. F"))
    }

    @Test
    fun aCompleteRepeatAndEverythingAfterItIsDropped() {
        assertEquals("Turn it up.", Transcript.dropRepeat("Turn it up. Turn it up. Turn"))
        assertEquals("Leiser!", Transcript.dropRepeat("Leiser! leiser"))
    }

    @Test
    fun aRestartThatBeganWithAVariantTokenIsDropped() {
        // the restart is no literal prefix of the sentence, but the guard stopped the decode inside it
        assertEquals("Mach das Licht an.", Transcript.dropRepeat("Mach das Licht an. Mac"))
        assertEquals("Licht an!!", Transcript.dropRepeat("Licht an!!ich"))
    }

    @Test
    fun aDifferentSecondSentenceStays() {
        assertEquals("Next channel. Then louder.", Transcript.dropRepeat("Next channel. Then louder."))
    }

    @Test
    fun textWithoutASentenceEndIsUnchanged() {
        assertEquals("Licht bitte", Transcript.dropRepeat(" Licht bitte "))
        assertEquals("", Transcript.dropRepeat(""))
    }
}
