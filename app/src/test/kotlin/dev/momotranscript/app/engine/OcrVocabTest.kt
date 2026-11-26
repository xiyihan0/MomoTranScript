package dev.momotranscript.app.engine

import org.junit.Assert.assertEquals
import org.junit.Test

class OcrVocabTest {

    @Test
    fun parsesYamlCharacterDictWithQuotesAndEscapes() {
        val yaml = """
            Global:
              model_name: PP-OCRv6_small_rec
            PostProcess:
              name: CTCLabelDecode
              character_dict:
              - 'a'
              - ''''
              - ":"
              - "\""
              - 阿
              - 洛
              - 娜
              - 　
            OtherSection:
              foo: bar
        """.trimIndent()

        val vocab = OcrVocab.fromYamlText(yaml)
        // index 0 = blank, 1..8 = dict, 9 = " "
        assertEquals(10, vocab.charList.size)
        assertEquals("blank", vocab.charList[0])
        assertEquals("a", vocab.charList[1])
        assertEquals("'", vocab.charList[2])
        assertEquals(":", vocab.charList[3])
        assertEquals("\"", vocab.charList[4])
        assertEquals("阿", vocab.charList[5])
        assertEquals("洛", vocab.charList[6])
        assertEquals("娜", vocab.charList[7])
        assertEquals("\u3000", vocab.charList[8])
        assertEquals(" ", vocab.charList[9])

        // CTC collapse: [0, 5, 5, 0, 6, 7, 7, 9, 1] -> "阿洛娜 a"
        val decoded = vocab.decodeCtc(
            longArrayOf(0, 5, 5, 0, 6, 7, 7, 9, 1),
            floatArrayOf(0.99f, 0.90f, 0.80f, 0.99f, 0.95f, 0.70f, 0.60f, 0.85f, 1.00f),
        )
        assertEquals("阿洛娜 a", decoded.text)
        assertEquals(0.70f, decoded.minCharScore, 1e-4f)
        assertEquals((0.90f + 0.95f + 0.70f + 0.85f + 1.00f) / 5f, decoded.score, 1e-4f)
    }
}
