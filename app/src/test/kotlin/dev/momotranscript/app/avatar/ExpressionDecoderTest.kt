package dev.momotranscript.app.avatar

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class ExpressionDecoderTest {
    @get:Rule
    val tempFolder = TemporaryFolder()

    private fun sampleDecoder(): ExpressionDecoder = ExpressionDecoder(
        listOf(
            ExpressionTagEntry(0, "smile", "mouth", "微笑", 0.55f),
            ExpressionTagEntry(1, "grin", "mouth", "露齿笑", 0.50f),
            ExpressionTagEntry(2, "closed_mouth", "mouth", "闭嘴", 0.30f),
            ExpressionTagEntry(3, "blush", "effects", "脸红", 0.45f),
            ExpressionTagEntry(4, "sweatdrop", "effects", "流汗", 0.40f),
            ExpressionTagEntry(5, "sweat", "effects", "冒汗", 0.45f),
            ExpressionTagEntry(6, "closed_eyes", "eyes", "闭眼", 0.50f),
            ExpressionTagEntry(7, "surprised", "emotions", "惊讶", 0.45f),
        ),
    )

    @Test
    fun decodeCaptionAppliesThresholdsDedupAndCategoryOrder() {
        val decoder = sampleDecoder()
        // High logits for closed_mouth (ignored), smile + grin (deduped to grin since higher),
        // blush, sweatdrop + sweat (deduped to sweatdrop), surprised
        val logits = floatArrayOf(
            1.2f,  // smile (~0.768)
            2.4f,  // grin (~0.917) -> wins over smile in "smile" dedup group
            5.0f,  // closed_mouth -> filtered out
            1.8f,  // blush (~0.858) -> effects
            2.0f,  // sweatdrop (~0.881) -> effects, wins over sweat
            0.5f,  // sweat (~0.622) -> deduped
            -2.0f, // closed_eyes (~0.119) -> below threshold
            1.5f,  // surprised (~0.818) -> emotions
        )
        // Top-3 by score: grin (0.917), sweatdrop (0.881), blush (0.858)
        // Re-ordered by category priority: effects (sweatdrop, blush), mouth (grin)
        assertEquals("流汗/脸红/露齿笑", decoder.decodeCaption(logits, maxTags = 3))

        // If surprised is highest, emotions comes first
        val logits2 = floatArrayOf(
            1.2f,  // smile
            -1.0f, // grin
            5.0f,  // closed_mouth
            1.8f,  // blush
            -1.0f, // sweatdrop
            -1.0f, // sweat
            2.5f,  // closed_eyes (~0.924)
            3.0f,  // surprised (~0.953)
        )
        // Top-3 by score: surprised (emotions), closed_eyes (eyes), blush (effects)
        // Sorted by category: emotions -> effects -> eyes
        assertEquals("惊讶/脸红/闭眼", decoder.decodeCaption(logits2, maxTags = 3))
    }

    @Test
    fun loadIfPresentParsesVocabJson() {
        val f = File(tempFolder.root, "vocab_48.json")
        f.writeText(
            """
            {
              "version": "v2",
              "num_tags": 2,
              "tags": [
                {"id": 0, "tag": "blush", "category": "effects", "zh": "脸红", "recommended_threshold": 0.45},
                {"id": 1, "tag": "smile", "category": "mouth", "zh": "微笑", "recommended_threshold": 0.55}
              ]
            }
            """.trimIndent(),
            Charsets.UTF_8,
        )
        val dec = ExpressionDecoder.loadIfPresent(f)
        assertNotNull(dec)
        assertEquals("脸红/微笑", dec!!.decodeCaption(floatArrayOf(2.0f, 1.5f)))
    }

    @Test
    fun annotateMomoScriptImageLinesHandlesSpeakersStandaloneEmojiAndBond() {
        val raw = """
            > 柚子: [image]
            [image]
            > [image]
            - [image]
            > 美咲: 普通台词
        """.trimIndent() + "\n"

        val slots = listOf(
            InsertVisualMatch(
                kind = "charface",
                status = "matched",
                character = "柚子",
                expressionSummary = "流汗/脸红",
            ),
            InsertVisualMatch(
                kind = "charface",
                status = "matched",
                character = "爱丽丝",
                expressionSummary = "惊讶/闭眼",
            ),
            InsertVisualMatch(
                kind = "emoji",
                status = "matched",
                expressionSummary = "微笑",
            ),
            null, // bond or unmatched without expression -> keep [image]
        )

        val annotated = ExpressionDecoder.annotateMomoScriptImageLines(raw, slots)
        val expected = """
            > 柚子: [image: 流汗/脸红]
            [image: 爱丽丝 · 惊讶/闭眼]
            > [image: 表情包 · 微笑]
            - [image]
            > 美咲: 普通台词
        """.trimIndent() + "\n"
        assertEquals(expected, annotated)
    }
}
