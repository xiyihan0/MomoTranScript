package dev.momotranscript.app.avatar

import dev.momotranscript.app.nativecore.NativeCore
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
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

    @Test
    fun decodeCaptionRejectsNonFiniteLogitsInsteadOfClampingInfToOne() {
        val decoder = sampleDecoder()
        // +Inf must NEVER be clamped to 30f -> sigmoid(30) == 1.0f fake confidence tag
        val posInf = floatArrayOf(Float.POSITIVE_INFINITY, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
        val negInf = floatArrayOf(2.5f, 0f, 0f, Float.NEGATIVE_INFINITY, 0f, 0f, 0f, 0f)
        val nanVec = floatArrayOf(2.5f, 0f, 0f, Float.NaN, 0f, 0f, 0f, 0f)
        val allInf = FloatArray(8) { Float.POSITIVE_INFINITY }

        assertEquals("", decoder.decodeCaption(posInf))
        assertEquals("", decoder.decodeCaption(negInf))
        assertEquals("", decoder.decodeCaption(nanVec))
        assertEquals("", decoder.decodeCaption(allInf))
    }

    @Test
    fun fakeEncoderInjectingNanAndInfIsolatesBadSlotsWithoutDecodingOrCaching() {
        val decoder = sampleDecoder()
        val healthyLogits = floatArrayOf(
            1.5f,  // smile
            -1.0f, // grin
            -1.0f, // closed_mouth
            2.0f,  // blush
            -1.0f, // sweatdrop
            -1.0f, // sweat
            -1.0f, // closed_eyes
            -1.0f, // surprised
        )
        val fakeOutputs = listOf(
            // slot 0: healthy
            NativeCore.UnifiedEncodeOutput(floatArrayOf(3f, 4f), healthyLogits),
            // slot 1: +Inf in exprLogits (would have falsely decoded to 1.0 without guard)
            NativeCore.UnifiedEncodeOutput(floatArrayOf(1f, 0f), floatArrayOf(Float.POSITIVE_INFINITY, 0f, 0f, 0f, 0f, 0f, 0f, 0f)),
            // slot 2: NaN in idEmb
            NativeCore.UnifiedEncodeOutput(floatArrayOf(Float.NaN, 1f), healthyLogits),
            // slot 3: -Inf in idEmb
            NativeCore.UnifiedEncodeOutput(floatArrayOf(Float.NEGATIVE_INFINITY, 1f), healthyLogits),
            // slot 4: all-zero idEmb (degenerate norm)
            NativeCore.UnifiedEncodeOutput(floatArrayOf(0f, 0f), healthyLogits),
            // slot 5: healthy
            NativeCore.UnifiedEncodeOutput(floatArrayOf(0f, 2f), healthyLogits),
        )

        val n = fakeOutputs.size
        val missSlots = (0 until n).toList()
        val inputs = List(n) { floatArrayOf(it.toFloat()) }
        val thumbs = List(n) { floatArrayOf(it.toFloat()) }
        val repEmbs = arrayOfNulls<FloatArray>(n)
        val repExprs = Array(n) { "" }
        val cache = ArrayList<OnnxAvatarMatcher.CachedThumbEmb>()

        val failed = OnnxAvatarMatcher.encodeAndCacheMisses(
            missSlots = missSlots,
            inputsNchw = inputs,
            repThumbs = thumbs,
            repEmbeddings = repEmbs,
            repExprSummaries = repExprs,
            cache = cache,
            maxCacheSize = 64,
            isUnified = true,
            project = { it },
            decoder = decoder,
            runEncoder = { inp -> fakeOutputs[inp[0].toInt()] },
        )

        assertEquals(4, failed)
        // Healthy slots 0 & 5 decoded, normalized, and cached
        assertNotNull(repEmbs[0])
        assertEquals(0.6f, repEmbs[0]!![0], 1e-5f)
        assertEquals(0.8f, repEmbs[0]!![1], 1e-5f)
        assertEquals("脸红/微笑", repExprs[0])

        assertNotNull(repEmbs[5])
        assertEquals(0.0f, repEmbs[5]!![0], 1e-5f)
        assertEquals(1.0f, repEmbs[5]!![1], 1e-5f)
        assertEquals("脸红/微笑", repExprs[5])

        // Corrupted slots 1..4 failed cleanly: null emb, empty expr, NOT written to cache
        for (bad in 1..4) {
            assertNull("slot $bad should have null emb", repEmbs[bad])
            assertEquals("slot $bad should not decode expression", "", repExprs[bad])
        }
        assertEquals(2, cache.size)
        assertTrue(cache.all { entry -> entry.emb.all { it.isFinite() } })

        // End-to-end MomoScript annotation: failed slot keeps plain [image], healthy slot & OCR untouched
        val script = "> 柚子: [image]\n> 葛叶: [image]\n> 葛叶: 正常 OCR 台词\n"
        val matches = listOf(
            InsertVisualMatch(kind = "charface", status = "matched", character = "柚子", expressionSummary = repExprs[0]),
            InsertVisualMatch(kind = "illustration", status = "encode_failed", expressionSummary = repExprs[1]),
        )
        val out = ExpressionDecoder.annotateMomoScriptImageLines(script, matches)
        assertEquals("> 柚子: [image: 脸红/微笑]\n> 葛叶: [image]\n> 葛叶: 正常 OCR 台词\n", out)
    }
}
