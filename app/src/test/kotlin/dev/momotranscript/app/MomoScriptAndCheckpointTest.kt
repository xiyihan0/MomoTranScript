package dev.momotranscript.app

import dev.momotranscript.app.ui.INLINE_TOKEN_REGEX
import dev.momotranscript.app.ui.LEFT_SPEAKER_REGEX
import dev.momotranscript.app.ui.parseMomoScriptPages
import dev.momotranscript.app.ui.replaceMomoScriptPage
import dev.momotranscript.app.work.CheckpointBranchInput
import dev.momotranscript.app.work.CheckpointStore
import dev.momotranscript.app.work.TranscriptionCheckpoint
import dev.momotranscript.app.work.TranscriptionJobSpec
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class MomoScriptAndCheckpointTest {
    @get:Rule
    val tempFolder = TemporaryFolder()

    @Test
    fun parseAndReplaceMultiPageMomoScript() {
        val raw = """
            === 第 1 页 ===
            > 宫子: 老师好
            < 早上好

            === 第 2 页 ===
            - 十分钟后
            > 柚子: [image]
            > [unreadable]
        """.trimIndent()

        val pages = parseMomoScriptPages(raw)
        assertEquals(2, pages.size)
        assertEquals("第 1 页", pages[0].title)
        assertTrue(pages[0].body.contains("> 宫子: 老师好"))
        assertEquals("第 2 页", pages[1].title)
        assertTrue(pages[1].body.contains("[unreadable]"))

        val edited = replaceMomoScriptPage(raw, 1, "- 十分钟后\n> 柚子: [image]\n> 校对后的台词")
        val pagesAfter = parseMomoScriptPages(edited)
        assertEquals(2, pagesAfter.size)
        assertTrue(pagesAfter[0].body.contains("> 宫子: 老师好"))
        assertTrue(pagesAfter[1].body.contains("> 校对后的台词"))
        assertTrue(!pagesAfter[1].body.contains("[unreadable]"))
    }

    @Test
    fun checkpointStoreSavePauseResumeAndComplete() {
        val jobsRoot = tempFolder.newFolder("jobs")
        val jobDir = File(jobsRoot, "abc12345").apply { mkdirs() }
        val spec = TranscriptionJobSpec(
            jobId = "abc12345",
            kind = "pdf",
            title = "测试合集",
            uris = listOf("content://example/test.pdf"),
            displayNames = listOf("test.pdf"),
        )
        File(jobDir, "spec.json").writeText(
            Json.encodeToString(TranscriptionJobSpec.serializer(), spec),
            Charsets.UTF_8,
        )
        CheckpointStore.saveTempImage(jobDir, "Image-0001", "data:image/webp;base64,AAAA")
        val cp = TranscriptionCheckpoint(
            jobId = "abc12345",
            title = "测试合集",
            completedItems = 3,
            totalItems = 10,
            totalWindows = 15,
            totalLines = 99,
            matchedOfficialAvatars = 4,
            matchedVisualAssets = 2,
            elapsedMs = 12_500L,
            customCounter = 1,
            imageCounter = 1,
            tempImageIds = listOf("Image-0001"),
            branchInputs = listOf(
                CheckpointBranchInput(
                    label = "P001",
                    messages = listOf(buildJsonObject { put("content", "hello") }),
                ),
            ),
            combinedMomoScript = "=== 第 1 页 ===\n> 宫子: 你好\n",
        )
        CheckpointStore.saveCheckpoint(jobDir, cp)
        CheckpointStore.markPaused(jobDir)

        val found = CheckpointStore.findLatestResumable(jobsRoot)
        assertNotNull(found)
        assertEquals("abc12345", found!!.spec.jobId)
        assertEquals(3, found.checkpoint.completedItems)
        assertEquals(10, found.checkpoint.totalItems)
        assertTrue(found.wasPausedByUser)

        val imgs = CheckpointStore.loadTempImages(jobDir, found.checkpoint.tempImageIds)
        assertEquals("data:image/webp;base64,AAAA", imgs["Image-0001"])

        CheckpointStore.markCompleted(jobDir)
        assertNull(CheckpointStore.findLatestResumable(jobsRoot))
    }

    @Test
    fun syntaxHighlightRegexDoesNotTreatImageColonAsSpeakerName() {
        // Continuation line with `[image: ...]` must NOT match as a speaker name!
        assertNull(LEFT_SPEAKER_REGEX.matchEntire("> [image: 得意/半睁眼/微笑]"))
        val token = INLINE_TOKEN_REGEX.find("> [image: 得意/半睁眼/微笑]")?.value
        assertEquals("[image: 得意/半睁眼/微笑]", token)

        // Real speaker + `[image: ...]` must split cleanly
        val mSpeaker = LEFT_SPEAKER_REGEX.matchEntire("> 葛叶: [image: 闭眼/张嘴/微笑]")
        assertNotNull(mSpeaker)
        assertEquals("葛叶", mSpeaker!!.groupValues[1])
        assertEquals("[image: 闭眼/张嘴/微笑]", mSpeaker.groupValues[3])

        // `[unknown speaker]: ...` must still match as speaker
        val mUnknown = LEFT_SPEAKER_REGEX.matchEntire("> [unknown speaker]: 你好")
        assertNotNull(mUnknown)
        assertEquals("[unknown speaker]", mUnknown!!.groupValues[1])
    }
}
