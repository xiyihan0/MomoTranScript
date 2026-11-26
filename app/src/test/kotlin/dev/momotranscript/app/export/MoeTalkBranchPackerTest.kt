package dev.momotranscript.app.export

import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test

class MoeTalkBranchPackerTest {

    private fun sampleChat(text: String) = buildJsonObject {
        put("type", "chat")
        put("replyDepth", 0)
        put(
            "sCharacter",
            buildJsonObject {
                put("no", "Arona")
                put("index", "Arona_01")
            },
        )
        put("content", text)
        put("isFirst", true)
        put("isRight", false)
    }

    @Test
    fun sanitizeAndDeduplicateBranchNamesProtectsHtmlAttributes() {
        val names = MoeTalkBranchPacker.allocateUniqueBranchNames(
            listOf("第1话 \"序章\" <测试>", "第1话 \"序章\" <测试>", "0", ""),
        )
        assertEquals(4, names.size)
        assertEquals(4, names.toSet().size)
        for (name in names) {
            assertFalse(name.contains('"'))
            assertFalse(name.contains('<'))
            assertFalse(name.contains('>'))
            assertFalse(name.contains('&'))
            assertFalse(name.contains('\n'))
        }
        assertEquals("3", names[2])
        assertEquals("4", names[3])
    }

    @Test
    fun multiItemBranchPackingCreatesRootReplyCardsAndBranchMessages() {
        val archive = MoeTalkBranchPacker.packArchive(
            items = listOf(
                MoeTalkBranchPacker.BranchInput("1", listOf(sampleChat("第一章台词"))),
                MoeTalkBranchPacker.BranchInput("2", listOf(sampleChat("第二章台词A"), sampleChat("第二章台词B"))),
            ),
            title = "合集测试",
            mode = MoeTalkBranchPacker.MultiItemPackMode.BRANCH_PER_ITEM,
        )

        val chat = archive["CHAT"]!!.jsonArray
        // 2 root reply index cards + 1 message in branch "1" + 2 messages in branch "2"
        assertEquals(5, chat.size)

        val card0 = chat[0].jsonObject
        val card1 = chat[1].jsonObject
        assertEquals("reply", card0["type"]!!.jsonPrimitive.content)
        assertEquals(0, card0["replyDepth"]!!.jsonPrimitive.int)
        assertEquals("1", card0["content"]!!.jsonPrimitive.content)
        assertEquals("2", card1["content"]!!.jsonPrimitive.content)

        assertEquals("1", chat[2].jsonObject["replyDepth"]!!.jsonPrimitive.content)
        assertEquals("2", chat[3].jsonObject["replyDepth"]!!.jsonPrimitive.content)
        assertEquals("2", chat[4].jsonObject["replyDepth"]!!.jsonPrimitive.content)
    }

    @Test
    fun singleItemArchiveStaysFlatAtRootDepthZero() {
        val archive = MoeTalkBranchPacker.packArchive(
            items = listOf(MoeTalkBranchPacker.BranchInput("1", listOf(sampleChat("单图台词")))),
            mode = MoeTalkBranchPacker.MultiItemPackMode.BRANCH_PER_ITEM,
        )
        val chat = archive["CHAT"]!!.jsonArray
        assertEquals(1, chat.size)
        assertEquals(0, chat[0].jsonObject["replyDepth"]!!.jsonPrimitive.int)
    }
}
