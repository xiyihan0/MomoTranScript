package dev.momotranscript.app.export

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

/**
 * Packs one or more transcribed images / PDF pages into a single MoeTalkB-compatible `.json`
 * archive, using MoeTalk's native `分支管理` (`replyDepth`) mechanism for multi-item batches.
 *
 * Verified against MoeTalkB `MoeScript/TOP.js` (`replyDepth()`, `TOP_replyEdit()`):
 * - Root messages have numeric `replyDepth: 0`.
 * - Each packaged image/page gets a root `type: "reply"` index card (`replyDepth: 0`,
 *   `content: <branchName>`) and all of its transcribed messages carry `replyDepth: "<branchName>"`.
 * - Because `TOP.js` injects branch names directly into HTML attributes (`id="${reply[i]}"`),
 *   branch names are sanitized to remove quotes, angle brackets, ampersands, and newlines,
 *   and deduplicated within the archive's flat `replyDepth` namespace.
 */
object MoeTalkBranchPacker {
    const val MOETALK_ARCHIVE_VERSION: Double = 261005.214

    enum class MultiItemPackMode {
        /** One MoeTalk branch per image / PDF page with clickable root index cards (`分支管理`). */
        BRANCH_PER_ITEM,

        /** Concatenate all items sequentially into root (`replyDepth: 0`). */
        FLAT_SINGLE_STREAM,
    }

    data class BranchInput(
        val rawLabel: String,
        val messages: List<JsonObject>,
    )

    private val prettyJson = Json { prettyPrint = false }

    /**
     * Sanitizes a display label so it is safe inside MoeTalkB's `TOP_replyEdit()` HTML template
     * and never collides with root (`"0"`) or empty strings.
     */
    fun sanitizeBranchName(raw: String, fallbackIndex: Int): String {
        val cleaned = raw
            .replace("\r", " ")
            .replace("\n", " ")
            .replace("\"", "＂")
            .replace("'", "＇")
            .replace("<", "＜")
            .replace(">", "＞")
            .replace("&", "＆")
            .trim()
        if (cleaned.isEmpty() || cleaned == "0") {
            return (fallbackIndex + 1).toString()
        }
        return cleaned
    }

    /**
     * Produces unique, HTML-safe branch labels for a sequence of images / PDF pages.
     */
    fun allocateUniqueBranchNames(rawLabels: List<String>): List<String> {
        val used = HashSet<String>(rawLabels.size * 2)
        val out = ArrayList<String>(rawLabels.size)
        for ((idx, raw) in rawLabels.withIndex()) {
            val base = sanitizeBranchName(raw, idx)
            var candidate = base
            var suffix = 2
            while (!used.add(candidate)) {
                candidate = "$base ($suffix)"
                suffix++
            }
            out.add(candidate)
        }
        return out
    }

    /**
     * Builds a complete MoeTalkB archive [JsonObject] from one or more per-item message lists.
     */
    fun packArchive(
        items: List<BranchInput>,
        title: String = "无题",
        nickname: String = "MomoTranScript",
        date: String = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")),
        mode: MultiItemPackMode = MultiItemPackMode.BRANCH_PER_ITEM,
        tempChars: JsonObject = buildJsonObject {},
        tempImages: JsonObject = buildJsonObject {},
    ): JsonObject {
        val useBranches = mode == MultiItemPackMode.BRANCH_PER_ITEM && items.size > 1
        val chatList = ArrayList<JsonObject>()

        if (useBranches) {
            val branchNames = allocateUniqueBranchNames(items.map { it.rawLabel })
            // 1. Root directory cards (replyDepth = 0) matching MoeTalkB's branch entry UI.
            for (branchName in branchNames) {
                chatList.add(
                    buildJsonObject {
                        put("type", "reply")
                        put("replyDepth", 0)
                        put(
                            "sCharacter",
                            buildJsonObject {
                                put("no", "0")
                                put("index", "1")
                            },
                        )
                        put("content", branchName)
                    },
                )
            }
            // 2. Per-item branch messages (replyDepth = "<branchName>").
            for ((idx, item) in items.withIndex()) {
                val branchName = branchNames[idx]
                for (msg in item.messages) {
                    chatList.add(withReplyDepth(msg, JsonPrimitive(branchName)))
                }
            }
        } else {
            for (item in items) {
                for (msg in item.messages) {
                    chatList.add(withReplyDepth(msg, JsonPrimitive(0)))
                }
            }
        }

        return buildJsonObject {
            put("MoeTalk", MOETALK_ARCHIVE_VERSION)
            put(
                "INFO",
                buildJsonObject {
                    put("title", title.ifBlank { "无题" })
                    put("nickname", nickname.ifBlank { "无名" })
                    put("date", date)
                },
            )
            put("SETTING", defaultSettings())
            put(
                "TEMP",
                buildJsonObject {
                    put("CHAR", tempChars)
                    put("IMAGE", tempImages)
                },
            )
            put("CHAT", JsonArray(chatList))
            put("ERROR", "[]")
        }
    }

    fun encodeToJsonString(archive: JsonObject): String = prettyJson.encodeToString(JsonObject.serializer(), archive)

    private fun withReplyDepth(msg: JsonObject, depth: JsonPrimitive): JsonObject =
        buildJsonObject {
            for ((k, v) in msg) {
                if (k != "replyDepth") {
                    put(k, v)
                }
            }
            put("replyDepth", depth)
        }

    private fun defaultSettings(): JsonObject = buildJsonObject {
        put("顶部标题", "MoeTalk")
        put("宽度限制", 500)
        put("人物改名", buildJsonObject {})
        put("社团列表", buildJsonObject {})
        put("截图选项", buildJsonObject {})
        put("右侧发言", buildJsonObject {})
        put("表情信息", buildJsonObject {})
        put(
            "选择角色",
            buildJsonObject {
                put("no", "0")
                put("index", "1")
                put("list", buildJsonArray {})
            },
        )
        put(
            "风格样式",
            buildJsonObject {
                put("bgColor", "transparent")
                put("info", "background-color:rgb(220, 229, 232);")
            },
        )
    }
}
