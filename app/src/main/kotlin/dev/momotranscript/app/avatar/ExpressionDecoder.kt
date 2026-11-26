package dev.momotranscript.app.avatar

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File
import kotlin.math.exp

@Serializable
data class ExpressionTagEntry(
    val id: Int,
    val tag: String,
    val category: String = "",
    val zh: String = "",
    @SerialName("recommended_threshold") val recommendedThreshold: Float = 0.5f,
)

@Serializable
data class ExpressionVocabFile(
    val version: String = "v2",
    @SerialName("num_tags") val numTags: Int = 0,
    val tags: List<ExpressionTagEntry> = emptyList(),
)

/**
 * Decodes 48-class micro-expression logits from `AnimeFace-Unified-PE-Core-S16` into concise
 * Chinese captions (e.g. `闭眼/微笑/脸红`) and annotates `[image]` lines in MomoScript output.
 */
class ExpressionDecoder(private val entries: List<ExpressionTagEntry>) {

    data class Hit(
        val tag: String,
        val zh: String,
        val category: String,
        val score: Float,
    )

    fun decodeCaption(logits: FloatArray?, maxTags: Int = 3): String {
        if (logits == null || logits.isEmpty() || entries.isEmpty()) return ""
        val hits = ArrayList<Hit>()
        for (e in entries) {
            if (e.id !in logits.indices) continue
            val lg = logits[e.id].coerceIn(-30f, 30f)
            val prob = (1.0 / (1.0 + exp(-lg.toDouble()))).toFloat()
            if (prob >= e.recommendedThreshold && e.tag != "closed_mouth") {
                hits.add(Hit(e.tag, e.zh.ifBlank { e.tag }, e.category, prob))
            }
        }
        if (hits.isEmpty()) return ""
        hits.sortWith(compareByDescending<Hit> { it.score }.thenBy { categoryPriority(it.category) })

        val deduped = ArrayList<Hit>(maxTags)
        val seenGroups = HashSet<String>()
        for (h in hits) {
            val grp = DEDUP_GROUPS[h.tag] ?: h.tag
            if (seenGroups.add(grp)) {
                deduped.add(h)
                if (deduped.size >= maxTags) break
            }
        }
        deduped.sortWith(compareBy<Hit> { categoryPriority(it.category) }.thenByDescending { it.score })
        return deduped.joinToString("/") { it.zh }
    }

    companion object {
        private val json = Json { ignoreUnknownKeys = true }

        private val DEDUP_GROUPS = mapOf(
            "sweatdrop" to "sweat",
            "sweat" to "sweat",
            "tears" to "tears",
            "crying" to "tears",
            "tearing_up" to "tears",
            "blush" to "blush",
            "full-face_blush" to "blush",
            "smile" to "smile",
            "light_smile" to "smile",
            "grin" to "smile",
            ":d" to "smile",
            "open_mouth" to "mouth_open",
            "parted_lips" to "mouth_open",
            ":o" to "mouth_open",
            "closed_eyes" to "eyes_closed",
            "^_^" to "eyes_closed",
            ">_<" to "eyes_closed",
            "gloom_(expression)" to "gloom",
            "shaded_face" to "gloom",
            "frown" to "brow",
            "furrowed_brow" to "brow",
            "worried" to "anxious",
            "nervous" to "anxious",
        )

        private fun categoryPriority(cat: String): Int = when (cat) {
            "emotions" -> 0
            "effects" -> 1
            "eyes" -> 2
            "mouth" -> 3
            else -> 9
        }

        fun loadIfPresent(vocabFile: File): ExpressionDecoder? {
            if (!vocabFile.isFile || vocabFile.length() == 0L) return null
            return runCatching {
                val parsed = json.decodeFromString<ExpressionVocabFile>(vocabFile.readText(Charsets.UTF_8))
                if (parsed.tags.isNotEmpty()) ExpressionDecoder(parsed.tags) else null
            }.getOrNull()
        }

        fun formatImageMarker(vm: InsertVisualMatch?, hasSpeakerPrefix: Boolean): String {
            if (vm == null) return "[image]"
            val expr = vm.expressionSummary.trim()
            return when {
                vm.kind == "charface" && vm.status == "matched" -> {
                    val parts = ArrayList<String>(2)
                    if (vm.character.isNotBlank() && !hasSpeakerPrefix) parts.add(vm.character)
                    if (expr.isNotBlank()) parts.add(expr)
                    if (parts.isNotEmpty()) "[image: ${parts.joinToString(" · ")}]" else "[image]"
                }
                vm.kind == "emoji" && vm.status == "matched" -> {
                    if (expr.isNotBlank()) "[image: 表情包 · $expr]" else "[image: 表情包]"
                }
                expr.isNotBlank() -> "[image: $expr]"
                else -> "[image]"
            }
        }

        /**
         * Replaces `[image]` placeholders in page MomoScript text using the ordered list of
         * `InsertVisualMatch?` corresponding to each `insert_image` / `bond` message in reading order.
         */
        fun annotateMomoScriptImageLines(
            momoScript: String,
            imageMatchesInOrder: List<InsertVisualMatch?>,
        ): String {
            if (imageMatchesInOrder.isEmpty() || !momoScript.contains("[image]")) return momoScript
            val trailingNewline = momoScript.endsWith('\n')
            val lines = momoScript.trimEnd('\n').split('\n').toMutableList()
            var idx = 0
            var changed = false
            for (li in lines.indices) {
                if (idx >= imageMatchesInOrder.size) break
                val line = lines[li]
                if (!line.contains("[image]")) continue
                val vm = imageMatchesInOrder[idx++]
                val hasPrefix = line.startsWith("> ") || line.startsWith("< ")
                val marker = formatImageMarker(vm, hasSpeakerPrefix = hasPrefix)
                if (marker != "[image]") {
                    lines[li] = line.replaceFirst("[image]", marker)
                    changed = true
                }
            }
            if (!changed) return momoScript
            return lines.joinToString("\n") + if (trailingNewline) "\n" else ""
        }
    }
}
