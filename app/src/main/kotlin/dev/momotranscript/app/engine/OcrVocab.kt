package dev.momotranscript.app.engine

import java.io.File

data class CtcLineResult(
    val text: String,
    val score: Float,
    val minCharScore: Float,
)

/**
 * Parses PP-OCRv6 `ocr_vocab.yml` (`PostProcess.character_dict`) without external YAML libraries
 * and performs exact greedy CTC decoding matching `transcript.py:_decode_onnx_ctc`.
 */
class OcrVocab(val charList: List<String>) {

    fun decodeCtc(indices: LongArray, scores: FloatArray): CtcLineResult {
        val sb = StringBuilder()
        var scoreSum = 0.0
        var count = 0
        var minChar = 1.0f
        var prev = -1L

        val len = minOf(indices.size, scores.size)
        for (i in 0 until len) {
            val idx = indices[i]
            if (idx != 0L && idx != prev && idx > 0L && idx < charList.size.toLong()) {
                sb.append(charList[idx.toInt()])
                val sc = scores[i].coerceIn(0.0f, 1.0f)
                scoreSum += sc
                count++
                if (sc < minChar) minChar = sc
            }
            prev = idx
        }

        if (count == 0) {
            return CtcLineResult(text = "", score = 0.0f, minCharScore = 0.0f)
        }
        val avg = (scoreSum / count).toFloat().coerceIn(0.0f, 1.0f)
        return CtcLineResult(text = sb.toString(), score = avg, minCharScore = minChar)
    }

    fun decodeCtcSlice(indices: IntArray, scores: FloatArray, offset: Int, len: Int): CtcLineResult {
        val sb = StringBuilder()
        var scoreSum = 0.0
        var count = 0
        var minChar = 1.0f
        var prev = -1

        val end = minOf(offset + len, indices.size, scores.size)
        for (i in offset until end) {
            val idx = indices[i]
            if (idx != 0 && idx != prev && idx > 0 && idx < charList.size) {
                sb.append(charList[idx])
                val sc = scores[i].coerceIn(0.0f, 1.0f)
                scoreSum += sc
                count++
                if (sc < minChar) minChar = sc
            }
            prev = idx
        }

        if (count == 0) {
            return CtcLineResult(text = "", score = 0.0f, minCharScore = 0.0f)
        }
        val avg = (scoreSum / count).toFloat().coerceIn(0.0f, 1.0f)
        return CtcLineResult(text = sb.toString(), score = avg, minCharScore = minChar)
    }

    companion object {
        fun fromYamlFile(file: File): OcrVocab = fromYamlText(file.readText(Charsets.UTF_8))

        fun fromYamlText(yamlText: String): OcrVocab {
            val chars = ArrayList<String>(18_500)
            var inDict = false
            var dictIndent = -1

            for (rawLine in yamlText.lineSequence()) {
                val noCr = rawLine.trimEnd('\r')
                val trimmedAscii = noCr.trim { it == ' ' || it == '\t' }
                if (trimmedAscii.isEmpty() || trimmedAscii.startsWith("#")) continue

                val indent = noCr.indexOfFirst { it != ' ' }.let { if (it < 0) 0 else it }
                if (!inDict) {
                    if (trimmedAscii == "character_dict:" || trimmedAscii.startsWith("character_dict:")) {
                        inDict = true
                        dictIndent = indent
                    }
                    continue
                }

                // Exit character_dict when indentation drops back to or below dictIndent and not a list item
                if (indent <= dictIndent && !trimmedAscii.startsWith("-")) {
                    break
                }
                if (trimmedAscii.startsWith("-")) {
                    // Strip ONLY the YAML list dash + single separating ASCII space, preserving Unicode spaces like U+3000.
                    val afterDash = trimmedAscii.substring(1)
                    val valuePart = if (afterDash.startsWith(" ")) afterDash.substring(1) else afterDash
                    chars.add(parseYamlScalar(valuePart))
                }
            }

            // PP-OCRv6 CTC index 0 = blank, 1..N = character_dict, N+1 = space (" ")
            val full = ArrayList<String>(chars.size + 2)
            full.add("blank")
            full.addAll(chars)
            full.add(" ")
            return OcrVocab(full)
        }

        internal fun parseYamlScalar(raw: String): String {
            if (raw.isEmpty() || raw == "null" || raw == "~") return ""
            if (raw.length >= 2 && raw.startsWith("'") && raw.endsWith("'")) {
                return raw.substring(1, raw.length - 1).replace("''", "'")
            }
            if (raw.length >= 2 && raw.startsWith("\"") && raw.endsWith("\"")) {
                val inner = raw.substring(1, raw.length - 1)
                val sb = StringBuilder(inner.length)
                var i = 0
                while (i < inner.length) {
                    val c = inner[i]
                    if (c == '\\' && i + 1 < inner.length) {
                        when (val nxt = inner[i + 1]) {
                            '\\' -> { sb.append('\\'); i += 2 }
                            '"' -> { sb.append('"'); i += 2 }
                            'n' -> { sb.append('\n'); i += 2 }
                            'r' -> { sb.append('\r'); i += 2 }
                            't' -> { sb.append('\t'); i += 2 }
                            'x' -> {
                                if (i + 3 < inner.length) {
                                    val code = inner.substring(i + 2, i + 4).toIntOrNull(16)
                                    if (code != null) { sb.append(code.toChar()); i += 4; continue }
                                }
                                sb.append(nxt); i += 2
                            }
                            'u' -> {
                                if (i + 5 < inner.length) {
                                    val code = inner.substring(i + 2, i + 6).toIntOrNull(16)
                                    if (code != null) { sb.append(code.toChar()); i += 6; continue }
                                }
                                sb.append(nxt); i += 2
                            }
                            else -> { sb.append(nxt); i += 2 }
                        }
                    } else {
                        sb.append(c)
                        i++
                    }
                }
                return sb.toString()
            }
            return raw
        }
    }
}
