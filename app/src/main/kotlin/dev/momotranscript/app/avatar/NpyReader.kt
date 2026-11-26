package dev.momotranscript.app.avatar

import java.io.File
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets
import java.util.zip.ZipFile

data class FloatMatrix(
    val rows: Int,
    val cols: Int,
    val data: FloatArray,
)

/**
 * Pure-Kotlin reader for NumPy `.npy` (v1/v2/v3) and `.npz` archives containing C-contiguous
 * little-endian `float16` (`<f2`) or `float32` (`<f4`) 2D matrices (such as `stage1_embeddings.npz`).
 */
object NpyReader {

    fun loadNpzMatrices(file: File): Map<String, FloatMatrix> {
        val out = LinkedHashMap<String, FloatMatrix>()
        ZipFile(file).use { zf ->
            val entries = zf.entries()
            while (entries.hasMoreElements()) {
                val entry = entries.nextElement()
                if (!entry.name.endsWith(".npy")) continue
                val key = entry.name.removeSuffix(".npy")
                zf.getInputStream(entry).use { stream ->
                    out[key] = parseNpyFloatMatrix(stream)
                }
            }
        }
        return out
    }

    fun parseNpyFloatMatrix(stream: InputStream): FloatMatrix {
        val bytes = stream.readBytes()
        return parseNpyFloatMatrix(bytes)
    }

    fun parseNpyFloatMatrix(bytes: ByteArray): FloatMatrix {
        require(bytes.size >= 10 && bytes[0] == 0x93.toByte() &&
            bytes[1] == 'N'.code.toByte() && bytes[2] == 'U'.code.toByte() &&
            bytes[3] == 'M'.code.toByte() && bytes[4] == 'P'.code.toByte() &&
            bytes[5] == 'Y'.code.toByte()
        ) { "Not a valid NumPy .npy payload" }

        val major = bytes[6].toInt() and 0xFF
        val headerLen: Int
        val headerOffset: Int
        if (major == 1) {
            headerLen = ((bytes[8].toInt() and 0xFF) or ((bytes[9].toInt() and 0xFF) shl 8))
            headerOffset = 10
        } else {
            val bb = ByteBuffer.wrap(bytes, 8, 4).order(ByteOrder.LITTLE_ENDIAN)
            headerLen = bb.int
            headerOffset = 12
        }

        val headerStr = String(bytes, headerOffset, headerLen, StandardCharsets.ISO_8859_1)
        val descr = DESCR_REGEX.find(headerStr)?.groupValues?.get(1)
            ?: error("Missing dtype descr in .npy header: $headerStr")
        val shapeBody = SHAPE_REGEX.find(headerStr)?.groupValues?.get(1)
            ?: error("Missing shape in .npy header: $headerStr")
        val dims = shapeBody.split(',').mapNotNull { it.trim().toIntOrNull() }
        val rows = dims.getOrElse(0) { 0 }
        val cols = if (dims.size >= 2) dims[1] else 1
        val total = rows * cols
        val dataOffset = headerOffset + headerLen

        val fortranOrder = headerStr.contains("'fortran_order': True") || headerStr.contains("\"fortran_order\": True")
        val out = FloatArray(total)
        val bb = ByteBuffer.wrap(bytes, dataOffset, bytes.size - dataOffset).order(ByteOrder.LITTLE_ENDIAN)
        when (descr) {
            "<f2" -> {
                if (fortranOrder && rows > 0 && cols > 1) {
                    for (c in 0 until cols) {
                        for (r in 0 until rows) {
                            out[r * cols + c] = halfToFloat(bb.short.toInt() and 0xFFFF)
                        }
                    }
                } else {
                    for (i in 0 until total) {
                        out[i] = halfToFloat(bb.short.toInt() and 0xFFFF)
                    }
                }
            }
            "<f4" -> {
                if (fortranOrder && rows > 0 && cols > 1) {
                    for (c in 0 until cols) {
                        for (r in 0 until rows) {
                            out[r * cols + c] = bb.float
                        }
                    }
                } else {
                    for (i in 0 until total) {
                        out[i] = bb.float
                    }
                }
            }
            else -> error("Unsupported .npy dtype: $descr")
        }
        return FloatMatrix(rows = rows, cols = cols, data = out)
    }

    /**
     * Exact IEEE-754 binary16 -> binary32 conversion.
     */
    fun halfToFloat(bits: Int): Float {
        val sign = (bits ushr 15) and 0x1
        val exp = (bits ushr 10) and 0x1F
        val mant = bits and 0x3FF

        val fBits = when (exp) {
            0 -> {
                if (mant == 0) {
                    sign shl 31
                } else {
                    // Subnormal float16 -> normalized float32
                    var m = mant
                    var e = -14
                    while ((m and 0x400) == 0) {
                        m = m shl 1
                        e--
                    }
                    m = m and 0x3FF
                    (sign shl 31) or ((e + 127) shl 23) or (m shl 13)
                }
            }
            31 -> {
                (sign shl 31) or (0xFF shl 23) or (mant shl 13)
            }
            else -> {
                (sign shl 31) or ((exp - 15 + 127) shl 23) or (mant shl 13)
            }
        }
        return java.lang.Float.intBitsToFloat(fBits)
    }

    private val DESCR_REGEX = Regex("'descr'\\s*:\\s*'([^']+)'")
    private val SHAPE_REGEX = Regex("'shape'\\s*:\\s*\\(([^)]*)\\)")
}
