package dev.momotranscript.app.avatar

import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets

class NpyReaderTest {

    @Test
    fun halfToFloatMatchesIeee754StandardValues() {
        assertEquals(0.0f, NpyReader.halfToFloat(0x0000), 0.0f)
        assertEquals(1.0f, NpyReader.halfToFloat(0x3C00), 1e-6f)
        assertEquals(-2.0f, NpyReader.halfToFloat(0xC000), 1e-6f)
        assertEquals(0.5f, NpyReader.halfToFloat(0x3800), 1e-6f)
        assertEquals(65504.0f, NpyReader.halfToFloat(0x7BFF), 1e-2f)
        // Smallest positive subnormal: 2^-24
        assertEquals(5.9604645e-8f, NpyReader.halfToFloat(0x0001), 1e-12f)
    }

    @Test
    fun parsesFloat16MatrixNpyCleanly() {
        val header = "{'descr': '<f2', 'fortran_order': False, 'shape': (2, 2), }   \n"
        val out = ByteArrayOutputStream()
        out.write(byteArrayOf(0x93.toByte(), 'N'.code.toByte(), 'U'.code.toByte(), 'M'.code.toByte(), 'P'.code.toByte(), 'Y'.code.toByte(), 1, 0))
        val lenBuf = ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN).putShort(header.length.toShort())
        out.write(lenBuf.array())
        out.write(header.toByteArray(StandardCharsets.ISO_8859_1))

        val payload = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
            .putShort(0x3C00.toShort()) // 1.0
            .putShort(0x3800.toShort()) // 0.5
            .putShort(0xC000.toShort()) // -2.0
            .putShort(0x0000.toShort()) // 0.0
        out.write(payload.array())

        val mat = NpyReader.parseNpyFloatMatrix(out.toByteArray())
        assertEquals(2, mat.rows)
        assertEquals(2, mat.cols)
        assertEquals(1.0f, mat.data[0], 1e-6f)
        assertEquals(0.5f, mat.data[1], 1e-6f)
        assertEquals(-2.0f, mat.data[2], 1e-6f)
        assertEquals(0.0f, mat.data[3], 1e-6f)
    }
}
