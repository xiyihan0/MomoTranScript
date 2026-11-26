package dev.momotranscript.app.avatar

import android.graphics.Bitmap
import org.aomedia.avif.android.AvifDecoder
import java.nio.ByteBuffer

/**
 * Decodes single-frame `.avif` and multi-frame 4:4:4 `.avifs` sequences sliced from `MMTAVIF1`
 * using the official AOMedia `libavif` + `libdav1d` Android library (`org.aomedia.avif.android:avif`),
 * guaranteeing consistent AV1 High-Profile (yuv444p) decoding across all Android 8–16 devices.
 */
object AvifsFrameDecoder {

    fun decodeSequenceRgb(
        avifsBytes: ByteArray,
        expectedCount: Int,
        tplSize: Int,
        threads: Int = Runtime.getRuntime().availableProcessors().coerceIn(2, 4),
    ): ByteArray? {
        if (avifsBytes.isEmpty() || expectedCount <= 0 || tplSize <= 0) return null

        val direct = ByteBuffer.allocateDirect(avifsBytes.size)
        direct.put(avifsBytes)
        direct.flip()

        val decoder = AvifDecoder.create(direct, threads) ?: return null
        val frameW = decoder.width
        val frameH = decoder.height
        if (frameW <= 0 || frameH <= 0 || decoder.frameCount < expectedCount) {
            decoder.release()
            return null
        }

        val frameSize = tplSize * tplSize * 3
        val out = ByteArray(expectedCount * frameSize)
        val bmp = Bitmap.createBitmap(frameW, frameH, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(tplSize * tplSize)

        try {
            for (i in 0 until expectedCount) {
                val rc = decoder.nextFrame(bmp)
                if (rc != 0) return null

                val target = if (frameW == tplSize && frameH == tplSize) {
                    bmp
                } else {
                    Bitmap.createScaledBitmap(bmp, tplSize, tplSize, true)
                }
                target.getPixels(pixels, 0, tplSize, 0, 0, tplSize, tplSize)
                if (target !== bmp) target.recycle()

                var dst = i * frameSize
                for (px in pixels) {
                    out[dst++] = ((px ushr 16) and 0xFF).toByte()
                    out[dst++] = ((px ushr 8) and 0xFF).toByte()
                    out[dst++] = (px and 0xFF).toByte()
                }
            }
            return out
        } finally {
            bmp.recycle()
            decoder.release()
        }
    }
}
