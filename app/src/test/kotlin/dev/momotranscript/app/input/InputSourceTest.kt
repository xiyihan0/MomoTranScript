package dev.momotranscript.app.input

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.charset.StandardCharsets

class InputSourceTest {

    @Test
    fun naturalSortOrdersMultiSegmentNumbersNaturally() {
        val shuffled = listOf("2-10.png", "2-2.png", "2-1.png", "10-1.png", "1-1.png", "2-02.png")
        val sorted = shuffled.sortedWith(NaturalSort.comparator)
        assertEquals(
            listOf("1-1.png", "2-1.png", "2-2.png", "2-02.png", "2-10.png", "10-1.png"),
            sorted,
        )
    }

    @Test
    fun supportedImageFilterAcceptsCommonFormats() {
        assertTrue(NaturalSort.isSupportedImageName("chat_01.PNG"))
        assertTrue(NaturalSort.isSupportedImageName("page.webp"))
        assertTrue(NaturalSort.isSupportedImageName("shot.avif"))
        assertFalse(NaturalSort.isSupportedImageName("chapter.pdf"))
        assertFalse(NaturalSort.isSupportedImageName("notes.txt"))
        assertTrue(NaturalSort.isPdfName("TrainToSpring.PDF"))
    }

    @Test
    fun pdfBoxExtractorInspectsEmbeddedImagePagesAndFallbackPages() {
        val fakeJpeg1 = byteArrayOf(
            0xFF.toByte(), 0xD8.toByte(), 0xFF.toByte(), 0xE0.toByte(),
            0x11, 0x22, 0x33, 0xFF.toByte(), 0xD9.toByte(),
        )
        val tmp = File.createTempFile("mmt_pdfbox_test_", ".pdf")
        try {
            tmp.writeBytes(buildValidTwoPagePdf(fakeJpeg1))
            val result = PdfDocumentExtractor.inspect(tmp)
            assertEquals(2, result.pageCount)
            assertEquals(1, result.embeddedImageCount)
            assertEquals(1, result.fallbackPageCount)
            assertFalse(result.pages[0].needsRendererFallback)
            assertEquals(494, result.pages[0].images[0].width)
            assertEquals(22000, result.pages[0].images[0].height)
            assertTrue(result.pages[1].needsRendererFallback)
        } finally {
            tmp.delete()
        }
    }

    private fun buildValidTwoPagePdf(jpegBytes: ByteArray): ByteArray {
        val out = ByteArrayOutputStream()
        val offsets = ArrayList<Int>()
        fun w(s: String) = out.write(s.toByteArray(StandardCharsets.ISO_8859_1))

        w("%PDF-1.7\n")
        offsets.add(out.size()); w("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
        offsets.add(out.size()); w("2 0 obj\n<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>\nendobj\n")
        offsets.add(out.size()); w("3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 370 15000] /Resources << /XObject << /Im0 4 0 R >> >> >>\nendobj\n")
        offsets.add(out.size()); w("4 0 obj\n<< /Type /XObject /Subtype /Image /Width 494 /Height 22000 /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length ${jpegBytes.size} >>\nstream\n")
        out.write(jpegBytes)
        w("\nendstream\nendobj\n")
        offsets.add(out.size()); w("5 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 370 600] /Resources << >> >>\nendobj\n")

        val xrefPos = out.size()
        w("xref\n0 ${offsets.size + 1}\n")
        w("0000000000 65535 f \n")
        for (off in offsets) {
            w("%010d 00000 n \n".format(off))
        }
        w("trailer\n<< /Size ${offsets.size + 1} /Root 1 0 R >>\nstartxref\n$xrefPos\n%%EOF\n")
        return out.toByteArray()
    }
}
