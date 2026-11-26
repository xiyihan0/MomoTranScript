package dev.momotranscript.app.input

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader
import com.tom_roush.pdfbox.io.MemoryUsageSetting
import com.tom_roush.pdfbox.pdmodel.PDDocument
import com.tom_roush.pdfbox.pdmodel.PDPage
import com.tom_roush.pdfbox.pdmodel.graphics.image.PDImageXObject
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

data class PdfImageMeta(
    val width: Int,
    val height: Int,
    val suffix: String,
)

data class PdfPagePlan(
    val pageIndex: Int,
    val images: List<PdfImageMeta>,
    val needsRendererFallback: Boolean,
)

data class PdfInspectionResult(
    val pageCount: Int,
    val pages: List<PdfPagePlan>,
    val embeddedImageCount: Int,
    val fallbackPageCount: Int,
)

/**
 * Streaming embedded-image PDF inspector & lossless XObject extractor powered by Apache PDFBox
 * Android (`com.tom-roush:pdfbox-android`), replacing hundreds of lines of custom PDF parsing.
 *
 * Uses `MemoryUsageSetting.setupMixed(4 MiB)` so 200+ MiB PDFs stay bounded in RAM, and
 * automatically handles standard & chained filters (`/DCTDecode`, `[/FlateDecode /DCTDecode]`,
 * `/FlateDecode` + `/SMask`, `/JPXDecode`, inherited `/Resources`, and empty-password PDFs).
 */
object PdfDocumentExtractor {
    private val initialized = AtomicBoolean(false)

    fun ensureInitialized(context: Context) {
        if (initialized.compareAndSet(false, true)) {
            PDFBoxResourceLoader.init(context.applicationContext)
        }
    }

    private fun memorySetting(): MemoryUsageSetting =
        MemoryUsageSetting.setupMixed(4L * 1024L * 1024L)

    /**
     * Inspects a local PDF [File] (also usable in pure JVM unit tests).
     */
    fun inspect(file: File): PdfInspectionResult {
        PDDocument.load(file, memorySetting()).use { doc ->
            return inspectDocument(doc)
        }
    }

    /**
     * Inspects a SAF [Uri] without decoding any pixel data.
     */
    fun inspect(context: Context, uri: Uri): PdfInspectionResult {
        ensureInitialized(context)
        context.contentResolver.openInputStream(uri)?.use { stream ->
            PDDocument.load(stream, memorySetting()).use { doc ->
                return inspectDocument(doc)
            }
        }
        return PdfInspectionResult(0, emptyList(), 0, 0)
    }

    /**
     * Opens a streaming [PDDocument] for page-by-page XObject bitmap extraction.
     */
    fun openDocument(context: Context, uri: Uri): PDDocument? {
        ensureInitialized(context)
        val stream = context.contentResolver.openInputStream(uri) ?: return null
        return stream.use { PDDocument.load(it, memorySetting()) }
    }

    /**
     * Extracts all embedded `PDImageXObject` bitmaps from a single [PDPage] at native resolution.
     * Caller owns the returned [Bitmap] instances and must recycle them after use.
     */
    fun extractPageImages(page: PDPage): List<Bitmap> {
        val resources = page.resources ?: return emptyList()
        val out = ArrayList<Bitmap>()
        for (name in resources.xObjectNames) {
            if (!resources.isImageXObject(name)) continue
            val xobj = runCatching { resources.getXObject(name) as? PDImageXObject }.getOrNull() ?: continue
            runCatching { xobj.image }.getOrNull()?.let { out.add(it) }
        }
        return out
    }

    private fun inspectDocument(doc: PDDocument): PdfInspectionResult {
        val pages = ArrayList<PdfPagePlan>(doc.numberOfPages)
        var embeddedCount = 0
        var fallbackCount = 0

        for (i in 0 until doc.numberOfPages) {
            val page = doc.getPage(i)
            val resources = page.resources
            val imgs = ArrayList<PdfImageMeta>()
            if (resources != null) {
                for (name in resources.xObjectNames) {
                    if (!resources.isImageXObject(name)) continue
                    val xobj = runCatching { resources.getXObject(name) as? PDImageXObject }.getOrNull() ?: continue
                    if (xobj.width > 0 && xobj.height > 0) {
                        imgs.add(PdfImageMeta(xobj.width, xobj.height, xobj.suffix ?: "img"))
                        embeddedCount++
                    }
                }
            }
            val needsFallback = imgs.isEmpty()
            if (needsFallback) fallbackCount++
            pages.add(PdfPagePlan(pageIndex = i, images = imgs, needsRendererFallback = needsFallback))
        }

        return PdfInspectionResult(
            pageCount = pages.size,
            pages = pages,
            embeddedImageCount = embeddedCount,
            fallbackPageCount = fallbackCount,
        )
    }
}
