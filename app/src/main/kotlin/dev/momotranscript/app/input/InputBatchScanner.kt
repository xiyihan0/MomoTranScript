package dev.momotranscript.app.input

import android.content.Context
import android.graphics.BitmapFactory
import android.net.Uri
import android.provider.DocumentsContract
import android.provider.OpenableColumns
import dev.momotranscript.app.nativecore.NativeCore

sealed interface BatchInputSummary {
    data class Images(
        val items: List<ImageInputItem>,
        val folderName: String? = null,
    ) : BatchInputSummary

    data class PdfDocument(
        val uri: Uri,
        val displayName: String,
        val pageCount: Int,
        val directImageCount: Int,
        val fallbackPageCount: Int,
        val estimatedTotalWindows: Int,
        val firstPageSize: String?,
    ) : BatchInputSummary
}

data class ImageInputItem(
    val uri: Uri,
    val displayName: String,
    val width: Int,
    val height: Int,
    val plannedWindows: Int,
)

object InputBatchScanner {

    fun queryDisplayName(context: Context, uri: Uri): String {
        context.contentResolver.query(
            uri,
            arrayOf(OpenableColumns.DISPLAY_NAME),
            null,
            null,
            null,
        )?.use { cursor ->
            if (cursor.moveToFirst()) {
                val idx = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (idx >= 0) {
                    cursor.getString(idx)?.takeIf { it.isNotBlank() }?.let { return it }
                }
            }
        }
        return uri.lastPathSegment?.substringAfterLast('/') ?: "未命名文件"
    }

    fun probeImages(context: Context, uris: List<Uri>, folderName: String? = null): BatchInputSummary.Images {
        val items = uris.mapNotNull { uri ->
            val name = queryDisplayName(context, uri)
            val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            runCatching {
                context.contentResolver.openInputStream(uri)?.use { stream ->
                    BitmapFactory.decodeStream(stream, null, opts)
                }
            }
            if (opts.outWidth <= 0 || opts.outHeight <= 0) return@mapNotNull null
            val windows = NativeCore.planFixedWindows(opts.outWidth, opts.outHeight)
            ImageInputItem(
                uri = uri,
                displayName = name,
                width = opts.outWidth,
                height = opts.outHeight,
                plannedWindows = windows.size.coerceAtLeast(1),
            )
        }
        return BatchInputSummary.Images(items = items, folderName = folderName)
    }

    /**
     * Walks a SAF folder tree (`OpenDocumentTree`), collects all supported image files,
     * and orders them using [NaturalSort] (`2-2.png` before `2-10.png`).
     */
    fun scanFolderTree(context: Context, treeUri: Uri): BatchInputSummary.Images {
        val rootDocId = DocumentsContract.getTreeDocumentId(treeUri)
        val folderName = rootDocId.substringAfterLast(':').substringAfterLast('/').ifBlank { "所选文件夹" }

        val discovered = ArrayList<Pair<String, Uri>>()
        fun walk(parentDocId: String, prefix: String) {
            val childrenUri = DocumentsContract.buildChildDocumentsUriUsingTree(treeUri, parentDocId)
            val projection = arrayOf(
                DocumentsContract.Document.COLUMN_DOCUMENT_ID,
                DocumentsContract.Document.COLUMN_DISPLAY_NAME,
                DocumentsContract.Document.COLUMN_MIME_TYPE,
            )
            context.contentResolver.query(childrenUri, projection, null, null, null)?.use { cursor ->
                val idCol = cursor.getColumnIndexOrThrow(DocumentsContract.Document.COLUMN_DOCUMENT_ID)
                val nameCol = cursor.getColumnIndexOrThrow(DocumentsContract.Document.COLUMN_DISPLAY_NAME)
                val mimeCol = cursor.getColumnIndexOrThrow(DocumentsContract.Document.COLUMN_MIME_TYPE)
                val subDirs = ArrayList<Pair<String, String>>()
                while (cursor.moveToNext()) {
                    val docId = cursor.getString(idCol) ?: continue
                    val name = cursor.getString(nameCol) ?: continue
                    val mime = cursor.getString(mimeCol) ?: ""
                    val relPath = if (prefix.isEmpty()) name else "$prefix/$name"
                    if (mime == DocumentsContract.Document.MIME_TYPE_DIR) {
                        subDirs.add(docId to relPath)
                    } else if (mime.startsWith("image/") || NaturalSort.isSupportedImageName(name)) {
                        val docUri = DocumentsContract.buildDocumentUriUsingTree(treeUri, docId)
                        discovered.add(relPath to docUri)
                    }
                }
                subDirs.sortWith(compareBy(NaturalSort.comparator) { it.second })
                for ((subId, subPrefix) in subDirs) {
                    walk(subId, subPrefix)
                }
            }
        }

        walk(rootDocId, "")
        discovered.sortWith(compareBy(NaturalSort.comparator) { it.first })

        // Fast header probe for up to 300 images (or lightweight placeholder when huge)
        val items = discovered.map { (relPath, uri) ->
            val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            runCatching {
                context.contentResolver.openInputStream(uri)?.use { stream ->
                    BitmapFactory.decodeStream(stream, null, opts)
                }
            }
            val w = opts.outWidth.coerceAtLeast(0)
            val h = opts.outHeight.coerceAtLeast(0)
            val planned = if (w > 0 && h > 0) {
                NativeCore.planFixedWindows(w, h).size.coerceAtLeast(1)
            } else {
                0
            }
            ImageInputItem(
                uri = uri,
                displayName = relPath,
                width = w,
                height = h,
                plannedWindows = planned,
            )
        }
        return BatchInputSummary.Images(items = items, folderName = folderName)
    }

    /**
     * Streams a PDF via Apache PDFBox Android (`setupMixed(4 MiB)`) to index all
     * embedded image XObjects without rasterizing 20,000+ px pages.
     */
    fun scanPdfUri(context: Context, pdfUri: Uri): BatchInputSummary.PdfDocument {
        val name = queryDisplayName(context, pdfUri)
        val scan = PdfDocumentExtractor.inspect(context, pdfUri)

        var totalWindows = 0
        var firstSize: String? = null
        for (page in scan.pages) {
            if (page.images.isNotEmpty()) {
                for (img in page.images) {
                    if (firstSize == null) {
                        firstSize = "${img.width} × ${img.height} px"
                    }
                    totalWindows += NativeCore.planFixedWindows(img.width, img.height).size.coerceAtLeast(1)
                }
            } else {
                totalWindows += 1
            }
        }

        return BatchInputSummary.PdfDocument(
            uri = pdfUri,
            displayName = name,
            pageCount = scan.pageCount,
            directImageCount = scan.embeddedImageCount,
            fallbackPageCount = scan.fallbackPageCount,
            estimatedTotalWindows = totalWindows,
            firstPageSize = firstSize,
        )
    }
}
