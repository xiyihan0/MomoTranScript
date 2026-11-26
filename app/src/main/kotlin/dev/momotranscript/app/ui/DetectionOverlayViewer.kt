package dev.momotranscript.app.ui

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.BitmapRegionDecoder
import android.graphics.Rect
import android.net.Uri
import android.os.Build
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import dev.momotranscript.app.nativecore.NativeDetectedElement
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import java.io.File

@Serializable
data class PagePreviewRecord(
    val label: String,
    val imageUri: String? = null,
    val previewImagePath: String? = null,
    val width: Int,
    val height: Int,
    val elements: List<NativeDetectedElement> = emptyList(),
)

private data class ClassStyle(
    val name: String,
    val color: Color,
)

private val CLASS_STYLES = mapOf(
    0 to ClassStyle("头像", Color(0xFFE91E63)),
    1 to ClassStyle("姓名", Color(0xFF9C27B0)),
    2 to ClassStyle("左气泡", Color(0xFF1E88E5)),
    3 to ClassStyle("右气泡", Color(0xFF00ACC1)),
    4 to ClassStyle("旁白", Color(0xFF607D8B)),
    5 to ClassStyle("插图", Color(0xFFFF9800)),
    6 to ClassStyle("回复框", Color(0xFF43A047)),
    7 to ClassStyle("回复项", Color(0xFF7CB342)),
    8 to ClassStyle("羁绊", Color(0xFFF06292)),
    9 to ClassStyle("截断", Color(0xFFB71C1C)),
    10 to ClassStyle("附注", Color(0xFF795548)),
)

private const val TILE_HEIGHT_PX = 1600

/**
 * Tiled vertical viewer for arbitrarily tall chat screenshots (even 23,000+ px) with
 * synchronized 11-class vector detection box overlay. Decodes only visible 1600px vertical
 * strips on demand via `BitmapRegionDecoder` so OpenGL texture limits are never exceeded.
 */
@Composable
fun DetectionOverlayViewer(
    pages: List<PagePreviewRecord>,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    if (pages.isEmpty()) {
        Box(modifier = modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text(
                text = "完成一次离线转录后，可在此查看原图矢量打框对照",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        return
    }

    val firstPageKey = pages.firstOrNull()?.let { "${it.label}|${it.imageUri}|${it.previewImagePath}" }
    var selectedPageIdx by remember(firstPageKey) { mutableIntStateOf(0) }
    val page = pages[selectedPageIdx.coerceIn(0, pages.lastIndex)]
    var showBoxes by remember { mutableStateOf(true) }

    // Open a region decoder for the currently selected page (closed on page switch / dispose)
    var regionDecoder by remember(page) { mutableStateOf<BitmapRegionDecoder?>(null) }
    LaunchedEffect(page) {
        regionDecoder = withContext(Dispatchers.IO) {
            openRegionDecoder(context, page)
        }
    }
    DisposableEffect(page) {
        onDispose {
            runCatching { regionDecoder?.recycle() }
            regionDecoder = null
        }
    }

    val tileRanges = remember(page.width, page.height) {
        if (page.width <= 0 || page.height <= 0) emptyList()
        else {
            val out = ArrayList<Pair<Int, Int>>()
            var y = 0
            while (y < page.height) {
                val y1 = minOf(page.height, y + TILE_HEIGHT_PX)
                out.add(y to y1)
                y = y1
            }
            out
        }
    }

    Column(modifier = modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        // Page selector chips when batch has multiple images/pages
        if (pages.size > 1) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                pages.forEachIndexed { idx, p ->
                    FilterChip(
                        selected = idx == selectedPageIdx,
                        onClick = { selectedPageIdx = idx },
                        label = { Text("${p.label} (${p.elements.size}框)") },
                    )
                }
            }
        }

        // Toggle + compact class legend row
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .horizontalScroll(rememberScrollState()),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            FilterChip(
                selected = showBoxes,
                onClick = { showBoxes = !showBoxes },
                label = { Text(if (showBoxes) "隐藏框 (${page.elements.size})" else "显示框 (${page.elements.size})") },
            )
            val presentClasses = remember(page.elements) {
                page.elements.map { it.classId }.distinct().sorted()
            }
            presentClasses.forEach { cid ->
                val st = CLASS_STYLES[cid] ?: ClassStyle("类$cid", Color.Gray)
                Surface(
                    shape = RoundedCornerShape(999.dp),
                    color = st.color.copy(alpha = 0.14f),
                    border = BorderStroke(1.dp, st.color.copy(alpha = 0.5f)),
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(4.dp),
                    ) {
                        Box(
                            modifier = Modifier
                                .size(8.dp)
                                .clip(CircleShape)
                                .background(st.color),
                        )
                        Text(
                            text = st.name,
                            style = MaterialTheme.typography.labelSmall,
                            color = st.color,
                            fontWeight = FontWeight.Medium,
                        )
                    }
                }
            }
        }

        if (regionDecoder == null) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    text = "正在加载原图分块预览 (${page.width}×${page.height})…",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        } else {
            val dec = regionDecoder!!
            BoxWithConstraints(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
                    .clip(RoundedCornerShape(16.dp))
                    .background(MaterialTheme.colorScheme.surfaceContainer),
            ) {
                val viewWidthPx = with(LocalDensity.current) { maxWidth.toPx() }.coerceAtLeast(1f)
                val scale = viewWidthPx / page.width.coerceAtLeast(1).toFloat()

                LazyColumn(modifier = Modifier.fillMaxSize()) {
                    items(tileRanges, key = { "${page.label}_${it.first}_${it.second}" }) { (y0, y1) ->
                        val tileH = y1 - y0
                        val tileHeightDp = with(LocalDensity.current) { (tileH * scale).toDp() }
                        val tileElements = remember(page.elements, y0, y1) {
                            page.elements.filter { el ->
                                el.box.size == 4 && el.box[3] > y0 && el.box[1] < y1
                            }
                        }

                        var tileBmp by remember(dec, y0, y1) { mutableStateOf<Bitmap?>(null) }
                        LaunchedEffect(dec, y0, y1) {
                            tileBmp = withContext(Dispatchers.IO) {
                                runCatching {
                                    synchronized(dec) {
                                        dec.decodeRegion(Rect(0, y0, page.width, y1), BitmapFactory.Options())
                                    }
                                }.getOrNull()
                            }
                        }
                        DisposableEffect(dec, y0, y1) {
                            onDispose {
                                tileBmp?.recycle()
                                tileBmp = null
                            }
                        }

                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(tileHeightDp),
                        ) {
                            tileBmp?.let { bmp ->
                                Image(
                                    bitmap = bmp.asImageBitmap(),
                                    contentDescription = null,
                                    contentScale = ContentScale.FillBounds,
                                    modifier = Modifier.fillMaxSize(),
                                )
                            }
                            if (showBoxes && tileElements.isNotEmpty()) {
                                Canvas(modifier = Modifier.fillMaxSize()) {
                                    val sx = size.width / page.width.coerceAtLeast(1).toFloat()
                                    val sy = size.height / tileH.coerceAtLeast(1).toFloat()
                                    val strokePx = 2.dp.toPx()

                                    clipRect {
                                        for (el in tileElements) {
                                            val st = CLASS_STYLES[el.classId] ?: ClassStyle("?", Color.Magenta)
                                            val bx0 = el.box[0] * sx
                                            val by0 = (el.box[1] - y0) * sy
                                            val bx1 = el.box[2] * sx
                                            val by1 = (el.box[3] - y0) * sy
                                            if (bx1 <= bx0 || by1 <= by0) continue

                                            // Draw using unclamped tile-relative coordinates inside clipRect
                                            // so boxes spanning a 1600px tile boundary never get fake seam lines!
                                            drawRect(
                                                color = st.color.copy(alpha = 0.12f),
                                                topLeft = Offset(bx0, by0),
                                                size = Size(bx1 - bx0, by1 - by0),
                                            )
                                            drawRect(
                                                color = st.color,
                                                topLeft = Offset(bx0, by0),
                                                size = Size(bx1 - bx0, by1 - by0),
                                                style = Stroke(width = strokePx),
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

private fun openRegionDecoder(context: Context, page: PagePreviewRecord): BitmapRegionDecoder? {
    val filePath = page.previewImagePath
    if (!filePath.isNullOrBlank() && File(filePath).isFile) {
        return runCatching {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                BitmapRegionDecoder.newInstance(filePath)
            } else {
                @Suppress("DEPRECATION")
                BitmapRegionDecoder.newInstance(filePath, false)
            }
        }.getOrNull()
    }
    val uriStr = page.imageUri
    if (!uriStr.isNullOrBlank()) {
        return runCatching {
            context.contentResolver.openInputStream(Uri.parse(uriStr))?.use { stream ->
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                    BitmapRegionDecoder.newInstance(stream)
                } else {
                    @Suppress("DEPRECATION")
                    BitmapRegionDecoder.newInstance(stream, false)
                }
            }
        }.getOrNull()
    }
    return null
}
