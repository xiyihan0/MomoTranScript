package dev.momotranscript.app.ui

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.KeyboardArrowLeft
import androidx.compose.material.icons.automirrored.rounded.KeyboardArrowRight
import androidx.compose.material.icons.rounded.Edit
import androidx.compose.material.icons.rounded.Menu
import androidx.compose.material.icons.rounded.Warning
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.OffsetMapping
import androidx.compose.ui.text.input.TransformedText
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

data class MomoScriptPage(
    val title: String,
    val body: String,
    val hasHeader: Boolean,
)

private val PAGE_HEADER_REGEX = Regex("^===\\s+(.+?)\\s+===$")
private val LEFT_SPEAKER_REGEX = Regex("^>\\s+([^:\\n]{1,48}):(\\s?)(.*)$")
private val INLINE_TOKEN_REGEX = Regex("\\[(image(?:: [^\\]]+)?|unreadable|unknown speaker)\\]")

/**
 * Splits a combined MomoScript document into pages delimited by `=== Title ===` lines.
 */
fun parseMomoScriptPages(fullText: String): List<MomoScriptPage> {
    if (fullText.isEmpty()) {
        return listOf(MomoScriptPage(title = "全文", body = "", hasHeader = false))
    }
    val lines = fullText.split('\n')
    val pages = ArrayList<MomoScriptPage>()
    var currentTitle: String? = null
    var currentHasHeader = false
    val buf = ArrayList<String>()

    fun flush() {
        val t = currentTitle
        if (t != null || buf.isNotEmpty()) {
            val body = buf.joinToString("\n").trimEnd('\n')
            pages.add(MomoScriptPage(title = t ?: "前言", body = body, hasHeader = currentHasHeader))
            buf.clear()
        }
    }

    for (line in lines) {
        val m = PAGE_HEADER_REGEX.matchEntire(line.trim())
        if (m != null) {
            if (currentTitle != null || buf.any { it.isNotBlank() }) {
                flush()
            } else {
                buf.clear()
            }
            currentTitle = m.groupValues[1].trim()
            currentHasHeader = true
        } else {
            buf.add(line)
        }
    }
    flush()

    if (pages.isEmpty()) {
        return listOf(MomoScriptPage(title = "全文", body = fullText, hasHeader = false))
    }
    if (pages.size == 1 && !pages[0].hasHeader) {
        return listOf(MomoScriptPage(title = "全文", body = fullText, hasHeader = false))
    }
    return pages
}

/**
 * Reassembles pages after editing a single page body in-place.
 */
fun replaceMomoScriptPage(fullText: String, pageIndex: Int, newPageBody: String): String {
    val pages = parseMomoScriptPages(fullText).toMutableList()
    if (pageIndex !in pages.indices) return newPageBody
    val old = pages[pageIndex]
    pages[pageIndex] = old.copy(body = newPageBody)
    if (pages.size == 1 && !pages[0].hasHeader) {
        return pages[0].body
    }
    return buildString {
        pages.forEachIndexed { idx, p ->
            if (idx > 0) append("\n\n")
            if (p.hasHeader) {
                append("=== ").append(p.title).append(" ===\n")
            }
            append(p.body.trimEnd('\n'))
        }
        if (isNotEmpty() && !endsWith('\n')) append('\n')
    }
}

private data class MmsPalette(
    val headerText: Color,
    val directiveText: Color,
    val leftSigil: Color,
    val speakerName: Color,
    val rightSigil: Color,
    val rightBg: Color,
    val narrationSigil: Color,
    val narrationText: Color,
    val narrationBg: Color,
    val imageBadge: Color,
    val alertBadge: Color,
    val noteText: Color,
    val lineNumber: Color,
    val bodyText: Color,
)

@Composable
private fun rememberMmsPalette(): MmsPalette {
    val dark = isSystemInDarkTheme()
    val cs = MaterialTheme.colorScheme
    return remember(dark, cs) {
        if (dark) {
            MmsPalette(
                headerText = Color(0xFFF48FB1),
                directiveText = Color(0xFFCE93D8),
                leftSigil = Color(0xFF64B5F6),
                speakerName = Color(0xFFFF80AB),
                rightSigil = Color(0xFF4FC3F7),
                rightBg = Color(0xFF16324A).copy(alpha = 0.45f),
                narrationSigil = Color(0xFFFFCA28),
                narrationText = Color(0xFFB0BEC5),
                narrationBg = Color(0xFF2B2F3A).copy(alpha = 0.55f),
                imageBadge = Color(0xFF4DB6AC),
                alertBadge = Color(0xFFFF7043),
                noteText = Color(0xFF90A4AE),
                lineNumber = Color(0xFF607D8B),
                bodyText = cs.onSurface,
            )
        } else {
            MmsPalette(
                headerText = Color(0xFFC2185B),
                directiveText = Color(0xFF7B1FA2),
                leftSigil = Color(0xFF1E88E5),
                speakerName = Color(0xFFD81B60),
                rightSigil = Color(0xFF0277BD),
                rightBg = Color(0xFFE3F2FD).copy(alpha = 0.72f),
                narrationSigil = Color(0xFFEF6C00),
                narrationText = Color(0xFF546E7A),
                narrationBg = Color(0xFFF1F3F6),
                imageBadge = Color(0xFF00796B),
                alertBadge = Color(0xFFD84315),
                noteText = Color(0xFF78909C),
                lineNumber = Color(0xFF90A4AE),
                bodyText = cs.onSurface,
            )
        }
    }
}

private fun AnnotatedString.Builder.appendBodyWithTokens(text: String, palette: MmsPalette) {
    // Split optional outward message note suffix ` | note`
    val noteIdx = text.lastIndexOf(" | ")
    val mainPart = if (noteIdx > 0) text.substring(0, noteIdx) else text
    val notePart = if (noteIdx > 0) text.substring(noteIdx) else ""

    var cursor = 0
    for (m in INLINE_TOKEN_REGEX.findAll(mainPart)) {
        if (m.range.first > cursor) {
            append(mainPart.substring(cursor, m.range.first))
        }
        val token = m.value
        val color = if (token.startsWith("[image")) palette.imageBadge else palette.alertBadge
        pushStyle(SpanStyle(color = color, fontWeight = FontWeight.SemiBold))
        append(token)
        pop()
        cursor = m.range.last + 1
    }
    if (cursor < mainPart.length) {
        append(mainPart.substring(cursor))
    }
    if (notePart.isNotEmpty()) {
        pushStyle(SpanStyle(color = palette.noteText, fontStyle = FontStyle.Italic))
        append(notePart)
        pop()
    }
}

private fun highlightSingleLine(line: String, palette: MmsPalette): AnnotatedString = buildAnnotatedString {
    val trimmed = line.trimStart()
    when {
        PAGE_HEADER_REGEX.matches(line.trim()) -> {
            pushStyle(SpanStyle(color = palette.headerText, fontWeight = FontWeight.Bold))
            append(line)
            pop()
        }
        trimmed == "@reply" || trimmed == "@end" || trimmed == "@header" -> {
            pushStyle(SpanStyle(color = palette.directiveText, fontWeight = FontWeight.Bold))
            append(line)
            pop()
        }
        trimmed.startsWith("> ") -> {
            val spMatch = LEFT_SPEAKER_REGEX.matchEntire(trimmed)
            if (spMatch != null) {
                pushStyle(SpanStyle(color = palette.leftSigil, fontWeight = FontWeight.Bold))
                append("> ")
                pop()
                val sp = spMatch.groupValues[1]
                val spColor = if (sp == "[unknown speaker]") palette.alertBadge else palette.speakerName
                pushStyle(SpanStyle(color = spColor, fontWeight = FontWeight.Bold))
                append(sp)
                append(":")
                pop()
                append(spMatch.groupValues[2])
                appendBodyWithTokens(spMatch.groupValues[3], palette)
            } else {
                pushStyle(SpanStyle(color = palette.leftSigil.copy(alpha = 0.7f), fontWeight = FontWeight.Bold))
                append("> ")
                pop()
                appendBodyWithTokens(trimmed.removePrefix("> "), palette)
            }
        }
        trimmed.startsWith("< ") -> {
            pushStyle(SpanStyle(color = palette.rightSigil, fontWeight = FontWeight.Bold))
            append("< ")
            pop()
            appendBodyWithTokens(trimmed.removePrefix("< "), palette)
        }
        trimmed.startsWith("- ") -> {
            pushStyle(SpanStyle(color = palette.narrationSigil, fontWeight = FontWeight.Bold))
            append("- ")
            pop()
            pushStyle(SpanStyle(color = palette.narrationText))
            appendBodyWithTokens(trimmed.removePrefix("- "), palette)
            pop()
        }
        else -> appendBodyWithTokens(line, palette)
    }
}

private class MomoScriptVisualTransformation(
    private val palette: MmsPalette,
) : VisualTransformation {
    override fun filter(text: AnnotatedString): TransformedText {
        val src = text.text
        val highlighted = buildAnnotatedString {
            val lines = src.split('\n')
            lines.forEachIndexed { idx, line ->
                if (idx > 0) append('\n')
                append(highlightSingleLine(line, palette))
            }
        }
        return TransformedText(highlighted, OffsetMapping.Identity)
    }
}

/**
 * Syntax-highlighted, paginated MomoScript viewer & per-page editor.
 *
 * - Default **Highlight Reader** mode uses a `LazyColumn` so 100+ page transcripts scroll smoothly.
 * - **Source Edit** mode scopes editing to the selected page (or full document) with live syntax
 *   highlighting, eliminating multi-thousand-line EditText lag.
 */
@Composable
fun MomoScriptViewer(
    scriptText: String,
    onScriptChange: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    val palette = rememberMmsPalette()
    val pages = remember(scriptText) { parseMomoScriptPages(scriptText) }
    val multiPage = pages.size > 1

    // -1 means "全部" (all pages combined); 0..pages.lastIndex means a specific page
    var selectedPageIdx by remember(pages.size) {
        mutableIntStateOf(if (pages.size > 1) 0 else -1)
    }
    LaunchedEffect(pages.size) {
        if (pages.size <= 1) {
            selectedPageIdx = -1
        } else if (selectedPageIdx >= pages.size) {
            selectedPageIdx = pages.lastIndex
        }
    }

    var isEditMode by remember { mutableStateOf(false) }
    var onlyAlerts by remember { mutableStateOf(false) }

    val displayedText = remember(scriptText, pages, selectedPageIdx) {
        if (selectedPageIdx in pages.indices) pages[selectedPageIdx].body else scriptText
    }
    val displayedLines = remember(displayedText) {
        if (displayedText.isEmpty()) emptyList() else displayedText.split('\n')
    }
    val alertCount = remember(displayedLines) {
        displayedLines.count { it.contains("[unreadable]") || it.contains("[unknown speaker]") }
    }
    LaunchedEffect(alertCount) {
        if (alertCount == 0 && onlyAlerts) onlyAlerts = false
    }

    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        // 1. Reader / Editor toggle + Alert filter chip
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            FilterChip(
                selected = !isEditMode,
                onClick = { isEditMode = false },
                label = { Text("高亮阅读", maxLines = 1) },
                leadingIcon = { Icon(Icons.Rounded.Menu, contentDescription = null, modifier = Modifier.size(16.dp)) },
            )
            FilterChip(
                selected = isEditMode,
                onClick = { isEditMode = true },
                label = { Text("源码编辑", maxLines = 1) },
                leadingIcon = { Icon(Icons.Rounded.Edit, contentDescription = null, modifier = Modifier.size(16.dp)) },
            )
            if (!isEditMode && alertCount > 0) {
                FilterChip(
                    selected = onlyAlerts,
                    onClick = { onlyAlerts = !onlyAlerts },
                    label = { Text("待校对 ($alertCount)", maxLines = 1) },
                    leadingIcon = {
                        Icon(
                            Icons.Rounded.Warning,
                            contentDescription = null,
                            tint = palette.alertBadge,
                            modifier = Modifier.size(15.dp),
                        )
                    },
                )
            }
            Spacer(Modifier.weight(1f))
            Text(
                text = "${displayedLines.size} 行",
                style = MaterialTheme.typography.labelSmall,
                fontFamily = FontFamily.Monospace,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        // 2. Pagination bar (shown automatically for multi-page / multi-image transcripts)
        if (multiPage) {
            Surface(
                color = MaterialTheme.colorScheme.surfaceContainerLow,
                shape = RoundedCornerShape(14.dp),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant),
                modifier = Modifier.fillMaxWidth(),
            ) {
                Column(
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 6.dp),
                    verticalArrangement = Arrangement.spacedBy(4.dp),
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween,
                    ) {
                        IconButton(
                            onClick = {
                                selectedPageIdx = when {
                                    selectedPageIdx <= 0 -> 0
                                    else -> selectedPageIdx - 1
                                }
                            },
                            enabled = selectedPageIdx > 0,
                            modifier = Modifier.size(30.dp),
                        ) {
                            Icon(Icons.AutoMirrored.Rounded.KeyboardArrowLeft, contentDescription = "上一页")
                        }
                        Text(
                            text = if (selectedPageIdx in pages.indices) {
                                "${pages[selectedPageIdx].title}（${selectedPageIdx + 1}/${pages.size} 页）"
                            } else {
                                "全部页面（共 ${pages.size} 页）"
                            },
                            style = MaterialTheme.typography.labelMedium,
                            fontWeight = FontWeight.SemiBold,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            modifier = Modifier.padding(horizontal = 6.dp),
                        )
                        IconButton(
                            onClick = {
                                selectedPageIdx = when {
                                    selectedPageIdx < 0 -> 0
                                    selectedPageIdx < pages.lastIndex -> selectedPageIdx + 1
                                    else -> pages.lastIndex
                                }
                            },
                            enabled = selectedPageIdx < pages.lastIndex,
                            modifier = Modifier.size(30.dp),
                        ) {
                            Icon(Icons.AutoMirrored.Rounded.KeyboardArrowRight, contentDescription = "下一页")
                        }
                    }

                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .horizontalScroll(rememberScrollState()),
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                    ) {
                        PagePillChip(
                            label = "全部",
                            selected = selectedPageIdx == -1,
                            onClick = { selectedPageIdx = -1 },
                        )
                        pages.forEachIndexed { idx, pg ->
                            PagePillChip(
                                label = pg.title,
                                selected = selectedPageIdx == idx,
                                onClick = { selectedPageIdx = idx },
                            )
                        }
                    }
                }
            }
        }

        // 3. Main viewer / editor pane
        if (isEditMode) {
            val visualTransformation = remember(palette) { MomoScriptVisualTransformation(palette) }
            OutlinedTextField(
                value = displayedText,
                onValueChange = { updated ->
                    if (selectedPageIdx in pages.indices) {
                        onScriptChange(replaceMomoScriptPage(scriptText, selectedPageIdx, updated))
                    } else {
                        onScriptChange(updated)
                    }
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
                shape = RoundedCornerShape(16.dp),
                textStyle = MaterialTheme.typography.bodyMedium.copy(
                    fontFamily = FontFamily.Monospace,
                    lineHeight = 20.sp,
                ),
                visualTransformation = visualTransformation,
                placeholder = { Text("转录完成后的 MomoScript 将显示在这里…") },
            )
        } else {
            val listState = rememberLazyListState()
            LaunchedEffect(selectedPageIdx, onlyAlerts) {
                listState.scrollToItem(0)
            }
            val visibleRows = remember(displayedLines, onlyAlerts) {
                displayedLines.mapIndexedNotNull { idx, line ->
                    val isAlert = line.contains("[unreadable]") || line.contains("[unknown speaker]")
                    if (!onlyAlerts || isAlert) (idx + 1) to line else null
                }
            }
            Surface(
                color = MaterialTheme.colorScheme.surfaceContainerLowest,
                shape = RoundedCornerShape(16.dp),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant),
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
            ) {
                if (visibleRows.isEmpty()) {
                    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                        Text(
                            text = if (onlyAlerts) "当前页没有待校对异常项" else "暂无转录正文",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                } else {
                    SelectionContainer {
                        LazyColumn(
                            state = listState,
                            contentPadding = PaddingValues(horizontal = 10.dp, vertical = 10.dp),
                            verticalArrangement = Arrangement.spacedBy(3.dp),
                            modifier = Modifier.fillMaxSize(),
                        ) {
                            itemsIndexed(
                                items = visibleRows,
                                key = { _, pair -> "${selectedPageIdx}_${pair.first}" },
                            ) { _, (lineNo, line) ->
                                val trimmed = line.trimStart()
                                val isHeader = PAGE_HEADER_REGEX.matches(line.trim())
                                val isRight = trimmed.startsWith("< ")
                                val isNarration = trimmed.startsWith("- ")
                                val isDirective = trimmed == "@reply" || trimmed == "@end" || trimmed == "@header"
                                val rowBg = when {
                                    isHeader -> MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.45f)
                                    isRight -> palette.rightBg
                                    isNarration -> palette.narrationBg
                                    isDirective -> MaterialTheme.colorScheme.tertiaryContainer.copy(alpha = 0.35f)
                                    else -> Color.Transparent
                                }
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .clip(RoundedCornerShape(8.dp))
                                        .background(rowBg)
                                        .padding(horizontal = 6.dp, vertical = if (isHeader) 5.dp else 3.dp),
                                    verticalAlignment = Alignment.Top,
                                ) {
                                    Text(
                                        text = lineNo.toString().padStart(3, ' '),
                                        style = MaterialTheme.typography.labelSmall,
                                        fontFamily = FontFamily.Monospace,
                                        color = palette.lineNumber,
                                        modifier = Modifier.width(30.dp),
                                    )
                                    Text(
                                        text = highlightSingleLine(line, palette),
                                        style = MaterialTheme.typography.bodyMedium.copy(
                                            fontFamily = FontFamily.Monospace,
                                            lineHeight = 20.sp,
                                        ),
                                        color = palette.bodyText,
                                        modifier = Modifier.weight(1f),
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

@Composable
private fun PagePillChip(
    label: String,
    selected: Boolean,
    onClick: () -> Unit,
) {
    Surface(
        color = if (selected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceContainer,
        shape = RoundedCornerShape(999.dp),
        modifier = Modifier
            .clip(RoundedCornerShape(999.dp))
            .clickable(onClick = onClick),
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal,
            color = if (selected) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurfaceVariant,
            maxLines = 1,
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp),
        )
    }
}
