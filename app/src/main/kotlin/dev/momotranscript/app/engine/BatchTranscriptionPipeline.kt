package dev.momotranscript.app.engine

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.pdf.PdfRenderer
import android.os.Build
import android.util.Base64
import dev.momotranscript.app.avatar.AvatarVisualMatch
import dev.momotranscript.app.avatar.InsertVisualMatch
import dev.momotranscript.app.avatar.OnnxAvatarMatcher
import dev.momotranscript.app.export.MoeTalkBranchPacker
import dev.momotranscript.app.input.BatchInputSummary
import dev.momotranscript.app.input.PdfDocumentExtractor
import dev.momotranscript.app.nativecore.NativeAssembledMessage
import dev.momotranscript.app.nativecore.NativeDetectedElement
import dev.momotranscript.app.nativecore.WindowBox
import dev.momotranscript.app.ui.PagePreviewRecord
import java.io.File
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import kotlin.math.abs
import kotlin.math.roundToInt

data class BatchProgress(
    val completedItems: Int,
    val totalItems: Int,
    val currentLabel: String,
    val detail: String,
)

data class PagePerfRecord(
    val pageNo: Int,
    val totalPages: Int,
    val label: String,
    val width: Int,
    val height: Int,
    val windowCount: Int,
    val lineCount: Int,
    val escalatedLines: Int,
    val adoptedMedium: Int,
    val extractMs: Long,
    val windowPlanMs: Long,
    val detPrepMs: Long,
    val detInferMs: Long,
    val detDecodeMs: Long,
    val cvLinesMs: Long,
    val recSmallMs: Long,
    val recMediumMs: Long,
    val assembleMs: Long,
    val ccipAvTotal: Int,
    val ccipAvUnique: Int,
    val ccipInsTotal: Int,
    val ccipInsUnique: Int,
    val ccipInferMs: Long,
    val ccipStage2Ms: Long,
    val ccipTotalMs: Long,
    val pageTotalMs: Long,
)

data class PartialBatchSnapshot(
    val title: String,
    val completedItems: Int,
    val totalItems: Int,
    val totalWindows: Int,
    val totalLines: Int,
    val matchedOfficialAvatars: Int,
    val matchedVisualAssets: Int,
    val elapsedMs: Long,
    val combinedMomoScript: String,
    val moetalkArchiveJson: String?,
    val usedBranchPacking: Boolean,
    val previewPages: List<PagePreviewRecord>,
    val lastPagePerf: PagePerfRecord,
)

data class BatchTranscriptionOutput(
    val title: String,
    val itemCount: Int,
    val totalWindows: Int,
    val totalLines: Int,
    val matchedOfficialAvatars: Int,
    val matchedVisualAssets: Int,
    val totalElapsedMs: Long,
    val combinedMomoScript: String,
    val moetalkArchiveJson: String,
    val usedBranchPacking: Boolean,
    val previewPages: List<PagePreviewRecord> = emptyList(),
)

/**
 * Streaming batch orchestrator for single/multi images, natural-sorted folders, and huge
 * embedded-image PDFs. Decodes & recycles one image/page at a time so peak RAM stays bounded
 * even for 200+ page fanfic PDFs, runs optional CCIP + Stage-2 visual matching when available,
 * emits per-page partial snapshots for instant live preview/export, and records stage-by-stage
 * timing profiles for on-device debugging.
 */
object BatchTranscriptionPipeline {

    suspend fun run(
        context: Context,
        engine: OnnxTranscriptionEngine,
        input: BatchInputSummary,
        avatarMatcher: OnnxAvatarMatcher? = null,
        packMode: MoeTalkBranchPacker.MultiItemPackMode = MoeTalkBranchPacker.MultiItemPackMode.BRANCH_PER_ITEM,
        previewCacheDir: File? = null,
        onLog: ((String) -> Unit)? = null,
        onPageCompleted: ((PartialBatchSnapshot) -> Unit)? = null,
        onProgress: (BatchProgress) -> Unit,
    ): BatchTranscriptionOutput {
        val t0 = System.nanoTime()
        val branchInputs = ArrayList<MoeTalkBranchPacker.BranchInput>()
        val previewPages = ArrayList<PagePreviewRecord>()
        val scriptBuilder = StringBuilder()

        val customKeyToId = LinkedHashMap<String, String>()
        val tempChars = LinkedHashMap<String, JsonObject>()
        val tempImages = LinkedHashMap<String, String>()
        var customCounter = 0
        var imageCounter = 0

        var totalWindows = 0
        var totalLines = 0
        var matchedOfficialAvatars = 0
        var matchedVisualAssets = 0
        var lastMoeJsonFlushCostMs = 0L
        var pagesSinceMoeJsonFlush = 0

        val tentativeTitle: String = when (input) {
            is BatchInputSummary.Images ->
                input.folderName ?: input.items.firstOrNull()?.displayName?.substringBeforeLast('.') ?: "MomoTranScript"
            is BatchInputSummary.PdfDocument ->
                input.displayName.substringBeforeLast('.').ifBlank { "PDF转录合集" }
        }

        fun packCurrentMoeTalkJson(title: String): String {
            val tempCharsObj = buildJsonObject { for ((k, v) in tempChars) put(k, v) }
            val tempImagesObj = buildJsonObject { for ((k, v) in tempImages) put(k, v) }
            val archiveObj = MoeTalkBranchPacker.packArchive(
                items = branchInputs,
                title = title,
                mode = packMode,
                tempChars = tempCharsObj,
                tempImages = tempImagesObj,
            )
            return MoeTalkBranchPacker.encodeToJsonString(archiveObj)
        }

        fun emitPageSnapshot(completed: Int, total: Int, perf: PagePerfRecord) {
            val cb = onPageCompleted ?: return
            pagesSinceMoeJsonFlush++
            // Flush MoeTalk JSON every page when fast (<250ms), or every 3 pages if large base64 payload
            val shouldSerializeJson = completed <= 5 || completed == total || lastMoeJsonFlushCostMs < 250L || pagesSinceMoeJsonFlush >= 3
            val jsonStr = if (shouldSerializeJson) {
                val tj0 = System.nanoTime()
                val s = packCurrentMoeTalkJson(tentativeTitle)
                lastMoeJsonFlushCostMs = (System.nanoTime() - tj0) / 1_000_000L
                pagesSinceMoeJsonFlush = 0
                s
            } else {
                null
            }
            val usedBranches = packMode == MoeTalkBranchPacker.MultiItemPackMode.BRANCH_PER_ITEM && branchInputs.size > 1
            cb(
                PartialBatchSnapshot(
                    title = tentativeTitle,
                    completedItems = completed,
                    totalItems = total,
                    totalWindows = totalWindows,
                    totalLines = totalLines,
                    matchedOfficialAvatars = matchedOfficialAvatars,
                    matchedVisualAssets = matchedVisualAssets,
                    elapsedMs = (System.nanoTime() - t0) / 1_000_000L,
                    combinedMomoScript = scriptBuilder.toString(),
                    moetalkArchiveJson = jsonStr,
                    usedBranchPacking = usedBranches,
                    previewPages = previewPages.toList(),
                    lastPagePerf = perf,
                ),
            )
        }

        fun ensureCustomChar(
            clusterKey: String,
            displayName: String,
            rgb: ByteArray?,
            imgW: Int,
            imgH: Int,
            cropBox: WindowBox?,
        ): Pair<String, String> {
            val existing = customKeyToId[clusterKey]
            if (existing != null) return existing to existing
            customCounter++
            val cid = "custom-%04d".format(customCounter)
            customKeyToId[clusterKey] = cid
            tempChars[cid] = buildJsonObject {
                put("name", displayName.ifBlank { cid })
                put("school", "自定义")
                put("club", "自定义")
                put("head", buildJsonArray { add(JsonPrimitive(cid)) })
                put("names", buildJsonObject {})
            }
            if (rgb != null && cropBox != null) {
                encodeCropWebpDataUrl(rgb, imgW, imgH, cropBox, square = true, maxSide = 300)?.let { url ->
                    tempImages[cid] = url
                }
            }
            return cid to cid
        }

        fun embedIllustration(rgb: ByteArray, imgW: Int, imgH: Int, box: WindowBox): String {
            val url = encodeCropWebpDataUrl(rgb, imgW, imgH, box, square = false, maxSide = 600) ?: return ""
            imageCounter++
            val id = "Image-%04d".format(imageCounter)
            tempImages[id] = url
            return id
        }

        fun resolveCharFromVisual(
            vm: AvatarVisualMatch,
            fallbackName: String,
            rgb: ByteArray,
            imgW: Int,
            imgH: Int,
        ): Pair<String, String> {
            val gallery = avatarMatcher?.gallery
            if (vm.status == "matched" && vm.character.isNotBlank() && gallery != null) {
                val head = if (vm.source == "moetalk:Char" && vm.assetId.startsWith("moetalk:Char/")) {
                    vm.assetId.removePrefix("moetalk:Char/")
                } else {
                    gallery.canonToDefaultHead[vm.character] ?: vm.skin.uppercase()
                }
                if (head.isNotBlank()) {
                    matchedOfficialAvatars++
                    return vm.character to head
                }
            }
            val key = if (vm.customClusterId.isNotBlank()) "cluster:${vm.customClusterId}" else "sp:$fallbackName"
            return ensureCustomChar(key, fallbackName, rgb, imgW, imgH, vm.subBox)
        }

        data class MoePageConversion(
            val messages: List<JsonObject>,
            val ccipAvTotal: Int,
            val ccipAvUnique: Int,
            val ccipInsTotal: Int,
            val ccipInsUnique: Int,
            val ccipInferMs: Long,
            val ccipStage2Ms: Long,
            val ccipTotalMs: Long,
        )

        fun convertPageToMoeTalk(
            rgb: ByteArray,
            imgW: Int,
            imgH: Int,
            pageRes: PageTranscriptionResult,
            onStage: (String) -> Unit,
        ): MoePageConversion {
            val tConv0 = System.nanoTime()
            val elements = pageRes.elements
            val messages = pageRes.messages

            // Optional CCIP + Stage-2 matching for avatars & insert_images on this page
            val avatarElems = elements.filter { it.classId == 0 }
            val insertElems = elements.filter { it.classId == 5 }

            val avatarMatchesByElemId = HashMap<String, List<AvatarVisualMatch>>()
            val insertMatchesByBoxKey = HashMap<String, InsertVisualMatch>()
            var avTotal = 0
            var avUniq = 0
            var insTotal = 0
            var insUniq = 0
            var ccipInferMs = 0L
            var ccipStage2Ms = 0L

            if (avatarMatcher != null) {
                if (avatarElems.isNotEmpty()) {
                    onStage("CCIP 匹配 ${avatarElems.size} 个头像…")
                    val avBoxes = avatarElems.map { WindowBox(it.box[0], it.box[1], it.box[2], it.box[3]) }
                    val avSpeakers = avBoxes.map { ab -> findNearestSpeakerForAvatar(ab, messages, imgW) }
                    val matchedGroups = avatarMatcher.matchAvatarBoxes(rgb, imgW, imgH, avBoxes, avSpeakers)
                    for (i in avatarElems.indices) {
                        avatarMatchesByElemId[avatarElems[i].id] = matchedGroups[i]
                    }
                    val st = avatarMatcher.lastAvatarStats
                    avTotal = st.inputBoxes
                    avUniq = st.uniqueCrops
                    ccipInferMs += st.inferMs
                    ccipStage2Ms += st.stage2Ms
                    val perCrop = if (st.uniqueCrops > 0) st.inferMs / st.uniqueCrops else 0L
                    onLog?.invoke(
                        "CCIP 头像: ${st.inputBoxes} 框 -> 实算 ${st.uniqueCrops} (缓存命中 ${st.cacheHits}) | " +
                            "inf=${st.inferMs}ms (${perCrop}ms/图) st2=${st.stage2Ms}ms total=${st.totalMs}ms",
                    )
                }
                if (insertElems.isNotEmpty()) {
                    onStage("CCIP 匹配 ${insertElems.size} 张插图/差分…")
                    val insBoxes = insertElems.map { WindowBox(it.box[0], it.box[1], it.box[2], it.box[3]) }
                    val insSpeakers = insBoxes.map { ib -> findContextSpeakerForInsert(ib, messages) }
                    val insMatched = avatarMatcher.matchInsertBoxes(rgb, imgW, imgH, insBoxes, insSpeakers)
                    for (i in insertElems.indices) {
                        insertMatchesByBoxKey[boxKey(insertElems[i].box)] = insMatched[i]
                    }
                    val st = avatarMatcher.lastInsertStats
                    insTotal = st.inputBoxes
                    insUniq = st.uniqueCrops
                    ccipInferMs += st.inferMs
                    ccipStage2Ms += st.stage2Ms
                    val perCrop = if (st.uniqueCrops > 0) st.inferMs / st.uniqueCrops else 0L
                    val emCnt = insMatched.count { it.status == "matched" && it.kind == "emoji" }
                    val cfCnt = insMatched.count { it.status == "matched" && it.kind == "charface" }
                    onLog?.invoke(
                        "CCIP 插图: ${st.inputBoxes} 框 -> 实算 ${st.uniqueCrops} (缓存命中 ${st.cacheHits}) | " +
                            "inf=${st.inferMs}ms (${perCrop}ms/图) st2=${st.stage2Ms}ms | Emoji=$emCnt 差分=$cfCnt 拒识=${st.inputBoxes - emCnt - cfCnt}",
                    )
                }
            }

            // Track active resolved speaker character per side across consecutive bubbles
            var lastLeftChar: Pair<String, String>? = null
            var lastRightChar: Pair<String, String>? = null
            val out = ArrayList<JsonObject>(messages.size)

            for (m in messages) {
                when (m.kind) {
                    "bubble_left", "bubble_right" -> {
                        val isProtagonist = m.isRight && m.speaker.isBlank()
                        var headsObj: JsonObject? = null
                        val scharPair: Pair<String, String> = if (isProtagonist) {
                            "0" to "1"
                        } else {
                            // Try matching nearest same-side avatar box near this bubble's top (allowing intervening insert_image)
                            val nearestAv = findNearestAvatarElement(m, avatarElems, insertElems, imgW)
                            val avMatches = nearestAv?.let { avatarMatchesByElemId[it.id] }.orEmpty()
                            if (avMatches.isNotEmpty()) {
                                val resolved = avMatches.map { vm ->
                                    resolveCharFromVisual(vm, m.speaker, rgb, imgW, imgH)
                                }
                                if (resolved.size > 1 && m.isFirstTurn && nearestAv != null) {
                                    val gw = (nearestAv.box[2] - nearestAv.box[0]).coerceAtLeast(1)
                                    val gh = (nearestAv.box[3] - nearestAv.box[1]).coerceAtLeast(1)
                                    headsObj = buildJsonObject {
                                        put("direction", if (gh > gw * 1.35) "column" else "row")
                                        put(
                                            "list",
                                            buildJsonArray {
                                                for (k in 1 until resolved.size) add(JsonPrimitive(resolved[k].second))
                                            },
                                        )
                                    }
                                }
                                resolved[0]
                            } else if (!m.isFirstTurn && (if (m.isRight) lastRightChar else lastLeftChar) != null) {
                                (if (m.isRight) lastRightChar else lastLeftChar)!!
                            } else {
                                // Resolve directly from official speaker alias if unambiguous, else custom char
                                val officialFromName = resolveOfficialFromSpeakerName(avatarMatcher, m.speaker)
                                officialFromName ?: ensureCustomChar(
                                    clusterKey = "name:${m.speaker.ifBlank { "unknown" }}",
                                    displayName = m.speaker.ifBlank { "未知角色" },
                                    rgb = rgb,
                                    imgW = imgW,
                                    imgH = imgH,
                                    cropBox = nearestAv?.let { WindowBox(it.box[0], it.box[1], it.box[2], it.box[3]) },
                                )
                            }
                        }

                        if (!isProtagonist) {
                            if (m.isRight) lastRightChar = scharPair else lastLeftChar = scharPair
                        }

                        out.add(
                            buildJsonObject {
                                put("type", "chat")
                                put("replyDepth", 0)
                                put(
                                    "sCharacter",
                                    buildJsonObject {
                                        put("no", scharPair.first)
                                        put("index", scharPair.second)
                                    },
                                )
                                put("content", m.content)
                                put("isFirst", m.isFirstTurn && !isProtagonist)
                                put("isRight", m.isRight && !isProtagonist)
                                if (m.speaker.isNotBlank() && !isProtagonist) {
                                    put("name", m.speaker)
                                }
                                if (headsObj != null) {
                                    put("heads", headsObj)
                                }
                                if (m.note.isNotBlank()) {
                                    put("time", m.note)
                                }
                            },
                        )
                    }

                    "insert_image" -> {
                        val b = if (m.box.size == 4) WindowBox(m.box[0], m.box[1], m.box[2], m.box[3]) else null
                        val insVm = if (m.box.size == 4) insertMatchesByBoxKey[boxKey(m.box)] else null
                        val gallery = avatarMatcher?.gallery

                        var content = ""
                        var fileRef = ""
                        if (insVm != null && insVm.status == "matched") {
                            when {
                                insVm.kind == "emoji" && insVm.emojiId.isNotBlank() -> {
                                    content = "Emoji"
                                    fileRef = "GameData/BLDA/Emoji/${insVm.emojiId}.webp"
                                    matchedVisualAssets++
                                }
                                insVm.kind == "charface" && insVm.source == "moetalk:CharFace" && insVm.expressionId.isNotBlank() -> {
                                    content = "CharFace"
                                    fileRef = "GameData/BLDA/CharFace/${insVm.expressionId}.webp"
                                    matchedVisualAssets++
                                }
                                insVm.kind == "charface" && insVm.packV2Preset != null && insVm.packV2Ordinal > 0 && gallery != null -> {
                                    val mtExpr = gallery.packV2ToMoetalkExpr[insVm.packV2Preset to insVm.packV2Ordinal]
                                    if (mtExpr != null) {
                                        content = "CharFace"
                                        fileRef = "GameData/BLDA/CharFace/$mtExpr.webp"
                                        matchedVisualAssets++
                                    } else if (b != null) {
                                        fileRef = embedIllustration(rgb, imgW, imgH, b)
                                    }
                                }
                                b != null -> fileRef = embedIllustration(rgb, imgW, imgH, b)
                            }
                        } else if (b != null) {
                            fileRef = embedIllustration(rgb, imgW, imgH, b)
                        }

                        val offset = if (b != null) (b.x0 + b.x1) - imgW else 0
                        val isCenter = b == null || abs(offset) < imgW * 0.08f
                        val isRight = !isCenter && offset > 0
                        val scharPair = when {
                            isCenter -> "0" to "1"
                            isRight -> lastRightChar ?: ("0" to "1")
                            else -> lastLeftChar ?: ("0" to "1")
                        }

                        out.add(
                            buildJsonObject {
                                put("type", "image")
                                put("replyDepth", 0)
                                put(
                                    "sCharacter",
                                    buildJsonObject {
                                        put("no", scharPair.first)
                                        put("index", scharPair.second)
                                    },
                                )
                                put("content", content)
                                put("file", fileRef)
                                put("isFirst", false)
                                put("isRight", isRight && scharPair.first != "0")
                                put("isCenter", isCenter)
                                if (m.note.isNotBlank()) put("time", m.note)
                            },
                        )
                    }

                    "narration" -> {
                        out.add(
                            buildJsonObject {
                                put("type", "info")
                                put("replyDepth", 0)
                                put("sCharacter", buildJsonObject { put("no", "0"); put("index", "1") })
                                put("content", m.content)
                            },
                        )
                    }

                    "reply" -> {
                        out.add(
                            buildJsonObject {
                                put("type", "reply")
                                put("replyDepth", 0)
                                put("sCharacter", buildJsonObject { put("no", "0"); put("index", "1") })
                                put("content", m.content)
                            },
                        )
                    }

                    "bond" -> {
                        val scharPair = lastLeftChar ?: ("0" to "1")
                        out.add(
                            buildJsonObject {
                                put("type", "heart")
                                put("replyDepth", 0)
                                put(
                                    "sCharacter",
                                    buildJsonObject {
                                        put("no", scharPair.first)
                                        put("index", scharPair.second)
                                    },
                                )
                                put("content", m.content)
                            },
                        )
                    }
                }
            }
            return MoePageConversion(
                messages = out,
                ccipAvTotal = avTotal,
                ccipAvUnique = avUniq,
                ccipInsTotal = insTotal,
                ccipInsUnique = insUniq,
                ccipInferMs = ccipInferMs,
                ccipStage2Ms = ccipStage2Ms,
                ccipTotalMs = (System.nanoTime() - tConv0) / 1_000_000L,
            )
        }

        val archiveTitle: String = when (input) {
            is BatchInputSummary.Images -> {
                val total = input.items.size
                for ((idx, item) in input.items.withIndex()) {
                    currentCoroutineContext().ensureActive()
                    val tPage0 = System.nanoTime()
                    val label = item.displayName.substringBeforeLast('.').ifBlank { (idx + 1).toString() }
                    onProgress(BatchProgress(idx, total, label, "解码原图…"))

                    val tExt0 = System.nanoTime()
                    val bitmap = context.contentResolver.openInputStream(item.uri)?.use { stream ->
                        BitmapFactory.decodeStream(stream)
                    } ?: continue
                    val (rgb, w, h) = bitmapToRgbAndRecycle(bitmap)
                    val extractMs = (System.nanoTime() - tExt0) / 1_000_000L
                    onLog?.invoke("[$label] 解码原图: ${w}x${h} (ext=${extractMs}ms)")

                    val pageRes = try {
                        engine.transcribeRgb(
                            rgbBytes = rgb,
                            width = w,
                            height = h,
                            onLog = { msg -> onLog?.invoke("[$label] $msg") },
                        ) { stage ->
                            onProgress(BatchProgress(idx, total, label, stage))
                        }
                    } catch (ce: CancellationException) {
                        throw ce
                    }

                    val conv = convertPageToMoeTalk(rgb, w, h, pageRes) { stage ->
                        onProgress(BatchProgress(idx, total, label, stage))
                    }

                    totalWindows += pageRes.windowCount
                    totalLines += pageRes.lineCount
                    if (total > 1) {
                        if (scriptBuilder.isNotEmpty()) scriptBuilder.append('\n')
                        scriptBuilder.append("=== ").append(label).append(" ===\n")
                    }
                    scriptBuilder.append(pageRes.momoScriptText)
                    branchInputs.add(MoeTalkBranchPacker.BranchInput(label, conv.messages))
                    previewPages.add(
                        PagePreviewRecord(
                            label = label,
                            imageUri = item.uri.toString(),
                            width = w,
                            height = h,
                            elements = pageRes.elements,
                        ),
                    )
                    val pageTotalMs = (System.nanoTime() - tPage0) / 1_000_000L
                    val pf = pageRes.profile
                    val perf = PagePerfRecord(
                        pageNo = idx + 1,
                        totalPages = total,
                        label = label,
                        width = w,
                        height = h,
                        windowCount = pageRes.windowCount,
                        lineCount = pageRes.lineCount,
                        escalatedLines = pageRes.escalatedLineCount,
                        adoptedMedium = pageRes.adoptedMediumCount,
                        extractMs = extractMs,
                        windowPlanMs = pf.windowPlanMs,
                        detPrepMs = pf.detPrepMs,
                        detInferMs = pf.detInferMs,
                        detDecodeMs = pf.detDecodeMs,
                        cvLinesMs = pf.cvLinesMs,
                        recSmallMs = pf.recSmallMs,
                        recMediumMs = pf.recMediumMs,
                        assembleMs = pf.assembleMs,
                        ccipAvTotal = conv.ccipAvTotal,
                        ccipAvUnique = conv.ccipAvUnique,
                        ccipInsTotal = conv.ccipInsTotal,
                        ccipInsUnique = conv.ccipInsUnique,
                        ccipInferMs = conv.ccipInferMs,
                        ccipStage2Ms = conv.ccipStage2Ms,
                        ccipTotalMs = conv.ccipTotalMs,
                        pageTotalMs = pageTotalMs,
                    )
                    emitPageSnapshot(idx + 1, total, perf)
                    onProgress(BatchProgress(idx + 1, total, label, "完成 ($pageTotalMs ms)"))
                }
                tentativeTitle
            }

            is BatchInputSummary.PdfDocument -> {
                PdfDocumentExtractor.openDocument(context, input.uri)?.use { doc ->
                    val total = doc.numberOfPages
                    var pdfRenderer: PdfRenderer? = null
                    try {
                        for (pageIndex in 0 until total) {
                            currentCoroutineContext().ensureActive()
                            val tPage0 = System.nanoTime()
                            val pageNo = pageIndex + 1
                            val label = "P%03d".format(pageNo)
                            onProgress(BatchProgress(pageIndex, total, label, "提取第 $pageNo/$total 页原图…"))

                            val tExt0 = System.nanoTime()
                            val pageBitmaps = ArrayList(PdfDocumentExtractor.extractPageImages(doc.getPage(pageIndex)))
                            if (pageBitmaps.isEmpty()) {
                                if (pdfRenderer == null) {
                                    context.contentResolver.openFileDescriptor(input.uri, "r")?.let { rPfd ->
                                        pdfRenderer = PdfRenderer(rPfd)
                                    }
                                }
                                pdfRenderer?.openPage(pageIndex)?.use { page ->
                                    val scale = (720f / page.width.coerceAtLeast(1)).coerceIn(1f, 2f)
                                    val rw = (page.width * scale).roundToInt().coerceAtLeast(1)
                                    val rh = (page.height * scale).roundToInt().coerceAtLeast(1)
                                    val bmp = Bitmap.createBitmap(rw, rh, Bitmap.Config.ARGB_8888)
                                    Canvas(bmp).drawColor(Color.WHITE)
                                    page.render(bmp, null, null, PdfRenderer.Page.RENDER_MODE_FOR_DISPLAY)
                                    pageBitmaps.add(bmp)
                                }
                            }
                            var extractMs = (System.nanoTime() - tExt0) / 1_000_000L

                            val pageMtMsgs = ArrayList<JsonObject>()
                            val pageText = StringBuilder()
                            var pageW = 0
                            var pageH = 0
                            var pWins = 0
                            var pLines = 0
                            var pEsc = 0
                            var pAdopt = 0
                            var pPlanMs = 0L
                            var pDetPrepMs = 0L
                            var pDetInfMs = 0L
                            var pDetDecMs = 0L
                            var pCvMs = 0L
                            var pRecSMs = 0L
                            var pRecMMs = 0L
                            var pAsmMs = 0L
                            var pAvTot = 0
                            var pAvUniq = 0
                            var pInsTot = 0
                            var pInsUniq = 0
                            var pCcipInfMs = 0L
                            var pCcipSt2Ms = 0L
                            var pCcipTotMs = 0L

                            for ((bi, bmp) in pageBitmaps.withIndex()) {
                                val tBmp = System.nanoTime()
                                val cachedPreviewPath = if (previewCacheDir != null && pageIndex < 120 && bi == 0) {
                                    previewCacheDir.mkdirs()
                                    val pf = File(previewCacheDir, "page_%03d.jpg".format(pageNo))
                                    runCatching {
                                        pf.outputStream().use { out -> bmp.compress(Bitmap.CompressFormat.JPEG, 78, out) }
                                        pf.absolutePath
                                    }.getOrNull()
                                } else {
                                    null
                                }
                                val (rgb, w, h) = bitmapToRgbAndRecycle(bmp)
                                extractMs += (System.nanoTime() - tBmp) / 1_000_000L
                                if (bi == 0) {
                                    pageW = w
                                    pageH = h
                                }
                                onLog?.invoke("[$label] 提取原图 #${bi + 1}: ${w}x${h} (ext=${extractMs}ms)")
                                val res = engine.transcribeRgb(
                                    rgbBytes = rgb,
                                    width = w,
                                    height = h,
                                    onLog = { msg -> onLog?.invoke("[$label] $msg") },
                                ) { stage ->
                                    onProgress(BatchProgress(pageIndex, total, label, stage))
                                }
                                val conv = convertPageToMoeTalk(rgb, w, h, res) { stage ->
                                    onProgress(BatchProgress(pageIndex, total, label, stage))
                                }
                                if (cachedPreviewPath != null) {
                                    previewPages.add(
                                        PagePreviewRecord(
                                            label = label,
                                            previewImagePath = cachedPreviewPath,
                                            width = w,
                                            height = h,
                                            elements = res.elements,
                                        ),
                                    )
                                }
                                totalWindows += res.windowCount
                                totalLines += res.lineCount
                                pWins += res.windowCount
                                pLines += res.lineCount
                                pEsc += res.escalatedLineCount
                                pAdopt += res.adoptedMediumCount
                                pPlanMs += res.profile.windowPlanMs
                                pDetPrepMs += res.profile.detPrepMs
                                pDetInfMs += res.profile.detInferMs
                                pDetDecMs += res.profile.detDecodeMs
                                pCvMs += res.profile.cvLinesMs
                                pRecSMs += res.profile.recSmallMs
                                pRecMMs += res.profile.recMediumMs
                                pAsmMs += res.profile.assembleMs
                                pAvTot += conv.ccipAvTotal
                                pAvUniq += conv.ccipAvUnique
                                pInsTot += conv.ccipInsTotal
                                pInsUniq += conv.ccipInsUnique
                                pCcipInfMs += conv.ccipInferMs
                                pCcipSt2Ms += conv.ccipStage2Ms
                                pCcipTotMs += conv.ccipTotalMs

                                pageText.append(res.momoScriptText)
                                pageMtMsgs.addAll(conv.messages)
                            }

                            if (total > 1) {
                                if (scriptBuilder.isNotEmpty()) scriptBuilder.append('\n')
                                scriptBuilder.append("=== 第 ").append(pageNo).append(" 页 ===\n")
                            }
                            scriptBuilder.append(pageText)
                            branchInputs.add(MoeTalkBranchPacker.BranchInput(label, pageMtMsgs))
                            val pageTotalMs = (System.nanoTime() - tPage0) / 1_000_000L
                            val perf = PagePerfRecord(
                                pageNo = pageNo,
                                totalPages = total,
                                label = label,
                                width = pageW,
                                height = pageH,
                                windowCount = pWins,
                                lineCount = pLines,
                                escalatedLines = pEsc,
                                adoptedMedium = pAdopt,
                                extractMs = extractMs,
                                windowPlanMs = pPlanMs,
                                detPrepMs = pDetPrepMs,
                                detInferMs = pDetInfMs,
                                detDecodeMs = pDetDecMs,
                                cvLinesMs = pCvMs,
                                recSmallMs = pRecSMs,
                                recMediumMs = pRecMMs,
                                assembleMs = pAsmMs,
                                ccipAvTotal = pAvTot,
                                ccipAvUnique = pAvUniq,
                                ccipInsTotal = pInsTot,
                                ccipInsUnique = pInsUniq,
                                ccipInferMs = pCcipInfMs,
                                ccipStage2Ms = pCcipSt2Ms,
                                ccipTotalMs = pCcipTotMs,
                                pageTotalMs = pageTotalMs,
                            )
                            emitPageSnapshot(pageNo, total, perf)
                            onProgress(BatchProgress(pageNo, total, label, "完成 ($pageTotalMs ms)"))
                        }
                    } finally {
                        pdfRenderer?.close()
                    }
                }
                tentativeTitle
            }
        }

        val usedBranches = packMode == MoeTalkBranchPacker.MultiItemPackMode.BRANCH_PER_ITEM && branchInputs.size > 1
        val elapsedMs = (System.nanoTime() - t0) / 1_000_000L

        return BatchTranscriptionOutput(
            title = archiveTitle,
            itemCount = branchInputs.size,
            totalWindows = totalWindows,
            totalLines = totalLines,
            matchedOfficialAvatars = matchedOfficialAvatars,
            matchedVisualAssets = matchedVisualAssets,
            totalElapsedMs = elapsedMs,
            combinedMomoScript = scriptBuilder.toString(),
            moetalkArchiveJson = packCurrentMoeTalkJson(archiveTitle),
            usedBranchPacking = usedBranches,
            previewPages = previewPages,
        )
    }

    private fun boxKey(box: List<Int>): String = "${box[0]},${box[1]},${box[2]},${box[3]}"

    private fun findNearestSpeakerForAvatar(
        avBox: WindowBox,
        messages: List<NativeAssembledMessage>,
        imgW: Int,
    ): String? {
        val avLeft = (avBox.x0 + avBox.x1) < imgW
        var best: String? = null
        var bestDist = Float.MAX_VALUE
        for (m in messages) {
            if (m.speaker.isBlank() || m.box.size != 4) continue
            if (m.isRight == avLeft) continue
            val dy = abs(m.box[1] - avBox.y0).toFloat()
            if (dy <= imgW * 0.14f && dy < bestDist) {
                bestDist = dy
                best = m.speaker
            }
        }
        return best
    }

    private fun findContextSpeakerForInsert(
        insBox: WindowBox,
        messages: List<NativeAssembledMessage>,
    ): String? {
        var ctx: String? = null
        for (m in messages) {
            if (m.box.size == 4 && m.box[1] > insBox.y1) break
            if (m.speaker.isNotBlank()) ctx = m.speaker
        }
        return ctx
    }

    private fun findNearestAvatarElement(
        msg: NativeAssembledMessage,
        avatarElems: List<NativeDetectedElement>,
        insertElems: List<NativeDetectedElement>,
        imgW: Int,
    ): NativeDetectedElement? {
        if (msg.box.size != 4 || avatarElems.isEmpty()) return null
        val by0 = msg.box[1]
        var best: NativeDetectedElement? = null
        var bestDist = Float.MAX_VALUE
        for (av in avatarElems) {
            val avLeft = (av.box[0] + av.box[2]) < imgW
            if (avLeft == msg.isRight) continue
            // Extend anchor bottom across aligned insert_image(s) directly under this avatar
            var anchorY1 = av.box[3]
            for (ins in insertElems) {
                if (ins.box[1] >= av.box[1] - imgW * 0.035f && ins.box[1] <= anchorY1 + imgW * 0.12f) {
                    val aligned = if (avLeft) {
                        ins.box[0] >= av.box[0] - imgW * 0.025f && ins.box[0] <= av.box[2] + imgW * 0.18f
                    } else {
                        ins.box[2] <= av.box[2] + imgW * 0.025f && ins.box[2] >= av.box[0] - imgW * 0.18f
                    }
                    if (aligned && ins.box[3] > anchorY1) {
                        anchorY1 = ins.box[3]
                    }
                }
            }
            if (by0 >= av.box[1] - imgW * 0.035f && by0 <= anchorY1 + imgW * 0.10f) {
                val dist = abs(by0 - av.box[1]).toFloat()
                if (dist < bestDist) {
                    bestDist = dist
                    best = av
                }
            }
        }
        return best
    }

    private fun resolveOfficialFromSpeakerName(
        matcher: OnnxAvatarMatcher?,
        speaker: String,
    ): Pair<String, String>? {
        val gallery = matcher?.gallery ?: return null
        val canons = gallery.resolveSpeakerCanons(speaker)
        if (canons.size != 1) return null
        val canon = canons.first()
        val head = gallery.canonToDefaultHead[canon] ?: return null
        return canon to head
    }

    private fun encodeCropWebpDataUrl(
        rgb: ByteArray,
        imgW: Int,
        imgH: Int,
        box: WindowBox,
        square: Boolean,
        maxSide: Int,
        quality: Int = 85,
    ): String? {
        var x0 = box.x0.coerceIn(0, imgW)
        var y0 = box.y0.coerceIn(0, imgH)
        var x1 = box.x1.coerceIn(0, imgW)
        var y1 = box.y1.coerceIn(0, imgH)
        var w = x1 - x0
        var h = y1 - y0
        if (w <= 0 || h <= 0) return null
        if (square && w != h) {
            val side = minOf(w, h)
            x0 += (w - side) / 2
            y0 += (h - side) / 2
            w = side
            h = side
        }
        val pixels = IntArray(w * h)
        for (y in 0 until h) {
            val srcRow = ((y0 + y) * imgW + x0) * 3
            val dstRow = y * w
            for (x in 0 until w) {
                val r = rgb[srcRow + x * 3 + 0].toInt() and 0xFF
                val g = rgb[srcRow + x * 3 + 1].toInt() and 0xFF
                val b = rgb[srcRow + x * 3 + 2].toInt() and 0xFF
                pixels[dstRow + x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        var bmp = Bitmap.createBitmap(pixels, w, h, Bitmap.Config.ARGB_8888)
        if (maxOf(w, h) > maxSide) {
            val scale = maxSide.toFloat() / maxOf(w, h).toFloat()
            val nw = (w * scale).roundToInt().coerceAtLeast(1)
            val nh = (h * scale).roundToInt().coerceAtLeast(1)
            val scaled = Bitmap.createScaledBitmap(bmp, nw, nh, true)
            bmp.recycle()
            bmp = scaled
        }
        val baos = ByteArrayOutputStream()
        val fmt = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            Bitmap.CompressFormat.WEBP_LOSSY
        } else {
            @Suppress("DEPRECATION")
            Bitmap.CompressFormat.WEBP
        }
        bmp.compress(fmt, quality, baos)
        bmp.recycle()
        val b64 = Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP)
        return "data:image/webp;base64,$b64"
    }

    private fun bitmapToRgbAndRecycle(src: Bitmap): Triple<ByteArray, Int, Int> {
        val bmp = if (src.config == Bitmap.Config.ARGB_8888) {
            src
        } else {
            val copy = src.copy(Bitmap.Config.ARGB_8888, false)
            src.recycle()
            copy
        }
        val w = bmp.width
        val h = bmp.height
        val rgb = ByteArray(w * h * 3)
        val rowPixels = IntArray(w)
        var dst = 0
        for (y in 0 until h) {
            bmp.getPixels(rowPixels, 0, w, 0, y, w, 1)
            for (x in 0 until w) {
                val px = rowPixels[x]
                rgb[dst++] = ((px ushr 16) and 0xFF).toByte()
                rgb[dst++] = ((px ushr 8) and 0xFF).toByte()
                rgb[dst++] = (px and 0xFF).toByte()
            }
        }
        bmp.recycle()
        return Triple(rgb, w, h)
    }
}
