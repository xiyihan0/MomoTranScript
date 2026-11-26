package dev.momotranscript.app.engine

import dev.momotranscript.app.nativecore.NativeAssembledMessage
import dev.momotranscript.app.nativecore.NativeCore
import dev.momotranscript.app.nativecore.NativeDetectedElement
import dev.momotranscript.app.nativecore.TextLineBox
import java.io.Closeable
import java.io.File

data class EngineInitProfile(
    val vocabLoadMs: Long,
    val detLoadMs: Long,
    val recSmallLoadMs: Long,
    val recMediumLoadMs: Long,
    val intraOpThreads: Int,
    val hasCascadeMedium: Boolean,
    val detEp: String = "CPU(FP16)",
    val recSmallEp: String = "CPU(FP16)",
    val recMediumEp: String = "off",
)

data class PageProfileTiming(
    val windowPlanMs: Long = 0L,
    val detPrepMs: Long = 0L,
    val detInferMs: Long = 0L,
    val detDecodeMs: Long = 0L,
    val cvLinesMs: Long = 0L,
    val recSmallMs: Long = 0L,
    val recSmallBatches: Int = 0,
    val recMediumMs: Long = 0L,
    val assembleMs: Long = 0L,
    val totalMs: Long = 0L,
)

data class PageTranscriptionResult(
    val momoScriptText: String,
    val messages: List<NativeAssembledMessage>,
    val elements: List<NativeDetectedElement>,
    val windowCount: Int,
    val lineCount: Int,
    val escalatedLineCount: Int,
    val adoptedMediumCount: Int,
    val elapsedMs: Long,
    val profile: PageProfileTiming = PageProfileTiming(totalMs = elapsedMs),
)

/**
 * On-device MNN 3.6.2 transcription engine (OpenCL GPU / Vulkan GPU / ARMv9 FP16 CPU):
 * 1. Adaptive blank-band windowing (C++20)
 * 2. 11-class PP-DocLayoutV3 detector (`mnn/det_boxes_order_w8.mnn`)
 * 3. V2 query-winner + same-window TR suppression + IoU 0.55 fusion (C++20)
 * 4. Dual-polarity 8-CC physical CV line localization (C++20, zero OpenCV)
 * 5. Aspect-bucketed PP-OCRv6 Small recognition (`mnn/rec_small_fp16.mnn`)
 *    + optional Medium cascade (`mnn/rec_medium_fp16.mnn`) for weak lines
 * 6. MomoScript plaintext + structured message assembly (C++20)
 */
class OnnxTranscriptionEngine(
    private val bundleRoot: File,
    private val intraOpThreads: Int = Runtime.getRuntime().availableProcessors().coerceIn(2, 4),
    private val backendMode: Int = BACKEND_AUTO_GPU_CPU,
    private val cacheDir: File? = null,
) : Closeable {

    private val detHandle: NativeCore.MnnDetHandle
    private val recSmallHandle: NativeCore.MnnRecHandle
    private val recMediumHandle: NativeCore.MnnRecHandle?
    private val vocab: OcrVocab
    val initProfile: EngineInitProfile

    init {
        val detFile = File(bundleRoot, "mnn/det_boxes_order_ln_w8.mnn")
        val recSmallFile = File(bundleRoot, "mnn/rec_small_fp16.mnn")
        val recMedFile = File(bundleRoot, "mnn/rec_medium_fp16.mnn")
        val vocabFile = File(bundleRoot, "models/ocr_vocab.yml")

        require(detFile.isFile) { "缺少 MNN Fused-LN 检测模型：${detFile.absolutePath}（请前往「模型」页更新基础转录包）" }
        require(recSmallFile.isFile) { "缺少 MNN 基础识别模型：${recSmallFile.absolutePath}（请前往「模型」页更新基础转录包）" }
        require(vocabFile.isFile) { "缺少 OCR 词表：${vocabFile.absolutePath}" }

        val detReq = when (backendMode) {
            BACKEND_AUTO_GPU_CPU, BACKEND_OPENCL_ALL -> NativeCore.MNN_BACKEND_OPENCL
            BACKEND_VULKAN_ALL -> NativeCore.MNN_BACKEND_VULKAN
            else -> NativeCore.MNN_BACKEND_CPU_FP16
        }
        val recReq = when (backendMode) {
            BACKEND_OPENCL_ALL -> NativeCore.MNN_BACKEND_OPENCL
            BACKEND_VULKAN_ALL -> NativeCore.MNN_BACKEND_VULKAN
            else -> NativeCore.MNN_BACKEND_CPU_FP16 // AUTO uses ARMv9 FP16 CPU for dynamic-width OCR
        }

        val mnnCache = cacheDir?.let { File(it, "mnn_cache").apply { mkdirs() } }
        fun cachePathFor(tag: String, req: Int): String =
            if (mnnCache != null && req != NativeCore.MNN_BACKEND_CPU_FP16) {
                File(mnnCache, "${tag}_b${req}.bin").absolutePath
            } else {
                ""
            }

        val tV = System.nanoTime()
        vocab = OcrVocab.fromYamlFile(vocabFile)
        val vocabMs = (System.nanoTime() - tV) / 1_000_000L

        val tD = System.nanoTime()
        detHandle = NativeCore.MnnDetHandle(
            modelPath = detFile.absolutePath,
            backendReq = detReq,
            threads = intraOpThreads,
            cachePath = cachePathFor("det_ln", detReq),
        )
        val detMs = (System.nanoTime() - tD) / 1_000_000L

        val tS = System.nanoTime()
        recSmallHandle = NativeCore.MnnRecHandle(
            modelPath = recSmallFile.absolutePath,
            backendReq = recReq,
            threads = intraOpThreads,
            cachePath = cachePathFor("recS", recReq),
        )
        val recSmallMs = (System.nanoTime() - tS) / 1_000_000L

        val tM = System.nanoTime()
        recMediumHandle = if (recMedFile.isFile) {
            NativeCore.MnnRecHandle(
                modelPath = recMedFile.absolutePath,
                backendReq = recReq,
                threads = intraOpThreads,
                cachePath = cachePathFor("recM", recReq),
            )
        } else {
            null
        }
        val recMedMs = if (recMediumHandle != null) (System.nanoTime() - tM) / 1_000_000L else 0L

        initProfile = EngineInitProfile(
            vocabLoadMs = vocabMs,
            detLoadMs = detMs,
            recSmallLoadMs = recSmallMs,
            recMediumLoadMs = recMedMs,
            intraOpThreads = intraOpThreads,
            hasCascadeMedium = recMediumHandle != null,
            detEp = detHandle.activeBackend,
            recSmallEp = recSmallHandle.activeBackend,
            recMediumEp = recMediumHandle?.activeBackend ?: "off",
        )
    }

    val hasCascadeMedium: Boolean
        get() = recMediumHandle != null

    fun transcribeRgb(
        rgbBytes: ByteArray,
        width: Int,
        height: Int,
        onLog: ((String) -> Unit)? = null,
        onProgress: ((String) -> Unit)? = null,
    ): PageTranscriptionResult {
        val t0 = System.nanoTime()

        // 1. Adaptive blank-band windowing
        onProgress?.invoke("切窗规划中…")
        val tPlan0 = System.nanoTime()
        val windows = NativeCore.planAdaptiveWindows(rgbBytes, width, height)
            .ifEmpty { NativeCore.planFixedWindows(width, height) }
        val windowPlanMs = (System.nanoTime() - tPlan0) / 1_000_000L
        val avgWinH = if (windows.isNotEmpty()) windows.sumOf { it.height } / windows.size else 0
        onLog?.invoke("切窗规划: ${width}x${height} -> ${windows.size} 窗 (plan=${windowPlanMs}ms, 均窗高 ${avgWinH}px)")

        // 2. Run detector window by window (zero Java boxing via flat C++ MNN buffers)
        val numQueries = 300
        val numClasses = 11
        val allBboxes = FloatArray(windows.size * numQueries * 4)
        val allLogits = FloatArray(windows.size * numQueries * numClasses)
        val winBboxes = FloatArray(numQueries * 4)
        val winLogits = FloatArray(numQueries * numClasses)

        var detPrepNs = 0L
        var detInferNs = 0L
        for ((wi, win) in windows.withIndex()) {
            onProgress?.invoke("结构检测 ${wi + 1}/${windows.size} 窗…")
            val tp0 = System.nanoTime()
            val inputData = NativeCore.prepareDetInput(rgbBytes, width, height, win, DET_INPUT_SIZE)
            detPrepNs += System.nanoTime() - tp0
            val ti0 = System.nanoTime()
            detHandle.runWindow(inputData, winBboxes, winLogits)
            System.arraycopy(winBboxes, 0, allBboxes, wi * numQueries * 4, numQueries * 4)
            System.arraycopy(winLogits, 0, allLogits, wi * numQueries * numClasses, numQueries * numClasses)
            detInferNs += System.nanoTime() - ti0
            val doneW = wi + 1
            if (doneW == 1 || doneW % 5 == 0 || doneW == windows.size) {
                val avgMs = (detInferNs / 1_000_000L) / doneW
                onLog?.invoke("  det 进度 $doneW/${windows.size} 窗 · 均 ${avgMs}ms/窗")
            }
        }

        // 3. Native V2 decode + TR suppression + IoU 0.55 fusion
        val tDec0 = System.nanoTime()
        val elements = NativeCore.decodeAndFuseDetections(
            bboxesFlat = allBboxes,
            logitsFlat = allLogits,
            numQueries = numQueries,
            windows = windows,
            imgW = width,
            imgH = height,
        )
        val detDecodeMs = (System.nanoTime() - tDec0) / 1_000_000L
        val clsCounts = IntArray(11)
        for (e in elements) if (e.classId in 0..10) clsCounts[e.classId]++
        val detInfMs = detInferNs / 1_000_000L
        val detPrpMs = detPrepNs / 1_000_000L
        val avgDetW = if (windows.isNotEmpty()) detInfMs / windows.size else 0L
        onLog?.invoke(
            "det 完成 [${detHandle.activeBackend}]: ${windows.size} 窗 ${detPrpMs + detInfMs + detDecodeMs}ms " +
                "(prep=${detPrpMs}ms inf=${detInfMs}ms[${avgDetW}ms/w] dec=${detDecodeMs}ms) | " +
                "共 ${elements.size} 框 (头像${clsCounts[0]} 姓名${clsCounts[1]} 左泡${clsCounts[2]} " +
                "右泡${clsCounts[3]} 旁白${clsCounts[4]} 插图${clsCounts[5]} 回复${clsCounts[6]} 附注${clsCounts[10]})",
        )

        // 4. Native CV physical line localization per text element
        onProgress?.invoke("物理切行中 (${elements.size} 个区域)…")
        val tCv0 = System.nanoTime()
        data class IndexedLine(
            val elemIndex: Int,
            val lineInElem: Int,
            val box: TextLineBox,
        )

        val allLines = ArrayList<IndexedLine>()
        for ((ei, elem) in elements.withIndex()) {
            if (!isTextClass(elem.classId)) continue
            val b = elem.box
            val tight = elem.classId == 1 || elem.classId == 10
            val proposals = NativeCore.segmentLines(rgbBytes, width, height, b[0], b[1], b[2], b[3], tight)
            for ((li, p) in proposals.withIndex()) {
                allLines.add(IndexedLine(ei, li, p))
            }
        }
        val cvLinesMs = (System.nanoTime() - tCv0) / 1_000_000L

        val inkHeights = allLines.map { it.box.inkHeight }.filter { it >= 6 }.sorted()
        val boxHeights = allLines.map { it.box.height }.filter { it >= 6 }.sorted()
        val medInkH = if (inkHeights.isNotEmpty()) inkHeights[inkHeights.size / 2].toFloat() else null
        val medBoxH = if (boxHeights.isNotEmpty()) boxHeights[boxHeights.size / 2] else 0

        fun shortLineTargetH(line: TextLineBox): Int {
            if (medInkH == null || medBoxH <= 0) return 0
            val iw = line.inkWidth
            val ih = line.inkHeight
            if (iw <= 0 || ih <= 0) return 0
            val ratio = iw.toFloat() / ih.toFloat()
            return if (ih <= maxOf(5.0f, medInkH * 0.42f) && ratio in 2.5f..45.0f) medBoxH else 0
        }
        val shortPadCount = allLines.count { shortLineTargetH(it.box) > 0 }
        onLog?.invoke(
            "CV 切行: ${allLines.size} 行 (${cvLinesMs}ms, 中位墨高=${medInkH?.toInt() ?: 0}px, 垫高短行=$shortPadCount)",
        )

        // 5. Aspect-ratio bucketed batch recognition (Small)
        val tRecS0 = System.nanoTime()
        var recSmallBatches = 0
        val recResults = arrayOfNulls<CtcLineResult>(allLines.size)
        val sortedIndices = allLines.indices.sortedBy { idx ->
            val ln = allLines[idx].box
            ln.width.toFloat() / ln.height.coerceAtLeast(1).toFloat()
        }

        val totalSmallBatches = (sortedIndices.size + REC_SMALL_BATCH_SIZE - 1) / REC_SMALL_BATCH_SIZE
        var doneLines = 0
        for (start in sortedIndices.indices step REC_SMALL_BATCH_SIZE) {
            val end = minOf(start + REC_SMALL_BATCH_SIZE, sortedIndices.size)
            val batchIdx = sortedIndices.subList(start, end)
            onProgress?.invoke("OCR 识别 $doneLines/${allLines.size} 行…")
            val batchOut = runRecBatch(
                handle = recSmallHandle,
                rgbBytes = rgbBytes,
                width = width,
                height = height,
                lines = batchIdx.map { allLines[it].box },
                padHeights = batchIdx.map { shortLineTargetH(allLines[it].box) },
            )
            for (k in batchIdx.indices) {
                recResults[batchIdx[k]] = batchOut[k]
            }
            doneLines += batchIdx.size
            recSmallBatches++
            if (recSmallBatches % 15 == 0 && doneLines < allLines.size) {
                val curMs = (System.nanoTime() - tRecS0) / 1_000_000L
                onLog?.invoke("  rec_small 进度 $doneLines/${allLines.size} 行 ($recSmallBatches/$totalSmallBatches 批, ${curMs}ms)")
            }
        }
        val recSmallMs = (System.nanoTime() - tRecS0) / 1_000_000L
        val avgSmallLine = if (allLines.isNotEmpty()) recSmallMs.toDouble() / allLines.size else 0.0
        onLog?.invoke(
            "rec_small 完成 [${recSmallHandle.activeBackend}]: ${allLines.size} 行 / $recSmallBatches 批 = ${recSmallMs}ms (${"%.1f".format(avgSmallLine)}ms/行)",
        )

        // 6. Optional Small -> Medium cascade for weak / blank lines
        var escalatedCount = 0
        var adoptedMedCount = 0
        val tRecM0 = System.nanoTime()
        if (recMediumHandle != null && allLines.isNotEmpty()) {
            val hardIndices = sortedIndices.filter { idx ->
                val r = recResults[idx] ?: return@filter true
                r.text.isBlank() || r.score < CASCADE_SCORE_THRESHOLD || r.minCharScore < CASCADE_MIN_CHAR_THRESHOLD
            }
            escalatedCount = hardIndices.size
            if (hardIndices.isNotEmpty()) {
                onProgress?.invoke("高精度复核 ${hardIndices.size} 行…")
                var medBatches = 0
                for (start in hardIndices.indices step REC_MEDIUM_BATCH_SIZE) {
                    val end = minOf(start + REC_MEDIUM_BATCH_SIZE, hardIndices.size)
                    val batchIdx = hardIndices.subList(start, end)
                    val medOut = runRecBatch(
                        handle = recMediumHandle,
                        rgbBytes = rgbBytes,
                        width = width,
                        height = height,
                        lines = batchIdx.map { allLines[it].box },
                        padHeights = batchIdx.map { shortLineTargetH(allLines[it].box) },
                    )
                    medBatches++
                    for (k in batchIdx.indices) {
                        val globalIdx = batchIdx[k]
                        val smallRes = recResults[globalIdx]!!
                        val medRes = medOut[k]
                        if (preferMedium(smallRes, medRes)) {
                            recResults[globalIdx] = medRes
                            adoptedMedCount++
                        }
                    }
                }
                val medMsNow = (System.nanoTime() - tRecM0) / 1_000_000L
                val avgMed = medMsNow.toDouble() / escalatedCount
                onLog?.invoke(
                    "rec_medium 完成 [${recMediumHandle.activeBackend}]: 复核 $escalatedCount 行 / $medBatches 批 = ${medMsNow}ms (${"%.1f".format(avgMed)}ms/行) | 采纳 $adoptedMedCount 行",
                )
            }
        }
        val recMediumMs = if (escalatedCount > 0) (System.nanoTime() - tRecM0) / 1_000_000L else 0L

        // 7. Re-group accepted line texts back into elements & assemble MomoScript via C++20
        val tAsm0 = System.nanoTime()
        val perElemLines = Array(elements.size) { ArrayList<String>() }
        for (i in allLines.indices) {
            val item = allLines[i]
            val res = recResults[i] ?: continue
            val stripped = res.text.trim()
            val isPunctOnly = stripped.isNotEmpty() && stripped.none { it.isLetterOrDigit() }
            val accepted = if (stripped.isNotEmpty() && (res.score >= MIN_TEXT_SCORE || isPunctOnly)) stripped else ""
            perElemLines[item.elemIndex].add(accepted)
        }
        val regionTexts = elements.indices.map { ei -> perElemLines[ei].joinToString("\n") }
        val assembled = NativeCore.assembleMomoScript(elements, regionTexts, width, height)
        val assembleMs = (System.nanoTime() - tAsm0) / 1_000_000L
        onLog?.invoke("组装 MomoScript: ${assembled.messages.size} 条消息, ${assembled.text.length} 字符 (${assembleMs}ms)")

        val elapsedMs = (System.nanoTime() - t0) / 1_000_000L
        return PageTranscriptionResult(
            momoScriptText = assembled.text,
            messages = assembled.messages,
            elements = elements,
            windowCount = windows.size,
            lineCount = allLines.size,
            escalatedLineCount = escalatedCount,
            adoptedMediumCount = adoptedMedCount,
            elapsedMs = elapsedMs,
            profile = PageProfileTiming(
                windowPlanMs = windowPlanMs,
                detPrepMs = detPrepNs / 1_000_000L,
                detInferMs = detInferNs / 1_000_000L,
                detDecodeMs = detDecodeMs,
                cvLinesMs = cvLinesMs,
                recSmallMs = recSmallMs,
                recSmallBatches = recSmallBatches,
                recMediumMs = recMediumMs,
                assembleMs = assembleMs,
                totalMs = elapsedMs,
            ),
        )
    }

    private fun runRecBatch(
        handle: NativeCore.MnnRecHandle,
        rgbBytes: ByteArray,
        width: Int,
        height: Int,
        lines: List<TextLineBox>,
        padHeights: List<Int>,
    ): List<CtcLineResult> {
        val batchSize = lines.size
        val prepared = ArrayList<Pair<FloatArray, Int>>(batchSize)
        var maxW = REC_BASE_W
        for (i in lines.indices) {
            val chw = NativeCore.prepareRecLine(
                rgbBytes = rgbBytes,
                imgW = width,
                imgH = height,
                line = lines[i],
                targetPadH = padHeights[i],
                dstH = REC_IMG_H,
                maxW = REC_MAX_W,
            )
            val rw = (chw.size / (3 * REC_IMG_H)).coerceAtLeast(1)
            if (rw > maxW) maxW = rw
            prepared.add(chw to rw)
        }
        val alignedW = minOf(REC_MAX_W, ((maxW + 15) / 16) * 16)
        val batchBuf = FloatArray(batchSize * 3 * REC_IMG_H * alignedW)

        for (b in 0 until batchSize) {
            val (chw, rw) = prepared[b]
            val copyW = minOf(rw, alignedW)
            val srcPlane = REC_IMG_H * rw
            val dstPlane = REC_IMG_H * alignedW
            val dstBatchBase = b * 3 * dstPlane
            for (c in 0 until 3) {
                val sBase = c * srcPlane
                val dBase = dstBatchBase + c * dstPlane
                for (y in 0 until REC_IMG_H) {
                    System.arraycopy(chw, sBase + y * rw, batchBuf, dBase + y * alignedW, copyW)
                }
            }
        }

        val raw = handle.runBatch(batchBuf, batchSize, alignedW)
        return List(batchSize) { b ->
            vocab.decodeCtcSlice(raw.indices, raw.scores, b * raw.seqLen, raw.seqLen)
        }
    }

    private fun preferMedium(small: CtcLineResult, med: CtcLineResult): Boolean {
        val st = small.text.trim()
        val mt = med.text.trim()
        if (mt.isEmpty()) return false
        if (st.isEmpty()) return true
        if (med.score >= small.score - 0.08f) return true
        if (med.minCharScore > small.minCharScore) return true
        return false
    }

    private fun isTextClass(classId: Int): Boolean =
        classId == 1 || classId == 2 || classId == 3 || classId == 4 || classId == 7 || classId == 10

    override fun close() {
        detHandle.close()
        recSmallHandle.close()
        recMediumHandle?.close()
    }

    companion object {
        const val BACKEND_AUTO_GPU_CPU = 0
        const val BACKEND_OPENCL_ALL = 1
        const val BACKEND_VULKAN_ALL = 2
        const val BACKEND_CPU_FP16 = 3

        private const val DET_INPUT_SIZE = 672
        private const val REC_IMG_H = 48
        private const val REC_BASE_W = 320
        private const val REC_MAX_W = 3200
        private const val REC_SMALL_BATCH_SIZE = 8
        private const val REC_MEDIUM_BATCH_SIZE = 4
        private const val MIN_TEXT_SCORE = 0.55f
        private const val CASCADE_SCORE_THRESHOLD = 0.85f
        private const val CASCADE_MIN_CHAR_THRESHOLD = 0.65f
    }
}
