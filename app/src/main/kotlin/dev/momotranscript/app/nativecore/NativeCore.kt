package dev.momotranscript.app.nativecore

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

data class WindowBox(
    val x0: Int,
    val y0: Int,
    val x1: Int,
    val y1: Int,
) {
    val width: Int get() = x1 - x0
    val height: Int get() = y1 - y0
}

data class TextLineBox(
    val inkX0: Int,
    val inkY0: Int,
    val inkX1: Int,
    val inkY1: Int,
    val x0: Int,
    val y0: Int,
    val x1: Int,
    val y1: Int,
    val isLightForeground: Boolean,
) {
    val inkWidth: Int get() = inkX1 - inkX0
    val inkHeight: Int get() = inkY1 - inkY0
    val width: Int get() = x1 - x0
    val height: Int get() = y1 - y0
}

@Serializable
data class NativeDetectedElement(
    val id: String,
    val classId: Int,
    val score: Float,
    val box: List<Int>,
    val parentId: String = "",
)

@Serializable
data class NativeAssembledMessage(
    val kind: String,
    val speaker: String = "",
    val isRight: Boolean = false,
    val isFirstTurn: Boolean = false,
    val content: String = "",
    val note: String = "",
    val box: List<Int> = emptyList(),
)

@Serializable
data class NativeAssemblyResult(
    val text: String = "",
    val messages: List<NativeAssembledMessage> = emptyList(),
)

/**
 * JNI bridge into ``libmomotranscript_core.so`` (C++20 core).
 */
object NativeCore {
    private val json = Json { ignoreUnknownKeys = true }
    private var loaded: Boolean = false
    private var loadError: String? = null
    private var pluginDiag: String = ""

    init {
        loaded = try {
            // Load shared C++ runtime + MNN core + optional OpenCL/Vulkan GPU plugins before momotranscript_core
            runCatching { System.loadLibrary("c++_shared") }
            System.loadLibrary("MNN")
            System.loadLibrary("MNN_Express")
            val clRes = runCatching { System.loadLibrary("MNN_CL") }
            val vkRes = runCatching { System.loadLibrary("MNN_Vulkan") }
            pluginDiag = "CL=${if (clRes.isSuccess) "so" else "no"} VK=${if (vkRes.isSuccess) "so" else "no"}"
            System.loadLibrary("momotranscript_core")
            true
        } catch (e: UnsatisfiedLinkError) {
            loadError = e.message ?: e.toString()
            false
        }
    }

    val isAvailable: Boolean
        get() = loaded

    fun coreVersion(): String =
        if (loaded) nativeCoreVersion() else "Native core unavailable (${loadError ?: "JVM test"})"

    fun planFixedWindows(width: Int, height: Int): List<WindowBox> {
        if (!loaded || width <= 0 || height <= 0) return emptyList()
        return unpackWindows(nativePlanFixedWindows(width, height))
    }

    fun planAdaptiveWindows(rgbBytes: ByteArray, width: Int, height: Int): List<WindowBox> {
        if (!loaded || width <= 0 || height <= 0) return emptyList()
        return unpackWindows(nativePlanAdaptiveWindows(rgbBytes, width, height))
    }

    fun prepareDetInput(
        rgbBytes: ByteArray,
        imgW: Int,
        imgH: Int,
        win: WindowBox,
        dstSize: Int = 672,
    ): FloatArray {
        check(loaded) { "Native library momotranscript_core is not loaded" }
        return nativePrepareDetInput(rgbBytes, imgW, imgH, win.x0, win.y0, win.x1, win.y1, dstSize)
    }

    fun decodeAndFuseDetections(
        bboxesFlat: FloatArray,
        logitsFlat: FloatArray,
        numQueries: Int,
        windows: List<WindowBox>,
        imgW: Int,
        imgH: Int,
    ): List<NativeDetectedElement> {
        check(loaded) { "Native library momotranscript_core is not loaded" }
        val winFlat = IntArray(windows.size * 4)
        for (i in windows.indices) {
            val w = windows[i]
            winFlat[i * 4 + 0] = w.x0
            winFlat[i * 4 + 1] = w.y0
            winFlat[i * 4 + 2] = w.x1
            winFlat[i * 4 + 3] = w.y1
        }
        val raw = nativeDecodeAndFuseDetections(bboxesFlat, logitsFlat, numQueries, winFlat, imgW, imgH)
        return json.decodeFromString(raw)
    }

    fun segmentLines(
        rgbBytes: ByteArray,
        imgW: Int,
        imgH: Int,
        x0: Int,
        y0: Int,
        x1: Int,
        y1: Int,
        tight: Boolean,
    ): List<TextLineBox> {
        check(loaded) { "Native library momotranscript_core is not loaded" }
        val flat = nativeSegmentLines(rgbBytes, imgW, imgH, x0, y0, x1, y1, tight)
        val count = flat.size / 9
        return List(count) { i ->
            val b = i * 9
            TextLineBox(
                inkX0 = flat[b],
                inkY0 = flat[b + 1],
                inkX1 = flat[b + 2],
                inkY1 = flat[b + 3],
                x0 = flat[b + 4],
                y0 = flat[b + 5],
                x1 = flat[b + 6],
                y1 = flat[b + 7],
                isLightForeground = flat[b + 8] != 0,
            )
        }
    }

    fun prepareRecLine(
        rgbBytes: ByteArray,
        imgW: Int,
        imgH: Int,
        line: TextLineBox,
        targetPadH: Int = 0,
        dstH: Int = 48,
        maxW: Int = 3200,
    ): FloatArray {
        check(loaded) { "Native library momotranscript_core is not loaded" }
        return nativePrepareRecLine(
            rgbBytes, imgW, imgH,
            line.x0, line.y0, line.x1, line.y1,
            line.isLightForeground,
            targetPadH, dstH, maxW,
        )
    }

    fun assembleMomoScript(
        elements: List<NativeDetectedElement>,
        texts: List<String>,
        imgW: Int,
        imgH: Int,
    ): NativeAssemblyResult {
        check(loaded) { "Native library momotranscript_core is not loaded" }
        val n = elements.size
        val ids = Array(n) { elements[it].id }
        val cids = IntArray(n) { elements[it].classId }
        val scs = FloatArray(n) { elements[it].score }
        val boxes = IntArray(n * 4)
        val pids = Array(n) { elements[it].parentId }
        val txts = Array(n) { texts.getOrElse(it) { "" } }
        for (i in 0 until n) {
            val b = elements[i].box
            boxes[i * 4 + 0] = b[0]
            boxes[i * 4 + 1] = b[1]
            boxes[i * 4 + 2] = b[2]
            boxes[i * 4 + 3] = b[3]
        }
        val raw = nativeAssembleMomoScript(ids, cids, scs, boxes, pids, txts, imgW, imgH)
        return json.decodeFromString(raw)
    }

    fun inspectAvifsPack(path: String): String =
        if (loaded) nativeInspectAvifsPack(path) else """{"ok":false,"error":"native library not loaded"}"""

    fun splitAvatarBox(x0: Int, y0: Int, x1: Int, y1: Int, maxMembers: Int = 5): List<WindowBox> {
        if (!loaded) return listOf(WindowBox(x0, y0, x1, y1))
        return unpackWindows(nativeSplitAvatarBox(x0, y0, x1, y1, maxMembers))
    }

    data class CcipPreparedInput(
        val nchw: FloatArray,
        val std: Float,
        val aspect: Float,
    )

    fun prepareCcipInput(
        rgbBytes: ByteArray,
        imgW: Int,
        imgH: Int,
        box: WindowBox,
        circleMask: Boolean,
        trimBorders: Boolean,
        dstSize: Int = 384,
        normHalf: Boolean = false,
    ): CcipPreparedInput {
        check(loaded) { "Native library momotranscript_core is not loaded" }
        val meta = FloatArray(2)
        val nchw = nativePrepareCcipInput(
            rgbBytes, imgW, imgH,
            box.x0, box.y0, box.x1, box.y1,
            circleMask, trimBorders, dstSize, normHalf, meta,
        )
        return CcipPreparedInput(nchw = nchw, std = meta[0], aspect = meta[1])
    }

    fun computeRgbThumbnail(
        rgbBytes: ByteArray,
        imgW: Int,
        imgH: Int,
        box: WindowBox,
        circleMask: Boolean,
        thumbSize: Int = 16,
    ): FloatArray {
        check(loaded) { "Native library momotranscript_core is not loaded" }
        return nativeComputeRgbThumbnail(
            rgbBytes, imgW, imgH,
            box.x0, box.y0, box.x1, box.y1,
            circleMask, thumbSize,
        )
    }

    data class Stage2SkinResult(
        val bestIndex: Int,
        val diffZncc: Float,
        val structZncc: Float,
    )

    fun scoreStage2Skin(
        rgbBytes: ByteArray,
        imgW: Int,
        imgH: Int,
        box: WindowBox,
        stackRgb: ByteArray,
        count: Int,
        tplSize: Int,
        varWeights: FloatArray,
    ): Stage2SkinResult {
        check(loaded) { "Native library momotranscript_core is not loaded" }
        val res = nativeScoreStage2Skin(
            rgbBytes, imgW, imgH,
            box.x0, box.y0, box.x1, box.y1,
            stackRgb, count, tplSize, varWeights,
        )
        return Stage2SkinResult(
            bestIndex = res[0].toInt(),
            diffZncc = res[1],
            structZncc = res[2],
        )
    }

    fun scoreEmojiStructure(
        rgbBytes: ByteArray,
        imgW: Int,
        imgH: Int,
        box: WindowBox,
        tplRgb: ByteArray,
        tplSize: Int,
    ): Float {
        check(loaded) { "Native library momotranscript_core is not loaded" }
        return nativeScoreEmojiStructure(
            rgbBytes, imgW, imgH,
            box.x0, box.y0, box.x1, box.y1,
            tplRgb, tplSize,
        )
    }

    // ============================= MNN 3.6.2 Runtime API =============================

    const val MNN_BACKEND_CPU_FP16 = 0
    const val MNN_BACKEND_OPENCL = 1
    const val MNN_BACKEND_VULKAN = 2

    fun mnnDeviceInfo(): String =
        if (loaded) "${nativeMnnDeviceInfo()} | $pluginDiag"
        else "MNN unavailable (${loadError ?: "not loaded"})"

    class MnnDetHandle(
        modelPath: String,
        backendReq: Int,
        threads: Int,
        cachePath: String,
    ) : java.io.Closeable {
        private var handle: Long
        val activeBackend: String

        init {
            check(loaded) { "Native library momotranscript_core is not loaded" }
            val info = arrayOfNulls<String>(1)
            handle = nativeMnnDetCreate(modelPath, backendReq, threads, cachePath, info)
            require(handle != 0L) { info[0] ?: "创建 MNN 检测会话失败" }
            activeBackend = info[0] ?: "CPU(FP16)"
        }

        fun runWindow(inputNchw: FloatArray, outBboxes: FloatArray, outLogits: FloatArray) {
            check(handle != 0L) { "MnnDetHandle already closed" }
            check(nativeMnnDetRun(handle, inputNchw, outBboxes, outLogits)) { "MNN 检测推理失败" }
        }

        override fun close() {
            if (handle != 0L) {
                nativeMnnDetDestroy(handle)
                handle = 0L
            }
        }
    }

    data class MnnRecRawOutput(
        val indices: IntArray,
        val scores: FloatArray,
        val seqLen: Int,
    )

    class MnnRecHandle(
        modelPath: String,
        backendReq: Int,
        threads: Int,
        cachePath: String,
    ) : java.io.Closeable {
        private var handle: Long
        val activeBackend: String

        init {
            check(loaded) { "Native library momotranscript_core is not loaded" }
            val info = arrayOfNulls<String>(1)
            handle = nativeMnnRecCreate(modelPath, backendReq, threads, cachePath, info)
            require(handle != 0L) { info[0] ?: "创建 MNN 识别会话失败" }
            activeBackend = info[0] ?: "CPU(FP16)"
        }

        fun runBatch(inputNchw: FloatArray, batchSize: Int, width: Int): MnnRecRawOutput {
            check(handle != 0L) { "MnnRecHandle already closed" }
            val outPair = arrayOfNulls<Any>(2)
            val seqLen = nativeMnnRecRun(handle, inputNchw, batchSize, width, outPair)
            check(seqLen > 0) { "MNN 识别推理失败" }
            return MnnRecRawOutput(
                indices = outPair[0] as IntArray,
                scores = outPair[1] as FloatArray,
                seqLen = seqLen,
            )
        }

        override fun close() {
            if (handle != 0L) {
                nativeMnnRecDestroy(handle)
                handle = 0L
            }
        }
    }

    data class UnifiedEncodeOutput(
        val idEmb: FloatArray,
        val exprLogits: FloatArray? = null,
    )

    class MnnCcipHandle(
        modelPath: String,
        backendReq: Int,
        threads: Int,
        cachePath: String,
    ) : java.io.Closeable {
        private var handle: Long
        val activeBackend: String
        val inputSize: Int
        val idDim: Int
        val exprDim: Int
        val isUnified: Boolean get() = exprDim > 0

        init {
            check(loaded) { "Native library momotranscript_core is not loaded" }
            val info = arrayOfNulls<String>(4)
            handle = nativeMnnCcipCreate(modelPath, backendReq, threads, cachePath, info)
            require(handle != 0L) { info[0] ?: "创建 MNN 视觉编码会话失败" }
            activeBackend = info[0] ?: "CPU(FP16)"
            inputSize = info[1]?.toIntOrNull() ?: 384
            idDim = info[2]?.toIntOrNull() ?: 768
            exprDim = info[3]?.toIntOrNull() ?: 0
        }

        fun runUnified(inputNchw: FloatArray): UnifiedEncodeOutput {
            check(handle != 0L) { "MnnCcipHandle already closed" }
            val outId = FloatArray(idDim)
            val outExpr = if (exprDim > 0) FloatArray(exprDim) else null
            check(nativeMnnCcipRun(handle, inputNchw, outId, outExpr)) { "MNN 视觉编码推理失败" }
            return UnifiedEncodeOutput(idEmb = outId, exprLogits = outExpr)
        }

        fun runSingle(inputNchw: FloatArray): FloatArray = runUnified(inputNchw).idEmb

        override fun close() {
            if (handle != 0L) {
                nativeMnnCcipDestroy(handle)
                handle = 0L
            }
        }
    }

    private fun unpackWindows(flat: IntArray): List<WindowBox> {
        val count = flat.size / 4
        return List(count) { i ->
            val base = i * 4
            WindowBox(flat[base], flat[base + 1], flat[base + 2], flat[base + 3])
        }
    }

    private external fun nativeCoreVersion(): String
    private external fun nativePlanFixedWindows(width: Int, height: Int): IntArray
    private external fun nativePlanAdaptiveWindows(rgbBytes: ByteArray, width: Int, height: Int): IntArray
    private external fun nativePrepareDetInput(
        rgbBytes: ByteArray,
        imgW: Int,
        imgH: Int,
        x0: Int,
        y0: Int,
        x1: Int,
        y1: Int,
        dstSize: Int,
    ): FloatArray

    private external fun nativeDecodeAndFuseDetections(
        bboxesFlat: FloatArray,
        logitsFlat: FloatArray,
        numQueries: Int,
        windowsFlat: IntArray,
        imgW: Int,
        imgH: Int,
    ): String

    private external fun nativeSegmentLines(
        rgbBytes: ByteArray,
        imgW: Int,
        imgH: Int,
        x0: Int,
        y0: Int,
        x1: Int,
        y1: Int,
        tight: Boolean,
    ): IntArray

    private external fun nativePrepareRecLine(
        rgbBytes: ByteArray,
        imgW: Int,
        imgH: Int,
        x0: Int,
        y0: Int,
        x1: Int,
        y1: Int,
        isLight: Boolean,
        targetPadH: Int,
        dstH: Int,
        maxW: Int,
    ): FloatArray

    private external fun nativeAssembleMomoScript(
        ids: Array<String>,
        classIds: IntArray,
        scores: FloatArray,
        boxes: IntArray,
        parentIds: Array<String>,
        texts: Array<String>,
        imgW: Int,
        imgH: Int,
    ): String

    private external fun nativeInspectAvifsPack(path: String): String

    private external fun nativeSplitAvatarBox(
        x0: Int,
        y0: Int,
        x1: Int,
        y1: Int,
        maxMembers: Int,
    ): IntArray

    private external fun nativePrepareCcipInput(
        rgbBytes: ByteArray,
        imgW: Int,
        imgH: Int,
        x0: Int,
        y0: Int,
        x1: Int,
        y1: Int,
        circleMask: Boolean,
        trimBorders: Boolean,
        dstSize: Int,
        normHalf: Boolean,
        outMeta: FloatArray,
    ): FloatArray

    private external fun nativeComputeRgbThumbnail(
        rgbBytes: ByteArray,
        imgW: Int,
        imgH: Int,
        x0: Int,
        y0: Int,
        x1: Int,
        y1: Int,
        circleMask: Boolean,
        thumbSize: Int,
    ): FloatArray

    private external fun nativeScoreStage2Skin(
        rgbBytes: ByteArray,
        imgW: Int,
        imgH: Int,
        x0: Int,
        y0: Int,
        x1: Int,
        y1: Int,
        stackRgb: ByteArray,
        count: Int,
        tplSize: Int,
        varWeights: FloatArray,
    ): FloatArray

    private external fun nativeScoreEmojiStructure(
        rgbBytes: ByteArray,
        imgW: Int,
        imgH: Int,
        x0: Int,
        y0: Int,
        x1: Int,
        y1: Int,
        tplRgb: ByteArray,
        tplSize: Int,
    ): Float

    private external fun nativeMnnDeviceInfo(): String
    private external fun nativeMnnDetCreate(
        modelPath: String,
        backendReq: Int,
        threads: Int,
        cachePath: String,
        outInfo: Array<String?>,
    ): Long
    private external fun nativeMnnDetRun(
        handle: Long,
        inputNchw: FloatArray,
        outBboxes: FloatArray,
        outLogits: FloatArray,
    ): Boolean
    private external fun nativeMnnDetDestroy(handle: Long)

    private external fun nativeMnnRecCreate(
        modelPath: String,
        backendReq: Int,
        threads: Int,
        cachePath: String,
        outInfo: Array<String?>,
    ): Long
    private external fun nativeMnnRecRun(
        handle: Long,
        inputNchw: FloatArray,
        batchSize: Int,
        width: Int,
        outPair: Array<Any?>,
    ): Int
    private external fun nativeMnnRecDestroy(handle: Long)

    private external fun nativeMnnCcipCreate(
        modelPath: String,
        backendReq: Int,
        threads: Int,
        cachePath: String,
        outInfo: Array<String?>,
    ): Long
    private external fun nativeMnnCcipRun(
        handle: Long,
        inputNchw: FloatArray,
        outId: FloatArray,
        outExprLogits: FloatArray?,
    ): Boolean
    private external fun nativeMnnCcipDestroy(handle: Long)
}
