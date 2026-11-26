package dev.momotranscript.app.avatar

import dev.momotranscript.app.engine.OnnxTranscriptionEngine
import dev.momotranscript.app.nativecore.NativeCore
import dev.momotranscript.app.nativecore.WindowBox
import java.io.Closeable
import java.io.File
import kotlin.math.abs
import kotlin.math.sqrt

data class AvatarVisualMatch(
    val status: String, // "matched" | "unknown_avatar" | "encode_failed"
    val character: String = "",
    val skin: String = "",
    val assetId: String = "",
    val source: String = "",
    val expressionId: String = "",
    val expressionSummary: String = "",
    val customClusterId: String = "",
    val confidence: Float = 0f,
    val subBox: WindowBox,
)

data class InsertVisualMatch(
    val kind: String,   // "emoji" | "charface" | "illustration"
    val status: String, // "matched" | "open_set_rejected" | "encode_failed"
    val character: String = "",
    val expressionId: String = "",
    val expressionSummary: String = "",
    val emojiId: String = "",
    val source: String = "",
    val packV2Preset: String? = null,
    val packV2Ordinal: Int = 0,
    val confidence: Float = 0f,
)

data class CcipInitProfile(
    val galleryLoadMs: Long,
    val ccipLoadMs: Long,
    val intraOpThreads: Int,
    val skinCount: Int,
    val charCount: Int,
    val emojiCount: Int,
    val ep: String = "CPU",
)

data class CcipCallStats(
    val inputBoxes: Int = 0,
    val uniqueCrops: Int = 0,
    val cacheHits: Int = 0,
    val failedCrops: Int = 0,
    val inferMs: Long = 0L,
    val stage2Ms: Long = 0L,
    val totalMs: Long = 0L,
)

/**
 * Two-stage on-device visual matcher using MNN 3.6.2 CCIP (`mnn/ccip_caformer_24_w8.mnn`)
 * + Stage-1 SVD-128 cosine search + Stage-2 `MMTAVIF2` weighted ZNCC.
 */
class OnnxAvatarMatcher(
    bundleRoot: File,
    private val intraOpThreads: Int = Runtime.getRuntime().availableProcessors().coerceIn(2, 4),
    private val backendMode: Int = OnnxTranscriptionEngine.BACKEND_AUTO_GPU_CPU,
    private val cacheDir: File? = null,
    includeCfid6: Boolean = false,
) : Closeable {

    val gallery: AvatarGallery
    val initProfile: CcipInitProfile
    var lastAvatarStats: CcipCallStats = CcipCallStats()
        private set
    var lastInsertStats: CcipCallStats = CcipCallStats()
        private set
    private val ccipHandle: NativeCore.MnnCcipHandle

    // Concatenated [char_emb; skin_emb] for avatar Stage-1 matching
    private val nChars: Int
    private val avRows: Int
    private val embDim: Int
    private val avEmbHq: FloatArray
    private val avEmbLr: FloatArray?

    // Batch-wide custom OC character prototypes so the same OC keeps one custom ID across pages
    private val customProtos = ArrayList<FloatArray>()
    // Cross-page thumbnail -> (embedding, expressionSummary) cache
    internal data class CachedThumbEmb(val thumb: FloatArray, val emb: FloatArray, val exprSummary: String = "")
    private val avatarEmbCache = ArrayList<CachedThumbEmb>()
    private val insertEmbCache = ArrayList<CachedThumbEmb>()
    private val expressionDecoder: ExpressionDecoder?

    init {
        val avatarDir = File(bundleRoot, "avatar")
        val unifiedModelFile = dev.momotranscript.app.model.ModelPackCatalog.resolveFile(
            bundleRoot,
            dev.momotranscript.app.model.ModelPackCatalog.UNIFIED_AVATAR_MODEL,
            dev.momotranscript.app.model.ModelPackCatalog.UNIFIED_AVATAR_LEGACY,
        )
        val unifiedEmbFile = File(avatarDir, "stage1_embeddings_pe256.npz")
        val useUnified = unifiedModelFile.isFile && unifiedModelFile.length() > 0L &&
            unifiedEmbFile.isFile && unifiedEmbFile.length() > 0L

        val modelFile = if (useUnified) {
            unifiedModelFile
        } else {
            dev.momotranscript.app.model.ModelPackCatalog.resolveFile(
                bundleRoot,
                dev.momotranscript.app.model.ModelPackCatalog.CCIP_MODEL,
                dev.momotranscript.app.model.ModelPackCatalog.CCIP_LEGACY,
            )
        }
        require(modelFile.isFile) { "缺少 MNN 视觉编码模型：${modelFile.absolutePath}（请前往「模型」页更新视觉包）" }
        val tGal = System.nanoTime()
        gallery = AvatarGallery(avatarDir, preferPe256 = useUnified, includeCfid6 = includeCfid6 && useUnified)
        val galMs = (System.nanoTime() - tGal) / 1_000_000L

        val ccipReq = when (backendMode) {
            // PE-Core-S/16 ViT + W4-HQQ runs 3-4x faster on ARMv8.2-FP16 CPU and avoids Adreno OpenCL FP16
            // LayerNorm overflow; legacy 384px CCIP still uses OpenCL in auto mode.
            OnnxTranscriptionEngine.BACKEND_AUTO_GPU_CPU ->
                if (useUnified) NativeCore.MNN_BACKEND_CPU_FP16 else NativeCore.MNN_BACKEND_OPENCL
            OnnxTranscriptionEngine.BACKEND_OPENCL_ALL ->
                NativeCore.MNN_BACKEND_OPENCL
            OnnxTranscriptionEngine.BACKEND_VULKAN_ALL ->
                NativeCore.MNN_BACKEND_VULKAN
            else -> NativeCore.MNN_BACKEND_CPU_FP16
        }
        val cachePath = if (cacheDir != null && ccipReq != NativeCore.MNN_BACKEND_CPU_FP16) {
            val dir = File(cacheDir, "mnn_cache").apply { mkdirs() }
            val tag = modelFile.nameWithoutExtension
            File(dir, "${tag}_fp16_b${ccipReq}.bin").absolutePath
        } else {
            ""
        }

        val tSess = System.nanoTime()
        ccipHandle = NativeCore.MnnCcipHandle(
            modelPath = modelFile.absolutePath,
            backendReq = ccipReq,
            threads = intraOpThreads,
            cachePath = cachePath,
        )
        val sessMs = (System.nanoTime() - tSess) / 1_000_000L
        expressionDecoder = if (ccipHandle.isUnified) {
            ExpressionDecoder.loadIfPresent(File(avatarDir, "vocab_48.json"))
        } else {
            null
        }
        val encTag = when {
            ccipHandle.isUnified && expressionDecoder != null -> "PE256+48Expr"
            ccipHandle.isUnified -> "PE256"
            gallery.projMatrix != null -> "SVD${gallery.projMatrix.cols}"
            else -> "FP16-768"
        }

        initProfile = CcipInitProfile(
            galleryLoadMs = galMs,
            ccipLoadMs = sessMs,
            intraOpThreads = intraOpThreads,
            skinCount = gallery.catalog.skins.size,
            charCount = gallery.catalog.chars.size,
            emojiCount = gallery.catalog.emojis.size,
            ep = "${ccipHandle.activeBackend}+$encTag",
        )

        nChars = gallery.charEmb.rows
        embDim = gallery.charEmb.cols.coerceAtLeast(1)
        avRows = gallery.charEmb.rows + gallery.skinEmb.rows
        avEmbHq = FloatArray(avRows * embDim)
        System.arraycopy(gallery.charEmb.data, 0, avEmbHq, 0, gallery.charEmb.data.size)
        System.arraycopy(gallery.skinEmb.data, 0, avEmbHq, gallery.charEmb.data.size, gallery.skinEmb.data.size)

        val lr = gallery.charEmbLowres
        avEmbLr = if (lr != null && lr.rows == nChars) {
            FloatArray(avRows * embDim).also { arr ->
                System.arraycopy(lr.data, 0, arr, 0, lr.data.size)
                System.arraycopy(gallery.skinEmb.data, 0, arr, lr.data.size, gallery.skinEmb.data.size)
            }
        } else {
            null
        }
    }

    /**
     * Matches a list of avatar boxes (automatically splitting wide/tall multi-avatar groups)
     * with intra-page 16x16 thumbnail deduplication so repeated avatars are encoded only once.
     */
    fun matchAvatarBoxes(
        rgbBytes: ByteArray,
        imgW: Int,
        imgH: Int,
        avatarBoxes: List<WindowBox>,
        speakerNames: List<String?>,
    ): List<List<AvatarVisualMatch>> {
        if (avatarBoxes.isEmpty()) {
            lastAvatarStats = CcipCallStats()
            return emptyList()
        }
        val tCall0 = System.nanoTime()

        // Expand multi-avatar boxes into individual sub-boxes
        val expandedBoxes = ArrayList<WindowBox>()
        val expandedSpeakers = ArrayList<String?>()
        val groupSlices = ArrayList<IntRange>(avatarBoxes.size)
        for (i in avatarBoxes.indices) {
            val subs = NativeCore.splitAvatarBox(
                avatarBoxes[i].x0, avatarBoxes[i].y0, avatarBoxes[i].x1, avatarBoxes[i].y1,
            )
            val start = expandedBoxes.size
            for (sb in subs) {
                expandedBoxes.add(sb)
                expandedSpeakers.add(speakerNames.getOrNull(i))
            }
            groupSlices.add(start until expandedBoxes.size)
        }

        if (expandedBoxes.isEmpty()) return List(avatarBoxes.size) { emptyList() }

        // Shift-tolerant thumbnail dedup (±1px on 16x16, MAE <= 5.5) + cross-page cache lookup
        val (repIndices, itemToRep, repThumbs) = deduplicateByThumbnail(
            rgbBytes, imgW, imgH, expandedBoxes, circleMask = true, maeThreshold = 5.5f, shiftTolerant = true,
        )

        // Encode only cache-miss representative crops
        val repPrepared = repIndices.map { idx ->
            NativeCore.prepareCcipInput(
                rgbBytes, imgW, imgH, expandedBoxes[idx],
                circleMask = true, trimBorders = false,
                dstSize = ccipHandle.inputSize, normHalf = ccipHandle.isUnified,
            )
        }
        val repEmbeddings = arrayOfNulls<FloatArray>(repIndices.size)
        val repExprSummaries = Array(repIndices.size) { "" }
        val missSlots = ArrayList<Int>()
        var cacheHits = 0
        for (slot in repIndices.indices) {
            val hit = lookupThumbCache(avatarEmbCache, repThumbs[slot], 5.5f, shiftTolerant = true)
            if (hit != null) {
                repEmbeddings[slot] = hit.emb
                repExprSummaries[slot] = hit.exprSummary
                cacheHits++
            } else {
                missSlots.add(slot)
            }
        }
        val tInf0 = System.nanoTime()
        val failedAvatars = if (missSlots.isNotEmpty()) {
            encodeAndCacheMisses(
                missSlots = missSlots,
                inputsNchw = missSlots.map { repPrepared[it].nchw },
                repThumbs = repThumbs,
                repEmbeddings = repEmbeddings,
                repExprSummaries = repExprSummaries,
                cache = avatarEmbCache,
                maxCacheSize = MAX_THUMB_CACHE,
                isUnified = ccipHandle.isUnified,
                project = gallery::projectQuery,
                decoder = expressionDecoder,
                runEncoder = ccipHandle::runUnified,
            )
        } else {
            0
        }
        val inferMs = (System.nanoTime() - tInf0) / 1_000_000L
        var stage2Ns = 0L

        // Intra-page prototype clustering
        class Cluster(
            val sumVec: DoubleArray,
            var proto: FloatArray,
            var count: Int,
            var alphaSum: Float,
            var prior: Set<String>,
        )

        val clusters = ArrayList<Cluster>()
        val itemToCluster = IntArray(expandedBoxes.size) { -1 }
        val resolvedPriors = expandedSpeakers.map { gallery.resolveSpeakerCanons(it) }

        for (i in expandedBoxes.indices) {
            val repSlot = itemToRep[i]
            val vec = repEmbeddings[repSlot]
            if (repPrepared[repSlot].std < 8.0f || vec == null) {
                itemToCluster[i] = -1
                continue
            }
            val prior = resolvedPriors[i]
            val side = minOf(expandedBoxes[i].width, expandedBoxes[i].height).toFloat()
            val alpha = when {
                side <= 64f -> 1.0f
                side >= 140f -> 0.0f
                else -> (140f - side) / (140f - 64f)
            }

            var bestCi = -1
            var bestSim = -1f
            for (ci in clusters.indices) {
                val cl = clusters[ci]
                val sim = dot(vec, cl.proto)
                val samePrior = prior.isNotEmpty() && cl.prior.isNotEmpty() && prior == cl.prior
                val thr = if (samePrior) 0.84f else 0.91f
                if (sim >= thr && sim > bestSim) {
                    bestSim = sim
                    bestCi = ci
                }
            }
            if (bestCi < 0) {
                val sum = DoubleArray(embDim) { vec[it].toDouble() }
                clusters.add(Cluster(sum, vec.copyOf(), 1, alpha, prior))
                itemToCluster[i] = clusters.size - 1
            } else {
                val cl = clusters[bestCi]
                for (d in 0 until embDim) cl.sumVec[d] += vec[d].toDouble()
                cl.count++
                cl.alphaSum += alpha
                if (cl.prior.isEmpty() && prior.isNotEmpty()) cl.prior = prior
                cl.proto = normalizeDoubleToFloat(cl.sumVec)
                itemToCluster[i] = bestCi
            }
        }

        // Precompute Stage-1 similarity vectors per cluster
        val clusterHqSims = clusters.map { matVecDot(avEmbHq, avRows, embDim, it.proto) }
        val clusterLrSims = if (avEmbLr != null) {
            clusters.map { matVecDot(avEmbLr, avRows, embDim, it.proto) }
        } else {
            null
        }

        val expandedResults = ArrayList<AvatarVisualMatch>(expandedBoxes.size)
        for (i in expandedBoxes.indices) {
            val box = expandedBoxes[i]
            val ci = itemToCluster[i]
            if (ci < 0) {
                val repSlot = itemToRep[i]
                val isEncodeFailure = repPrepared[repSlot].std >= 8.0f && repEmbeddings[repSlot] == null
                expandedResults.add(
                    AvatarVisualMatch(
                        status = if (isEncodeFailure) "encode_failed" else "unknown_avatar",
                        customClusterId = if (isEncodeFailure) "encode_failed" else "blank_avatar",
                        confidence = 0f,
                        subBox = box,
                    ),
                )
                continue
            }

            val cl = clusters[ci]
            val alpha = cl.alphaSum / cl.count.toFloat()
            val hq = clusterHqSims[ci]
            val lr = clusterLrSims?.get(ci)
            val baseSims = if (lr != null && alpha > 0f) {
                FloatArray(avRows) { j -> (1f - alpha) * hq[j] + alpha * lr[j] }
            } else {
                hq
            }

            val priorCanons = resolvedPriors[i].ifEmpty { cl.prior }
            var bestJ = 0
            var bestAdj = Float.NEGATIVE_INFINITY
            for (j in 0 until avRows) {
                val candChar = candidateCharacter(j)
                val adj = baseSims[j] + if (priorCanons.isNotEmpty() && candChar in priorCanons) NAME_PRIOR_BONUS else 0f
                if (adj > bestAdj) {
                    bestAdj = adj
                    bestJ = j
                }
            }

            val rawSim = baseSims[bestJ]
            val bestChar = candidateCharacter(bestJ)
            val priorApplied = priorCanons.isNotEmpty() && bestChar in priorCanons
            val effThreshold = AVATAR_THRESHOLD - if (priorApplied) 0.04f else 0f

            val exprSum = repExprSummaries[itemToRep[i]]
            if (rawSim < effThreshold) {
                val cid = assignCustomCluster(cl.proto)
                expandedResults.add(
                    AvatarVisualMatch(
                        status = "unknown_avatar",
                        expressionSummary = exprSum,
                        customClusterId = cid,
                        confidence = rawSim.coerceAtLeast(0f),
                        subBox = box,
                    ),
                )
                continue
            }

            if (bestJ < nChars) {
                val ch = gallery.catalog.chars[bestJ]
                expandedResults.add(
                    AvatarVisualMatch(
                        status = "matched",
                        character = ch.character,
                        skin = ch.skin,
                        assetId = ch.assetId,
                        source = ch.source,
                        expressionSummary = exprSum,
                        confidence = rawSim,
                        subBox = box,
                    ),
                )
            } else {
                val skinIdx = bestJ - nChars
                val sk = gallery.catalog.skins[skinIdx]
                var exprId = sk.prototypeExpressionId
                val tSt2 = System.nanoTime()
                val decoded = gallery.getSkinTemplates(sk.skinIndex)
                if (decoded != null) {
                    val st2 = NativeCore.scoreStage2Skin(
                        rgbBytes, imgW, imgH, box,
                        decoded.stackRgb, decoded.count, decoded.tplSize, decoded.varWeights,
                    )
                    if (st2.structZncc >= CHARFACE_STRUCT_THRESHOLD && st2.bestIndex in sk.expressions.indices) {
                        exprId = sk.expressions[st2.bestIndex].expressionId
                    }
                }
                stage2Ns += System.nanoTime() - tSt2
                expandedResults.add(
                    AvatarVisualMatch(
                        status = "matched",
                        character = sk.character,
                        skin = sk.skinId,
                        assetId = sk.prototypeExpressionId,
                        source = sk.source,
                        expressionId = exprId,
                        expressionSummary = exprSum,
                        confidence = rawSim,
                        subBox = box,
                    ),
                )
            }
        }

        lastAvatarStats = CcipCallStats(
            inputBoxes = expandedBoxes.size,
            uniqueCrops = missSlots.size,
            cacheHits = cacheHits,
            failedCrops = failedAvatars,
            inferMs = inferMs,
            stage2Ms = stage2Ns / 1_000_000L,
            totalMs = (System.nanoTime() - tCall0) / 1_000_000L,
        )
        return groupSlices.map { range -> expandedResults.subList(range.first, range.last + 1) }
    }

    /**
     * Matches `insert_image` crops against official Emoji stickers and CharFace expression diffs
     * (Stage-1 CCIP + Stage-2 `MMTAVIF1` ZNCC), leaving CG illustrations as `open_set_rejected`.
     */
    fun matchInsertBoxes(
        rgbBytes: ByteArray,
        imgW: Int,
        imgH: Int,
        insertBoxes: List<WindowBox>,
        contextSpeakers: List<String?>,
    ): List<InsertVisualMatch> {
        if (insertBoxes.isEmpty()) {
            lastInsertStats = CcipCallStats()
            return emptyList()
        }
        val tCall0 = System.nanoTime()
        val results = arrayOfNulls<InsertVisualMatch>(insertBoxes.size)
        val eligibleIndices = ArrayList<Int>()
        val eligiblePrepared = ArrayList<NativeCore.CcipPreparedInput>()

        for (i in insertBoxes.indices) {
            val prep = NativeCore.prepareCcipInput(
                rgbBytes, imgW, imgH, insertBoxes[i],
                circleMask = false, trimBorders = true,
                dstSize = ccipHandle.inputSize, normHalf = ccipHandle.isUnified,
            )
            if (prep.aspect > 1.85f || prep.aspect < 1f / 1.85f || prep.std < 8.0f) {
                results[i] = InsertVisualMatch(kind = "illustration", status = "open_set_rejected")
            } else {
                eligibleIndices.add(i)
                eligiblePrepared.add(prep)
            }
        }
        if (eligibleIndices.isEmpty()) {
            lastInsertStats = CcipCallStats(
                inputBoxes = insertBoxes.size,
                uniqueCrops = 0,
                totalMs = (System.nanoTime() - tCall0) / 1_000_000L,
            )
            return results.map { it ?: InsertVisualMatch("illustration", "open_set_rejected") }
        }

        val eligibleBoxes = eligibleIndices.map { insertBoxes[it] }
        val (repLocalIndices, localToRep, repThumbs) = deduplicateByThumbnail(
            rgbBytes, imgW, imgH, eligibleBoxes, circleMask = false, maeThreshold = 0.8f, shiftTolerant = true,
        )
        val repEmbsArr = arrayOfNulls<FloatArray>(repLocalIndices.size)
        val repExprSummaries = Array(repLocalIndices.size) { "" }
        val missSlots = ArrayList<Int>()
        var cacheHits = 0
        for (slot in repLocalIndices.indices) {
            val hit = lookupThumbCache(insertEmbCache, repThumbs[slot], 0.8f, shiftTolerant = true)
            if (hit != null) {
                repEmbsArr[slot] = hit.emb
                repExprSummaries[slot] = hit.exprSummary
                cacheHits++
            } else {
                missSlots.add(slot)
            }
        }
        val tInf0 = System.nanoTime()
        val failedInserts = if (missSlots.isNotEmpty()) {
            encodeAndCacheMisses(
                missSlots = missSlots,
                inputsNchw = missSlots.map { eligiblePrepared[repLocalIndices[it]].nchw },
                repThumbs = repThumbs,
                repEmbeddings = repEmbsArr,
                repExprSummaries = repExprSummaries,
                cache = insertEmbCache,
                maxCacheSize = MAX_THUMB_CACHE,
                isUnified = ccipHandle.isUnified,
                project = gallery::projectQuery,
                decoder = expressionDecoder,
                runEncoder = ccipHandle::runUnified,
            )
        } else {
            0
        }
        val inferMs = (System.nanoTime() - tInf0) / 1_000_000L
        var stage2Ns = 0L

        val skinMat = gallery.skinEmb
        val emojiMat = gallery.emojiEmb
        val repSkinSims = repEmbsArr.map { emb ->
            if (emb != null) matVecDot(skinMat.data, skinMat.rows, skinMat.cols, emb) else FloatArray(0)
        }
        val repEmojiSims = repEmbsArr.map { emb ->
            if (emb != null) matVecDot(emojiMat.data, emojiMat.rows, emojiMat.cols, emb) else FloatArray(0)
        }

        for (localI in eligibleIndices.indices) {
            val origI = eligibleIndices[localI]
            val repSlot = localToRep[localI]
            if (repEmbsArr[repSlot] == null) {
                results[origI] = InsertVisualMatch(
                    kind = "illustration",
                    status = "encode_failed",
                    expressionSummary = "",
                )
                continue
            }
            val box = insertBoxes[origI]

            // 1. Check Emoji match
            var bestEmojiScore = -1f
            var bestEmojiMeta: CatalogEmoji? = null
            val eSims = repEmojiSims[repSlot]
            if (eSims.isNotEmpty()) {
                var bestEj = 0
                for (j in 1 until eSims.size) if (eSims[j] > eSims[bestEj]) bestEj = j
                val esim = eSims[bestEj]
                if (esim >= EMOJI_STAGE1_THRESHOLD) {
                    val etpl = gallery.getEmojiTemplate(bestEj)
                    val ezncc = if (etpl != null) {
                        NativeCore.scoreEmojiStructure(rgbBytes, imgW, imgH, box, etpl, gallery.tplSize)
                    } else {
                        1.0f
                    }
                    if (ezncc >= EMOJI_STRUCT_THRESHOLD) {
                        bestEmojiScore = esim
                        bestEmojiMeta = gallery.catalog.emojis[bestEj]
                    }
                }
            }

            // 2. Check CharFace top-K skins + Stage-2 ZNCC
            data class CfCandidate(
                val skin: CatalogSkin,
                val expr: CatalogExpression,
                val s1Raw: Float,
                val combo: Float,
            )

            var bestCf: CfCandidate? = null
            var bestMoetalkCf: CfCandidate? = null
            val sSimsRaw = repSkinSims[repSlot]
            if (sSimsRaw.isNotEmpty()) {
                val ctxCanons = gallery.resolveSpeakerCanons(contextSpeakers.getOrNull(origI))
                val sSimsAdj = sSimsRaw.copyOf()
                if (ctxCanons.isNotEmpty()) {
                    for (si in gallery.catalog.skins.indices) {
                        if (gallery.catalog.skins[si].character in ctxCanons) {
                            sSimsAdj[si] += 0.03f
                        }
                    }
                }
                val topSkins = topKIndices(sSimsAdj, TOP_K_SKINS)
                if (topSkins.isNotEmpty() && sSimsRaw[topSkins[0]] >= CHARFACE_STAGE1_THRESHOLD) {
                    for (si in topSkins) {
                        val s1Raw = sSimsRaw[si]
                        if (s1Raw < CHARFACE_STAGE1_THRESHOLD - 0.06f) continue
                        val sk = gallery.catalog.skins[si]
                        val tSt2 = System.nanoTime()
                        val decoded = gallery.getSkinTemplates(sk.skinIndex)
                        if (decoded != null) {
                            val st2 = NativeCore.scoreStage2Skin(
                                rgbBytes, imgW, imgH, box,
                                decoded.stackRgb, decoded.count, decoded.tplSize, decoded.varWeights,
                            )
                            stage2Ns += System.nanoTime() - tSt2
                            val minStruct = when {
                                s1Raw >= HIGH_S1_RELAX_THRESHOLD -> CHARFACE_RELAX_STRUCT
                                s1Raw < LOW_S1_STRICT_THRESHOLD -> CHARFACE_STRICT_STRUCT
                                else -> CHARFACE_STRUCT_THRESHOLD
                            }
                            val minDiff = when {
                                s1Raw >= HIGH_S1_RELAX_THRESHOLD -> CHARFACE_RELAX_DIFF
                                s1Raw < LOW_S1_STRICT_THRESHOLD -> CHARFACE_STRICT_DIFF
                                else -> CHARFACE_DIFF_THRESHOLD
                            }
                            if (st2.structZncc < minStruct || st2.diffZncc < minDiff) {
                                continue
                            }
                            val expr = sk.expressions.getOrElse(st2.bestIndex) { sk.expressions.firstOrNull() ?: continue }
                            val combo = 0.35f * s1Raw + 0.65f * st2.diffZncc
                            val cand = CfCandidate(sk, expr, s1Raw, combo)
                            if (bestCf == null || combo > bestCf.combo) {
                                bestCf = cand
                            }
                            val hasMoetalkRef = expr.source == "moetalk:CharFace" ||
                                (expr.packV2?.preset != null && expr.packV2.ordinal > 0 &&
                                    gallery.packV2ToMoetalkExpr.containsKey(expr.packV2.preset to expr.packV2.ordinal))
                            if (hasMoetalkRef && (bestMoetalkCf == null || combo > bestMoetalkCf.combo)) {
                                bestMoetalkCf = cand
                            }
                        } else if (s1Raw >= CHARFACE_STAGE1_THRESHOLD + 0.06f) {
                            stage2Ns += System.nanoTime() - tSt2
                            // Graceful fallback if device lacks AV1 High-Profile MediaCodec
                            val expr = sk.expressions.firstOrNull() ?: continue
                            val cand = CfCandidate(sk, expr, s1Raw, s1Raw)
                            if (bestCf == null || s1Raw > bestCf.combo) {
                                bestCf = cand
                            }
                            if (expr.source == "moetalk:CharFace" && (bestMoetalkCf == null || s1Raw > bestMoetalkCf.combo)) {
                                bestMoetalkCf = cand
                            }
                        }
                    }
                }
            }
            if (bestMoetalkCf != null && (bestCf == null || bestMoetalkCf.combo >= bestCf.combo - 0.12f)) {
                bestCf = bestMoetalkCf
            }

            val exprSum = repExprSummaries[repSlot]
            results[origI] = when {
                bestEmojiMeta != null && (bestCf == null || bestEmojiScore > bestCf.s1Raw + 0.04f) -> {
                    InsertVisualMatch(
                        kind = "emoji",
                        status = "matched",
                        expressionSummary = exprSum,
                        emojiId = bestEmojiMeta.emojiId,
                        source = bestEmojiMeta.source,
                        confidence = bestEmojiScore,
                    )
                }
                bestCf != null -> {
                    InsertVisualMatch(
                        kind = "charface",
                        status = "matched",
                        character = bestCf.skin.character,
                        expressionId = bestCf.expr.expressionId,
                        expressionSummary = exprSum,
                        source = bestCf.expr.source,
                        packV2Preset = bestCf.expr.packV2?.preset,
                        packV2Ordinal = bestCf.expr.packV2?.ordinal ?: 0,
                        confidence = bestCf.s1Raw,
                    )
                }
                else -> InsertVisualMatch(
                    kind = "illustration",
                    status = "open_set_rejected",
                    expressionSummary = exprSum,
                )
            }
        }

        lastInsertStats = CcipCallStats(
            inputBoxes = insertBoxes.size,
            uniqueCrops = missSlots.size,
            cacheHits = cacheHits,
            failedCrops = failedInserts,
            inferMs = inferMs,
            stage2Ms = stage2Ns / 1_000_000L,
            totalMs = (System.nanoTime() - tCall0) / 1_000_000L,
        )
        return results.map { it ?: InsertVisualMatch("illustration", "open_set_rejected") }
    }

    private fun lookupThumbCache(
        cache: List<CachedThumbEmb>,
        thumb: FloatArray,
        threshold: Float,
        shiftTolerant: Boolean = false,
    ): CachedThumbEmb? {
        for (entry in cache) {
            val d = if (shiftTolerant) shiftTolerantMae16(thumb, entry.thumb) else meanAbsError(thumb, entry.thumb)
            if (d <= threshold) return entry
        }
        return null
    }

    private fun deduplicateByThumbnail(
        rgbBytes: ByteArray,
        imgW: Int,
        imgH: Int,
        boxes: List<WindowBox>,
        circleMask: Boolean,
        maeThreshold: Float,
        shiftTolerant: Boolean = false,
    ): Triple<List<Int>, IntArray, List<FloatArray>> {
        val repIndices = ArrayList<Int>()
        val repThumbs = ArrayList<FloatArray>()
        val itemToRep = IntArray(boxes.size)
        for (i in boxes.indices) {
            val thumb = NativeCore.computeRgbThumbnail(rgbBytes, imgW, imgH, boxes[i], circleMask, 16)
            var matched = -1
            for (slot in repThumbs.indices) {
                val d = if (shiftTolerant) shiftTolerantMae16(thumb, repThumbs[slot]) else meanAbsError(thumb, repThumbs[slot])
                if (d <= maeThreshold) {
                    matched = slot
                    break
                }
            }
            if (matched < 0) {
                matched = repIndices.size
                repIndices.add(i)
                repThumbs.add(thumb)
            }
            itemToRep[i] = matched
        }
        return Triple(repIndices, itemToRep, repThumbs)
    }

    private fun candidateCharacter(index: Int): String =
        if (index < nChars) gallery.catalog.chars[index].character else gallery.catalog.skins[index - nChars].character

    private fun assignCustomCluster(vec: FloatArray): String {
        for (k in customProtos.indices) {
            if (dot(vec, customProtos[k]) >= CUSTOM_CLUSTER_THRESHOLD) {
                return "custom_char_%02d".format(k + 1)
            }
        }
        customProtos.add(vec.copyOf())
        return "custom_char_%02d".format(customProtos.size)
    }

    override fun close() {
        runCatching { ccipHandle.close() }
        runCatching { gallery.close() }
    }

    companion object {
        private const val MAX_THUMB_CACHE = 384
        private const val AVATAR_THRESHOLD = 0.78f
        private const val NAME_PRIOR_BONUS = 0.10f
        private const val CHARFACE_STAGE1_THRESHOLD = 0.76f
        private const val CHARFACE_STRUCT_THRESHOLD = 0.58f
        private const val CHARFACE_DIFF_THRESHOLD = 0.45f
        private const val LOW_S1_STRICT_THRESHOLD = 0.80f
        private const val CHARFACE_STRICT_STRUCT = 0.72f
        private const val CHARFACE_STRICT_DIFF = 0.65f
        private const val HIGH_S1_RELAX_THRESHOLD = 0.82f
        private const val CHARFACE_RELAX_STRUCT = 0.52f
        private const val CHARFACE_RELAX_DIFF = 0.38f
        private const val EMOJI_STAGE1_THRESHOLD = 0.84f
        private const val EMOJI_STRUCT_THRESHOLD = 0.70f
        private const val CUSTOM_CLUSTER_THRESHOLD = 0.86f
        private const val TOP_K_SKINS = 5

        fun isCfid6Available(bundleRoot: File): Boolean {
            val unifiedMnn = dev.momotranscript.app.model.ModelPackCatalog.resolveFile(
                bundleRoot,
                dev.momotranscript.app.model.ModelPackCatalog.UNIFIED_AVATAR_MODEL,
                dev.momotranscript.app.model.ModelPackCatalog.UNIFIED_AVATAR_LEGACY,
            )
            val npzPe = File(bundleRoot, dev.momotranscript.app.model.ModelPackCatalog.STAGE1_PE256)
            val cf6Cat = File(bundleRoot, dev.momotranscript.app.model.ModelPackCatalog.CFID6_CATALOG)
            val cf6Emb = File(bundleRoot, dev.momotranscript.app.model.ModelPackCatalog.CFID6_STAGE1_PE256)
            val cf6Pack = File(bundleRoot, dev.momotranscript.app.model.ModelPackCatalog.CFID6_STAGE2_PACK)
            return unifiedMnn.isFile && npzPe.isFile && cf6Cat.isFile && cf6Emb.isFile && cf6Pack.isFile
        }

        fun isAvailable(bundleRoot: File): Boolean {
            val unifiedMnn = dev.momotranscript.app.model.ModelPackCatalog.resolveFile(
                bundleRoot,
                dev.momotranscript.app.model.ModelPackCatalog.UNIFIED_AVATAR_MODEL,
                dev.momotranscript.app.model.ModelPackCatalog.UNIFIED_AVATAR_LEGACY,
            )
            val npzPe = File(bundleRoot, dev.momotranscript.app.model.ModelPackCatalog.STAGE1_PE256)
            val legacyMnn = dev.momotranscript.app.model.ModelPackCatalog.resolveFile(
                bundleRoot,
                dev.momotranscript.app.model.ModelPackCatalog.CCIP_MODEL,
                dev.momotranscript.app.model.ModelPackCatalog.CCIP_LEGACY,
            )
            val npzSvd = File(bundleRoot, "avatar/stage1_embeddings_svd128.npz")
            val npzLegacy = File(bundleRoot, "avatar/stage1_embeddings.npz")
            val catZst = File(bundleRoot, "avatar/catalog.json.zst")
            val catJson = File(bundleRoot, "avatar/catalog.json")
            val hasCatalog = catZst.isFile || catJson.isFile
            val hasUnified = unifiedMnn.isFile && npzPe.isFile
            val hasLegacy = legacyMnn.isFile && (npzSvd.isFile || npzLegacy.isFile)
            return hasCatalog && (hasUnified || hasLegacy)
        }

        internal fun isAllFinite(v: FloatArray?): Boolean {
            if (v == null || v.isEmpty()) return false
            for (x in v) if (!x.isFinite()) return false
            return true
        }

        internal fun isFiniteNonZero(v: FloatArray?): Boolean {
            if (!isAllFinite(v)) return false
            var sum = 0.0
            for (x in v!!) sum += x.toDouble() * x.toDouble()
            return sum.isFinite() && sum > 1e-16
        }

        /**
         * Encodes cache-miss slots with strict non-finite isolation: if either `idEmb` or `exprLogits`
         * contains `NaN`/`±Inf` (or `idEmb` has zero norm), that slot is marked failed (`null` embedding,
         * empty expression), expression decoding is skipped, and nothing is written to [cache].
         * Healthy slots in the same batch proceed normally.
         */
        internal fun encodeAndCacheMisses(
            missSlots: List<Int>,
            inputsNchw: List<FloatArray>,
            repThumbs: List<FloatArray>,
            repEmbeddings: Array<FloatArray?>,
            repExprSummaries: Array<String>,
            cache: MutableList<CachedThumbEmb>,
            maxCacheSize: Int,
            isUnified: Boolean,
            project: (FloatArray) -> FloatArray,
            decoder: ExpressionDecoder?,
            runEncoder: (FloatArray) -> NativeCore.UnifiedEncodeOutput,
        ): Int {
            var failedCount = 0
            for (k in missSlots.indices) {
                val slot = missSlots[k]
                val res = runEncoder(inputsNchw[k])
                val logitsOk = res.exprLogits == null || isAllFinite(res.exprLogits)
                if (!isFiniteNonZero(res.idEmb) || !logitsOk) {
                    repEmbeddings[slot] = null
                    repExprSummaries[slot] = ""
                    failedCount++
                    continue
                }
                val norm = l2Normalize(res.idEmb)
                val projected = if (isUnified) norm else project(norm)
                if (!isFiniteNonZero(projected)) {
                    repEmbeddings[slot] = null
                    repExprSummaries[slot] = ""
                    failedCount++
                    continue
                }
                val exprSum = decoder?.decodeCaption(res.exprLogits) ?: ""
                repEmbeddings[slot] = projected
                repExprSummaries[slot] = exprSum
                if (cache.size < maxCacheSize) {
                    cache.add(CachedThumbEmb(repThumbs[slot], projected, exprSum))
                }
            }
            return failedCount
        }

        private fun l2Normalize(v: FloatArray): FloatArray {
            var sum = 0.0
            for (x in v) sum += x.toDouble() * x.toDouble()
            val inv = (1.0 / (sqrt(sum) + 1e-12)).toFloat()
            return FloatArray(v.size) { v[it] * inv }
        }

        private fun normalizeDoubleToFloat(v: DoubleArray): FloatArray {
            var sum = 0.0
            for (x in v) sum += x * x
            val inv = 1.0 / (sqrt(sum) + 1e-12)
            return FloatArray(v.size) { (v[it] * inv).toFloat() }
        }

        private fun dot(a: FloatArray, b: FloatArray): Float {
            var s = 0f
            val n = minOf(a.size, b.size)
            for (i in 0 until n) s += a[i] * b[i]
            return s
        }

        private fun matVecDot(mat: FloatArray, rows: Int, cols: Int, vec: FloatArray): FloatArray {
            val out = FloatArray(rows)
            for (r in 0 until rows) {
                val base = r * cols
                var s = 0f
                for (c in 0 until cols) s += mat[base + c] * vec[c]
                out[r] = s
            }
            return out
        }

        /**
         * Shift-tolerant MAE on 16x16x3 RGB thumbnails: evaluates all 9 integer shifts
         * (dx, dy) in {-1, 0, +1}^2 on the central 14x14 crop so 1-2px detector box jitter
         * still deduplicates repeated avatars cleanly (~0.005 ms per pair).
         */
        private fun shiftTolerantMae16(a: FloatArray, b: FloatArray): Float {
            if (a.size != 16 * 16 * 3 || b.size != 16 * 16 * 3) return meanAbsError(a, b)
            var best = Float.MAX_VALUE
            val denom = 14f * 14f * 3f
            for (dy in -1..1) {
                for (dx in -1..1) {
                    var sum = 0f
                    for (y in 1..14) {
                        val aRow = y * 16
                        val bRow = (y + dy) * 16
                        for (x in 1..14) {
                            val ai = (aRow + x) * 3
                            val bi = (bRow + x + dx) * 3
                            sum += abs(a[ai] - b[bi]) + abs(a[ai + 1] - b[bi + 1]) + abs(a[ai + 2] - b[bi + 2])
                        }
                    }
                    val mae = sum / denom
                    if (mae < best) best = mae
                }
            }
            return best
        }

        private fun meanAbsError(a: FloatArray, b: FloatArray): Float {
            if (a.isEmpty()) return 0f
            var s = 0f
            val n = minOf(a.size, b.size)
            for (i in 0 until n) s += abs(a[i] - b[i])
            return s / n.toFloat()
        }

        private fun topKIndices(arr: FloatArray, k: Int): List<Int> {
            val kk = minOf(k, arr.size)
            return arr.indices.sortedByDescending { arr[it] }.take(kk)
        }
    }
}
