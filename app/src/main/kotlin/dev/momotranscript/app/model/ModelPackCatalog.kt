package dev.momotranscript.app.model

import java.io.File

data class AssetFileSpec(
    val relativePath: String,
    val approxSizeMiB: Double,
    val legacyFallbackPaths: List<String> = emptyList(),
) {
    fun resolveExisting(bundleRoot: File): File? {
        val primary = File(bundleRoot, relativePath)
        if (primary.isFile && primary.length() > 0L) return primary
        for (legacy in legacyFallbackPaths) {
            val f = File(bundleRoot, legacy)
            if (f.isFile && f.length() > 0L) return f
        }
        return null
    }
}

data class ModelPackSpec(
    val id: String,
    val title: String,
    val subtitle: String,
    val required: Boolean,
    val files: List<AssetFileSpec>,
) {
    val approxTotalMiB: Double
        get() = files.sumOf { it.approxSizeMiB }
}

data class ModelPackStatus(
    val spec: ModelPackSpec,
    val installedFiles: Int,
    val totalFiles: Int,
    val installedBytes: Long = 0L,
) {
    val isReady: Boolean get() = installedFiles == totalFiles
}

/**
 * Modular offline model & AVIFS asset pack definitions matching ``mmt_mobile_bundle_v1``.
 */
object ModelPackCatalog {
    const val DET_MODEL = "mnn/det_boxes_order_ln_w8.mnn"
    const val REC_SMALL_MODEL = "mnn/rec_small_w8.mnn"
    const val REC_SMALL_LEGACY = "mnn/rec_small_fp16.mnn"
    const val REC_MEDIUM_MODEL = "mnn/rec_medium_w8.mnn"
    const val REC_MEDIUM_LEGACY = "mnn/rec_medium_fp16.mnn"
    const val CCIP_MODEL = "mnn/ccip_caformer_24_ln_w4.mnn"
    const val CCIP_LEGACY = "mnn/ccip_caformer_24_ln_w8.mnn"
    const val STAGE2_PACK = "avatar/stage2_templates_q80.avifs.pack"
    const val STAGE2_LEGACY = "avatar/stage2_templates.avifs.pack"

    fun resolveFile(bundleRoot: File, primary: String, vararg legacyFallbacks: String): File {
        val p = File(bundleRoot, primary)
        if (p.isFile && p.length() > 0L) return p
        for (leg in legacyFallbacks) {
            val f = File(bundleRoot, leg)
            if (f.isFile && f.length() > 0L) return f
        }
        return p
    }

    val packs: List<ModelPackSpec> = listOf(
        ModelPackSpec(
            id = "core_text",
            title = "基础转录包（必选 · MNN）",
            subtitle = "PP-DocLayoutV3 Fused-LN W8 + PP-OCRv6 Small W8",
            required = true,
            files = listOf(
                AssetFileSpec(DET_MODEL, 32.75),
                AssetFileSpec(REC_SMALL_MODEL, 5.31, listOf(REC_SMALL_LEGACY)),
                AssetFileSpec("models/ocr_vocab.yml", 0.14),
            ),
        ),
        ModelPackSpec(
            id = "rec_medium",
            title = "高精度级联包（可选 · MNN）",
            subtitle = "PP-OCRv6 Medium W8，仅对弱字/低分行自动复核",
            required = false,
            files = listOf(
                AssetFileSpec(REC_MEDIUM_MODEL, 18.64, listOf(REC_MEDIUM_LEGACY)),
            ),
        ),
        ModelPackSpec(
            id = "moetalk_avatar",
            title = "视觉匹配扩展包（可选 · 外部源）",
            subtitle = "CCIP Fused-LN W4-HQQ + SVD-128 索引 + MMTAVIF2 q80 模板",
            required = false,
            files = listOf(
                AssetFileSpec(CCIP_MODEL, 21.15, listOf(CCIP_LEGACY)),
                AssetFileSpec("avatar/catalog.json.zst", 0.61),
                AssetFileSpec("avatar/stage1_embeddings_svd128.npz", 1.43),
                AssetFileSpec(STAGE2_PACK, 9.15, listOf(STAGE2_LEGACY)),
            ),
        ),
    )

    fun inspectInstalled(bundleRoot: File): List<ModelPackStatus> =
        packs.map { spec ->
            var present = 0
            var bytes = 0L
            for (fileSpec in spec.files) {
                val f = fileSpec.resolveExisting(bundleRoot)
                if (f != null) {
                    present++
                    bytes += f.length()
                }
            }
            ModelPackStatus(
                spec = spec,
                installedFiles = present,
                totalFiles = spec.files.size,
                installedBytes = bytes,
            )
        }
}
