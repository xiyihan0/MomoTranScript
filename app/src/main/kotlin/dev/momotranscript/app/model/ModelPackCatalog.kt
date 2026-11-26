package dev.momotranscript.app.model

import java.io.File

data class AssetFileSpec(
    val relativePath: String,
    val approxSizeMiB: Double,
)

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
    val packs: List<ModelPackSpec> = listOf(
        ModelPackSpec(
            id = "core_text",
            title = "基础转录包（必选 · MNN）",
            subtitle = "PP-DocLayoutV3 Fused-LN W8 + PP-OCRv6 Small FP16（防 FP16 溢出）",
            required = true,
            files = listOf(
                AssetFileSpec("mnn/det_boxes_order_ln_w8.mnn", 32.75),
                AssetFileSpec("mnn/rec_small_fp16.mnn", 10.22),
                AssetFileSpec("models/ocr_vocab.yml", 0.14),
            ),
        ),
        ModelPackSpec(
            id = "rec_medium",
            title = "高精度级联包（可选 · MNN）",
            subtitle = "PP-OCRv6 Medium FP16，仅对弱字/低分行自动复核",
            required = false,
            files = listOf(
                AssetFileSpec("mnn/rec_medium_fp16.mnn", 36.56),
            ),
        ),
        ModelPackSpec(
            id = "moetalk_avatar",
            title = "视觉匹配扩展包（可选 · 外部源）",
            subtitle = "CCIP Fused-LN W8 GPU + SVD-128 向量索引 + MMTAVIF2 模板容器",
            required = false,
            files = listOf(
                AssetFileSpec("mnn/ccip_caformer_24_ln_w8.mnn", 37.19),
                AssetFileSpec("avatar/catalog.json.zst", 0.61),
                AssetFileSpec("avatar/stage1_embeddings_svd128.npz", 1.43),
                AssetFileSpec("avatar/stage2_templates.avifs.pack", 13.14),
            ),
        ),
    )

    fun inspectInstalled(bundleRoot: File): List<ModelPackStatus> =
        packs.map { spec ->
            var present = 0
            var bytes = 0L
            for (fileSpec in spec.files) {
                val f = File(bundleRoot, fileSpec.relativePath)
                if (f.isFile && f.length() > 0L) {
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
