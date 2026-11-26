package dev.momotranscript.app.avatar

import com.github.luben.zstd.Zstd
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.ByteArrayOutputStream
import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets
import java.util.zip.Inflater

@Serializable
data class CatalogChar(
    val source: String = "moetalk:Char",
    @SerialName("asset_id") val assetId: String = "",
    val skin: String = "",
    val variant: String = "base",
    val character: String = "",
    val preset: String? = null,
)

@Serializable
data class CatalogPackV2(
    val preset: String? = null,
    val ordinal: Int = 0,
)

@Serializable
data class CatalogExpression(
    @SerialName("expression_id") val expressionId: String = "",
    val stem: String = "",
    val source: String = "moetalk:CharFace",
    @SerialName("pack_v2") val packV2: CatalogPackV2? = null,
)

@Serializable
data class CatalogSkin(
    @SerialName("skin_id") val skinId: String = "",
    val character: String = "",
    val preset: String? = null,
    val source: String = "moetalk:CharFace",
    @SerialName("prototype_expression_id") val prototypeExpressionId: String = "",
    val expressions: List<CatalogExpression> = emptyList(),
    @SerialName("skin_index") val skinIndex: Int = 0,
)

@Serializable
data class CatalogEmoji(
    @SerialName("emoji_id") val emojiId: String = "",
    val source: String = "moetalk:Emoji",
)

@Serializable
data class CatalogOptions(
    @SerialName("tpl_size") val tplSize: Int = 64,
)

@Serializable
data class AvatarCatalog(
    val kind: String = "",
    val options: CatalogOptions = CatalogOptions(),
    val aliases: Map<String, List<String>> = emptyMap(),
    val chars: List<CatalogChar> = emptyList(),
    val skins: List<CatalogSkin> = emptyList(),
    val emojis: List<CatalogEmoji> = emptyList(),
)

@Serializable
data class AvifsPackSkinEntry(
    val index: Int = 0,
    val count: Int = 0,
    @SerialName("frame_order") val frameOrder: List<Int> = emptyList(),
    @SerialName("avifs_offset") val avifsOffset: Long = 0L,
    @SerialName("avifs_len") val avifsLen: Int = 0,
    @SerialName("mask_offset") val maskOffset: Long = 0L,
    @SerialName("mask_len") val maskLen: Int = 0,
)

@Serializable
data class AvifsPackEmojiChunk(
    val start: Int = 0,
    val count: Int = 0,
    val offset: Long = 0L,
    val len: Int = 0,
)

@Serializable
data class AvifsPackHeader(
    @SerialName("schema_version") val schemaVersion: Int = 1,
    @SerialName("tpl_size") val tplSize: Int = 64,
    @SerialName("mask_dtype") val maskDtype: String = "float16",
    val skins: List<AvifsPackSkinEntry> = emptyList(),
    @SerialName("emoji_chunks") val emojiChunks: List<AvifsPackEmojiChunk> = emptyList(),
)

data class DecodedSkinTemplates(
    val stackRgb: ByteArray,   // count * tplSize * tplSize * 3 in original catalog order
    val varWeights: FloatArray, // tplSize * tplSize
    val count: Int,
    val tplSize: Int,
)

/**
 * Runtime loader for the mobile avatar bundle:
 * - `avatar/catalog.json.zst` (or uncompressed `catalog.json`)
 * - `avatar/stage1_embeddings.npz` (FP16 -> FP32 Stage-1 embeddings)
 * - `avatar/stage2_templates.avifs.pack` (`MMTAVIF1` O(1) seek container for Stage-2 templates)
 */
class AvatarGallery(avatarDir: File, preferPe256: Boolean = false) : Closeable {

    val catalog: AvatarCatalog
    val projMatrix: FloatMatrix? // (768, K) when SVD-compressed Stage-1 embeddings are loaded
    val charEmb: FloatMatrix
    val charEmbLowres: FloatMatrix?
    val skinEmb: FloatMatrix
    val emojiEmb: FloatMatrix
    val tplSize: Int

    val canonToDefaultHead: Map<String, String>
    val officialCanons: Set<String>
    val packV2ToMoetalkExpr: Map<Pair<String, Int>, String>

    private val packRaf: RandomAccessFile?
    private val packPayloadBase: Long
    private val packMaskIsUint8: Boolean
    private val skinPackByIndex: Map<Int, AvifsPackSkinEntry>
    private val emojiChunks: List<AvifsPackEmojiChunk>
    private val skinCache = LinkedHashMap<Int, DecodedSkinTemplates?>(32, 0.75f, true)
    private val emojiChunkCache = LinkedHashMap<Int, ByteArray?>(8, 0.75f, true)

    init {
        val json = Json { ignoreUnknownKeys = true }
        val zstFile = File(avatarDir, "catalog.json.zst")
        val rawJsonFile = File(avatarDir, "catalog.json")
        val catalogText = when {
            zstFile.isFile -> decompressZst(zstFile.readBytes()).toString(StandardCharsets.UTF_8)
            rawJsonFile.isFile -> rawJsonFile.readText(StandardCharsets.UTF_8)
            else -> error("缺少角色目录：${zstFile.absolutePath}")
        }
        catalog = json.decodeFromString(catalogText)
        require(catalog.kind == "mmt_avatar_gallery_v1") {
            "不支持的视觉索引格式：${catalog.kind}"
        }
        tplSize = catalog.options.tplSize

        val peNpzFile = File(avatarDir, "stage1_embeddings_pe256.npz")
        val svdNpzFile = File(avatarDir, "stage1_embeddings_svd128.npz")
        val legacyNpzFile = File(avatarDir, "stage1_embeddings.npz")
        val npzFile = when {
            preferPe256 && peNpzFile.isFile -> peNpzFile
            svdNpzFile.isFile -> svdNpzFile
            legacyNpzFile.isFile -> legacyNpzFile
            peNpzFile.isFile -> peNpzFile
            else -> error("缺少 Stage-1 向量文件：${peNpzFile.absolutePath}")
        }
        val matrices = NpyReader.loadNpzMatrices(npzFile)
        projMatrix = matrices["proj"]
        val defaultCols = projMatrix?.cols ?: matrices["skin_emb"]?.cols ?: matrices["char_emb"]?.cols ?: 768
        charEmb = matrices["char_emb"] ?: FloatMatrix(0, defaultCols, FloatArray(0))
        charEmbLowres = matrices["char_emb_lowres"]
        skinEmb = matrices["skin_emb"] ?: FloatMatrix(0, defaultCols, FloatArray(0))
        emojiEmb = matrices["emoji_emb"] ?: FloatMatrix(0, defaultCols, FloatArray(0))

        // Build MoeTalk official asset lookups
        val headMap = HashMap<String, String>()
        val prioMap = HashMap<String, Int>()
        val official = HashSet<String>()
        val prioRank = mapOf("base" to 0, "bg" to 1, "l2d" to 2)
        for (ch in catalog.chars) {
            if (ch.source != "moetalk:Char") continue
            official.add(ch.character)
            val head = if (ch.assetId.startsWith("moetalk:Char/")) {
                ch.assetId.removePrefix("moetalk:Char/")
            } else {
                ch.skin
            }
            if (head.isEmpty()) continue
            val p = prioRank[ch.variant] ?: 3
            if (ch.character !in headMap || p < (prioMap[ch.character] ?: 99)) {
                headMap[ch.character] = head
                prioMap[ch.character] = p
            }
        }
        val p2Map = HashMap<Pair<String, Int>, String>()
        for (sk in catalog.skins) {
            if (sk.source != "moetalk:CharFace") continue
            official.add(sk.character)
            for (expr in sk.expressions) {
                val p2 = expr.packV2
                if (p2 != null && !p2.preset.isNullOrEmpty() && p2.ordinal > 0) {
                    p2Map.putIfAbsent(p2.preset to p2.ordinal, expr.expressionId)
                }
            }
        }
        canonToDefaultHead = headMap
        officialCanons = official
        packV2ToMoetalkExpr = p2Map

        // Open optional MMTAVIF1 Stage-2 container (prefer slim q80 pack, fall back to legacy q92)
        val packQ80 = File(avatarDir, "stage2_templates_q80.avifs.pack")
        val packFile = if (packQ80.isFile && packQ80.length() > 0L) {
            packQ80
        } else {
            File(avatarDir, "stage2_templates.avifs.pack")
        }
        if (packFile.isFile) {
            val raf = RandomAccessFile(packFile, "r")
            val magic = ByteArray(8)
            raf.readFully(magic)
            if (String(magic, StandardCharsets.US_ASCII) == "MMTAVIF1") {
                val lenBytes = ByteArray(4)
                raf.readFully(lenBytes)
                val hdrLen = ByteBuffer.wrap(lenBytes).order(ByteOrder.LITTLE_ENDIAN).int
                val hdrBytes = ByteArray(hdrLen)
                raf.readFully(hdrBytes)
                val hdr = json.decodeFromString<AvifsPackHeader>(String(hdrBytes, StandardCharsets.UTF_8))
                packRaf = raf
                packPayloadBase = 12L + hdrLen.toLong()
                packMaskIsUint8 = hdr.maskDtype == "uint8" || hdr.schemaVersion >= 2
                skinPackByIndex = hdr.skins.associateBy { it.index }
                emojiChunks = hdr.emojiChunks
            } else {
                raf.close()
                packRaf = null
                packPayloadBase = 0L
                packMaskIsUint8 = true
                skinPackByIndex = emptyMap()
                emojiChunks = emptyList()
            }
        } else {
            packRaf = null
            packPayloadBase = 0L
            packMaskIsUint8 = true
            skinPackByIndex = emptyMap()
            emojiChunks = emptyList()
        }
    }

    /**
     * Projects a normalized 768-d CCIP embedding into the gallery's SVD-K subspace when SVD
     * compression is active, or returns `vec` unchanged for legacy 768-d galleries.
     */
    fun projectQuery(vec: FloatArray): FloatArray {
        val pm = projMatrix ?: return vec
        if (vec.size != pm.rows) return vec
        val k = pm.cols
        val out = FloatArray(k)
        val pData = pm.data
        for (r in 0 until pm.rows) {
            val vr = vec[r]
            if (vr == 0f) continue
            val rowBase = r * k
            for (c in 0 until k) {
                out[c] += vr * pData[rowBase + c]
            }
        }
        return out
    }

    fun resolveSpeakerCanons(speakerName: String?): Set<String> {
        val raw = speakerName?.trim().orEmpty()
        if (raw.isEmpty()) return emptySet()
        catalog.aliases[raw.lowercase()]?.takeIf { it.isNotEmpty() }?.let { return it.toSet() }
        val base = raw.substringBefore('(').substringBefore('（').trim()
        if (base.isNotEmpty()) {
            catalog.aliases[base.lowercase()]?.takeIf { it.isNotEmpty() }?.let { return it.toSet() }
        }
        return emptySet()
    }

    fun getSkinTemplates(skinIndex: Int): DecodedSkinTemplates? {
        if (skinCache.containsKey(skinIndex)) return skinCache[skinIndex]
        val raf = packRaf ?: return null
        val entry = skinPackByIndex[skinIndex] ?: return null
        if (entry.count <= 0 || entry.avifsLen <= 0) return null

        val avifsBytes = ByteArray(entry.avifsLen)
        val maskCompressed = ByteArray(entry.maskLen)
        synchronized(raf) {
            raf.seek(packPayloadBase + entry.avifsOffset)
            raf.readFully(avifsBytes)
            raf.seek(packPayloadBase + entry.maskOffset)
            raf.readFully(maskCompressed)
        }

        val decodedOrdered = AvifsFrameDecoder.decodeSequenceRgb(avifsBytes, entry.count, tplSize)
            ?: run {
                skinCache[skinIndex] = null
                return null
            }

        // Un-permute medoid-reordered frames back to original catalog expression order.
        val frameSize = tplSize * tplSize * 3
        val restored = ByteArray(entry.count * frameSize)
        for (seqPos in 0 until entry.count) {
            val origIdx = entry.frameOrder.getOrElse(seqPos) { seqPos }
            if (origIdx in 0 until entry.count) {
                System.arraycopy(decodedOrdered, seqPos * frameSize, restored, origIdx * frameSize, frameSize)
            }
        }

        // Inflate zlib-compressed variance mask (v2 uint8 [0..255] or legacy v1 float16)
        val maskBytes = inflateZlib(maskCompressed)
        val varWeights = FloatArray(tplSize * tplSize)
        if (maskBytes != null && (packMaskIsUint8 || maskBytes.size == varWeights.size) && maskBytes.size >= varWeights.size) {
            for (i in varWeights.indices) {
                varWeights[i] = (maskBytes[i].toInt() and 0xFF) / 255.0f
            }
        } else if (maskBytes != null && maskBytes.size >= varWeights.size * 2) {
            for (i in varWeights.indices) {
                val lo = maskBytes[i * 2].toInt() and 0xFF
                val hi = maskBytes[i * 2 + 1].toInt() and 0xFF
                val w = NpyReader.halfToFloat(lo or (hi shl 8))
                varWeights[i] = if (w.isFinite() && w >= 0f) w else 0f
            }
        } else {
            varWeights.fill(1.0f)
        }

        val tpl = DecodedSkinTemplates(restored, varWeights, entry.count, tplSize)
        if (skinCache.size >= 32) {
            val eldest = skinCache.keys.firstOrNull()
            if (eldest != null) skinCache.remove(eldest)
        }
        skinCache[skinIndex] = tpl
        return tpl
    }

    fun getEmojiTemplate(emojiIndex: Int): ByteArray? {
        val raf = packRaf ?: return null
        val chunkIdx = emojiChunks.indexOfFirst { emojiIndex in it.start until (it.start + it.count) }
        if (chunkIdx < 0) return null
        val chunk = emojiChunks[chunkIdx]
        val chunkRgb = emojiChunkCache.getOrPut(chunkIdx) {
            val buf = ByteArray(chunk.len)
            synchronized(raf) {
                raf.seek(packPayloadBase + chunk.offset)
                raf.readFully(buf)
            }
            AvifsFrameDecoder.decodeSequenceRgb(buf, chunk.count, tplSize)
        } ?: return null

        val local = emojiIndex - chunk.start
        val frameSize = tplSize * tplSize * 3
        val start = local * frameSize
        if (start + frameSize > chunkRgb.size) return null
        return chunkRgb.copyOfRange(start, start + frameSize)
    }

    override fun close() {
        runCatching { packRaf?.close() }
    }

    companion object {
        private fun decompressZst(compressed: ByteArray): ByteArray {
            val exactSize = Zstd.getFrameContentSize(compressed)
            if (exactSize > 0L && exactSize <= 64L * 1024L * 1024L) {
                return Zstd.decompress(compressed, exactSize.toInt())
            }
            // Streaming fallback if frame content size is absent
            val dst = ByteArray(16 * 1024 * 1024)
            val n = Zstd.decompress(dst, compressed).toInt()
            return dst.copyOf(n.coerceAtLeast(0))
        }

        private fun inflateZlib(compressed: ByteArray): ByteArray? {
            val inflater = Inflater()
            return try {
                inflater.setInput(compressed)
                val out = ByteArrayOutputStream(compressed.size * 3)
                val buf = ByteArray(4096)
                while (!inflater.finished()) {
                    val n = inflater.inflate(buf)
                    if (n > 0) out.write(buf, 0, n)
                    else if (inflater.needsInput() || inflater.needsDictionary()) break
                }
                out.toByteArray()
            } catch (_: Exception) {
                null
            } finally {
                inflater.end()
            }
        }
    }
}
