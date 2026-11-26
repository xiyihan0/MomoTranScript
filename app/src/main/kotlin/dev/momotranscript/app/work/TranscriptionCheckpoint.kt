package dev.momotranscript.app.work

import dev.momotranscript.app.ui.PagePreviewRecord
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import java.io.File

@Serializable
data class CheckpointBranchInput(
    val label: String,
    val messages: List<JsonObject>,
)

@Serializable
data class TranscriptionCheckpoint(
    val jobId: String,
    val title: String,
    val completedItems: Int,
    val totalItems: Int,
    val totalWindows: Int,
    val totalLines: Int,
    val matchedOfficialAvatars: Int,
    val matchedVisualAssets: Int,
    val elapsedMs: Long,
    val customCounter: Int,
    val imageCounter: Int,
    val customKeyToId: Map<String, String> = emptyMap(),
    val tempChars: Map<String, JsonObject> = emptyMap(),
    val tempImageIds: List<String> = emptyList(),
    val branchInputs: List<CheckpointBranchInput> = emptyList(),
    val combinedMomoScript: String = "",
    val previewPages: List<PagePreviewRecord> = emptyList(),
    val updatedAtMs: Long = System.currentTimeMillis(),
)

data class ResumableJobInfo(
    val jobDir: File,
    val specFile: File,
    val spec: TranscriptionJobSpec,
    val checkpoint: TranscriptionCheckpoint,
    val wasPausedByUser: Boolean,
)

/**
 * Atomic per-page checkpoint store under `filesDir/jobs/<jobId>/`.
 *
 * Base64 WebP crops are stored one-file-per-image under `temp_images/<id>.txt` on creation so
 * per-page `checkpoint.json` flushes remain tiny (<200 KiB, <5 ms) even on 200-page PDFs.
 */
object CheckpointStore {
    private val json = Json {
        ignoreUnknownKeys = true
        encodeDefaults = true
    }

    private const val CHECKPOINT_FILE = "checkpoint.json"
    private const val SPEC_FILE = "spec.json"
    private const val PAUSED_FLAG = "paused.flag"
    private const val COMPLETED_FLAG = "completed.flag"
    private const val TEMP_IMAGES_DIR = "temp_images"

    fun saveTempImage(jobDir: File, id: String, dataUrl: String) {
        runCatching {
            val dir = File(jobDir, TEMP_IMAGES_DIR).apply { mkdirs() }
            val safe = id.replace(Regex("[^A-Za-z0-9_.-]"), "_")
            val dst = File(dir, "$safe.txt")
            if (!dst.isFile) {
                dst.writeText(dataUrl, Charsets.UTF_8)
            }
        }
    }

    fun loadTempImages(jobDir: File, ids: List<String>): LinkedHashMap<String, String> {
        val out = LinkedHashMap<String, String>(ids.size)
        val dir = File(jobDir, TEMP_IMAGES_DIR)
        if (!dir.isDirectory) return out
        for (id in ids) {
            val safe = id.replace(Regex("[^A-Za-z0-9_.-]"), "_")
            val f = File(dir, "$safe.txt")
            if (f.isFile) {
                runCatching { out[id] = f.readText(Charsets.UTF_8) }
            }
        }
        return out
    }

    fun saveCheckpoint(jobDir: File, cp: TranscriptionCheckpoint) {
        runCatching {
            jobDir.mkdirs()
            val target = File(jobDir, CHECKPOINT_FILE)
            val tmp = File(jobDir, "$CHECKPOINT_FILE.tmp")
            tmp.writeText(json.encodeToString(TranscriptionCheckpoint.serializer(), cp), Charsets.UTF_8)
            if (!tmp.renameTo(target)) {
                target.writeText(tmp.readText(Charsets.UTF_8), Charsets.UTF_8)
                tmp.delete()
            }
        }
    }

    fun loadCheckpoint(jobDir: File): TranscriptionCheckpoint? {
        val f = File(jobDir, CHECKPOINT_FILE)
        if (!f.isFile) return null
        return runCatching {
            json.decodeFromString(TranscriptionCheckpoint.serializer(), f.readText(Charsets.UTF_8))
        }.getOrNull()
    }

    fun markPaused(jobDir: File) {
        runCatching {
            jobDir.mkdirs()
            File(jobDir, PAUSED_FLAG).writeText(System.currentTimeMillis().toString(), Charsets.UTF_8)
        }
    }

    fun clearPaused(jobDir: File) {
        runCatching { File(jobDir, PAUSED_FLAG).delete() }
    }

    fun isPaused(jobDir: File): Boolean = File(jobDir, PAUSED_FLAG).isFile

    fun markCompleted(jobDir: File) {
        runCatching {
            File(jobDir, PAUSED_FLAG).delete()
            File(jobDir, CHECKPOINT_FILE).delete()
            File(jobDir, TEMP_IMAGES_DIR).deleteRecursively()
            File(jobDir, COMPLETED_FLAG).writeText(System.currentTimeMillis().toString(), Charsets.UTF_8)
        }
    }

    fun deleteJob(jobDir: File) {
        runCatching { jobDir.deleteRecursively() }
    }

    fun findLatestResumable(jobsRoot: File): ResumableJobInfo? {
        if (!jobsRoot.isDirectory) return null
        val candidates = jobsRoot.listFiles { f -> f.isDirectory }
            ?.sortedByDescending { it.lastModified() }
            ?: return null
        for (dir in candidates) {
            if (File(dir, COMPLETED_FLAG).isFile) continue
            val specFile = File(dir, SPEC_FILE)
            if (!specFile.isFile) continue
            val cp = loadCheckpoint(dir) ?: continue
            if (cp.completedItems <= 0 || (cp.totalItems > 0 && cp.completedItems >= cp.totalItems)) {
                continue
            }
            val spec = runCatching {
                json.decodeFromString(TranscriptionJobSpec.serializer(), specFile.readText(Charsets.UTF_8))
            }.getOrNull() ?: continue
            return ResumableJobInfo(
                jobDir = dir,
                specFile = specFile,
                spec = spec,
                checkpoint = cp,
                wasPausedByUser = isPaused(dir),
            )
        }
        return null
    }
}
