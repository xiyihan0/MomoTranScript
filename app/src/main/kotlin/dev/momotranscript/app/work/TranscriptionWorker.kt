package dev.momotranscript.app.work

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.net.Uri
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.work.CoroutineWorker
import androidx.work.Data
import androidx.work.ForegroundInfo
import androidx.work.OneTimeWorkRequest
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import dev.momotranscript.app.BuildConfig
import dev.momotranscript.app.MainActivity
import dev.momotranscript.app.R
import dev.momotranscript.app.avatar.OnnxAvatarMatcher
import dev.momotranscript.app.engine.BatchTranscriptionPipeline
import dev.momotranscript.app.engine.OnnxTranscriptionEngine
import dev.momotranscript.app.engine.PartialBatchSnapshot
import dev.momotranscript.app.export.MoeTalkBranchPacker
import dev.momotranscript.app.input.BatchInputSummary
import dev.momotranscript.app.input.ImageInputItem
import dev.momotranscript.app.nativecore.NativeCore
import dev.momotranscript.app.ui.PagePreviewRecord
import kotlinx.coroutines.CancellationException
import kotlinx.serialization.Serializable
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.Json
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Serializable
data class TranscriptionJobSpec(
    val jobId: String,
    val kind: String, // "images" | "pdf"
    val title: String,
    val folderName: String? = null,
    val uris: List<String>,
    val displayNames: List<String>,
    val useBranches: Boolean = true,
    val useCcipVisual: Boolean = true,
    val backendMode: Int = OnnxTranscriptionEngine.BACKEND_AUTO_GPU_CPU,
)

/**
 * Foreground Jetpack `WorkManager` worker for long-running folder & 200-page PDF transcription.
 * Features:
 * - Keeps execution alive across screen lock / app backgrounding with progress notification.
 * - Writes live device/stage profiling logs to `run.log` (and Logcat) for mobile debugging.
 * - Atomically flushes completed pages (`result.txt`, `result.moetalk.json`, `preview_pages.json`)
 *   after every page so long PDFs can be previewed and exported mid-run or after cancellation.
 */
class TranscriptionWorker(
    appContext: Context,
    params: WorkerParameters,
) : CoroutineWorker(appContext, params) {

    override suspend fun doWork(): Result {
        val specFilePath = inputData.getString(KEY_SPEC_FILE) ?: return Result.failure(
            workDataOf(KEY_ERROR to "缺少任务描述文件"),
        )
        val specFile = File(specFilePath)
        if (!specFile.isFile) {
            return Result.failure(workDataOf(KEY_ERROR to "任务描述文件不存在"))
        }

        val spec = runCatching {
            Json.decodeFromString<TranscriptionJobSpec>(specFile.readText(Charsets.UTF_8))
        }.getOrElse { e ->
            return Result.failure(workDataOf(KEY_ERROR to "解析任务失败：${e.message}"))
        }

        val jobsRoot = File(applicationContext.filesDir, "jobs").apply { mkdirs() }
        pruneOldJobs(jobsRoot, keepJobId = spec.jobId, maxKeep = 2)

        val bundleRoot = File(applicationContext.getExternalFilesDir(null) ?: applicationContext.filesDir, "bundles")
        val outDir = File(jobsRoot, spec.jobId).apply { mkdirs() }
        val txtFile = File(outDir, "result.txt")
        val jsonFile = File(outDir, "result.moetalk.json")
        val previewFile = File(outDir, "preview_pages.json")
        val logFile = File(outDir, "run.log")
        val previewCacheDir = File(outDir, "previews")

        CheckpointStore.clearPaused(outDir)
        val resumeCheckpoint = CheckpointStore.loadCheckpoint(outDir)
        val initialDone = resumeCheckpoint?.completedItems ?: 0
        val initialTotal = (resumeCheckpoint?.totalItems ?: spec.uris.size).coerceAtLeast(1)

        val timeFmt = SimpleDateFormat("HH:mm:ss", Locale.US)
        var logSeq = 0
        var flushSeq = 0
        var profileSummary = ""
        var latestSnap: PartialBatchSnapshot? = null

        fun currentMemMb(): String {
            val rt = Runtime.getRuntime()
            val jvmMb = (rt.totalMemory() - rt.freeMemory()) / (1024 * 1024)
            val nativeMb = Debug.getNativeHeapAllocatedSize() / (1024 * 1024)
            return "jvm=${jvmMb}M native=${nativeMb}M"
        }

        fun appendLog(msg: String) {
            val line = "[${timeFmt.format(Date())}] $msg"
            Log.i(LOG_TAG, line)
            runCatching {
                logFile.appendText("$line\n", Charsets.UTF_8)
            }
            logSeq++
        }

        fun pushProgress(
            done: Int,
            total: Int,
            label: String,
            detail: String,
        ) {
            val snap = latestSnap
            setProgressAsync(
                workDataOf(
                    KEY_JOB_ID to spec.jobId,
                    KEY_DONE to done,
                    KEY_TOTAL to total,
                    KEY_LABEL to label,
                    KEY_PROGRESS_TEXT to detail,
                    KEY_TITLE to (snap?.title ?: spec.title),
                    KEY_TXT_PATH to txtFile.absolutePath,
                    KEY_JSON_PATH to jsonFile.absolutePath,
                    KEY_PREVIEW_PATH to previewFile.absolutePath,
                    KEY_LOG_PATH to logFile.absolutePath,
                    KEY_FLUSH_SEQ to flushSeq,
                    KEY_LOG_SEQ to logSeq,
                    KEY_PROFILE_SUMMARY to profileSummary,
                    KEY_ITEM_COUNT to (snap?.completedItems ?: done),
                    KEY_WINDOW_COUNT to (snap?.totalWindows ?: resumeCheckpoint?.totalWindows ?: 0),
                    KEY_LINE_COUNT to (snap?.totalLines ?: resumeCheckpoint?.totalLines ?: 0),
                    KEY_MATCHED_AVATARS to (snap?.matchedOfficialAvatars ?: resumeCheckpoint?.matchedOfficialAvatars ?: 0),
                    KEY_MATCHED_VISUALS to (snap?.matchedVisualAssets ?: resumeCheckpoint?.matchedVisualAssets ?: 0),
                    KEY_ELAPSED_MS to (snap?.elapsedMs ?: resumeCheckpoint?.elapsedMs ?: 0L),
                    KEY_USED_BRANCHES to (snap?.usedBranchPacking ?: spec.useBranches),
                ),
            )
        }

        ensureNotificationChannel()
        val notifManager = applicationContext.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        val startStatusText = if (initialDone > 0) {
            "从第 ${initialDone + 1}/$initialTotal 页恢复转录…"
        } else {
            "正在初始化模型…"
        }
        val fgResult = runCatching {
            setForeground(createForegroundInfo(initialDone, initialTotal, spec.title, startStatusText))
        }

        val pm = applicationContext.getSystemService(Context.POWER_SERVICE) as PowerManager
        val wakeLock = runCatching {
            pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "MomoTranScript:TranscribeJob").apply {
                setReferenceCounted(false)
                acquire(4L * 3600_000L)
            }
        }.getOrNull()

        appendLog(
            "=== MomoTranScript v${BuildConfig.VERSION_NAME} (build ${BuildConfig.VERSION_CODE}) | job=${spec.jobId}" +
                (if (initialDone > 0) " [RESUME $initialDone/$initialTotal]" else "") + " ===",
        )
        appendLog(
            "设备: ${Build.MANUFACTURER} ${Build.MODEL} | Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT}) | " +
                "ABI=${Build.SUPPORTED_ABIS.firstOrNull() ?: "?"} | CPU=${Runtime.getRuntime().availableProcessors()}核 | ${currentMemMb()}",
        )
        val ignoringOpt = runCatching { pm.isIgnoringBatteryOptimizations(applicationContext.packageName) }.getOrDefault(false)
        appendLog(
            "前台保活: FGS=${if (fgResult.isSuccess) "OK" else "FAIL(${fgResult.exceptionOrNull()?.javaClass?.simpleName}: ${fgResult.exceptionOrNull()?.message})"} " +
                "| WakeLock=${if (wakeLock?.isHeld == true) "ON" else "OFF"} | 电池白名单=${if (ignoringOpt) "已放行" else "受系统管控"}",
        )
        appendLog("硬件探针: ${NativeCore.mnnDeviceInfo()}")
        appendLog(
            "任务: kind=${spec.kind} title=${spec.title} branches=${spec.useBranches} ccip=${spec.useCcipVisual} backendMode=${spec.backendMode}",
        )
        pushProgress(initialDone, initialTotal, if (initialDone > 0) "断点恢复" else "初始化", startStatusText)

        val inputSummary: BatchInputSummary = when (spec.kind) {
            "pdf" -> BatchInputSummary.PdfDocument(
                uri = Uri.parse(spec.uris.first()),
                displayName = spec.displayNames.firstOrNull() ?: spec.title,
                pageCount = 0,
                directImageCount = 0,
                fallbackPageCount = 0,
                estimatedTotalWindows = 0,
                firstPageSize = null,
            )
            else -> BatchInputSummary.Images(
                items = spec.uris.indices.map { idx ->
                    ImageInputItem(
                        uri = Uri.parse(spec.uris[idx]),
                        displayName = spec.displayNames.getOrElse(idx) { "${idx + 1}.png" },
                        width = 0,
                        height = 0,
                        plannedWindows = 0,
                    )
                },
                folderName = spec.folderName,
            )
        }

        val packMode = if (spec.useBranches) {
            MoeTalkBranchPacker.MultiItemPackMode.BRANCH_PER_ITEM
        } else {
            MoeTalkBranchPacker.MultiItemPackMode.FLAT_SINGLE_STREAM
        }

        // Cumulative profiling accumulators
        var sumExtractMs = 0L
        var sumDetMs = 0L
        var sumCvMs = 0L
        var sumRecSMs = 0L
        var sumRecMMs = 0L
        var sumCcipMs = 0L
        var sumOtherMs = 0L
        var sumPageMs = 0L
        var sumWins = 0

        return try {
            var lastNotifMs = 0L
            val output = OnnxTranscriptionEngine(
                bundleRoot = bundleRoot,
                backendMode = spec.backendMode,
                cacheDir = applicationContext.cacheDir,
            ).use { engine ->
                val ep = engine.initProfile
                appendLog(
                    "引擎就绪 (threads=${ep.intraOpThreads}): vocab=${ep.vocabLoadMs}ms " +
                        "det=${ep.detLoadMs}ms[${ep.detEp}] recS=${ep.recSmallLoadMs}ms[${ep.recSmallEp}] " +
                        (if (ep.hasCascadeMedium) "recM=${ep.recMediumLoadMs}ms[${ep.recMediumEp}]" else "recM=off") +
                        " | ${currentMemMb()}",
                )
                pushProgress(0, spec.uris.size.coerceAtLeast(1), "引擎就绪", "模型加载完成，准备推理…")

                val matcher = if (spec.useCcipVisual && OnnxAvatarMatcher.isAvailable(bundleRoot)) {
                    pushProgress(0, spec.uris.size.coerceAtLeast(1), "视觉索引", "加载 CCIP 视觉模型与索引…")
                    OnnxAvatarMatcher(
                        bundleRoot = bundleRoot,
                        backendMode = spec.backendMode,
                        cacheDir = applicationContext.cacheDir,
                    ).also { m ->
                        val cp = m.initProfile
                        appendLog(
                            "CCIP 就绪 (threads=${cp.intraOpThreads}, EP=${cp.ep}): gallery=${cp.galleryLoadMs}ms " +
                                "(skins=${cp.skinCount}, chars=${cp.charCount}, emojis=${cp.emojiCount}) " +
                                "ccip=${cp.ccipLoadMs}ms | ${currentMemMb()}",
                        )
                    }
                } else {
                    appendLog("CCIP 视觉匹配: 关闭")
                    null
                }

                try {
                    BatchTranscriptionPipeline.run(
                        context = applicationContext,
                        engine = engine,
                        input = inputSummary,
                        avatarMatcher = matcher,
                        packMode = packMode,
                        previewCacheDir = previewCacheDir,
                        checkpointDir = outDir,
                        jobId = spec.jobId,
                        resumeCheckpoint = resumeCheckpoint,
                        onLog = { msg ->
                            appendLog("$msg | ${currentMemMb()}")
                        },
                        onPageCompleted = { snap ->
                            latestSnap = snap
                            val p = snap.lastPagePerf
                            val detTotal = p.windowPlanMs + p.detPrepMs + p.detInferMs + p.detDecodeMs
                            val known = p.extractMs + detTotal + p.cvLinesMs + p.recSmallMs + p.recMediumMs + p.ccipTotalMs
                            val other = (p.pageTotalMs - known).coerceAtLeast(0L)

                            sumExtractMs += p.extractMs
                            sumDetMs += detTotal
                            sumCvMs += p.cvLinesMs
                            sumRecSMs += p.recSmallMs
                            sumRecMMs += p.recMediumMs
                            sumCcipMs += p.ccipTotalMs
                            sumOtherMs += other
                            sumPageMs += p.pageTotalMs
                            sumWins += p.windowCount

                            val msPerWin = if (p.windowCount > 0) p.detInferMs / p.windowCount else 0L
                            val avgPageMs = if (snap.completedItems > 0) sumPageMs / snap.completedItems else 0L
                            val remainPages = (snap.totalItems - snap.completedItems).coerceAtLeast(0)
                            val etaStr = formatDurationMs(avgPageMs * remainPages)

                            val ccipPart = if (matcher != null) {
                                " ccip=${p.ccipTotalMs}ms(av ${p.ccipAvTotal}->${p.ccipAvUnique},ins ${p.ccipInsTotal}->${p.ccipInsUnique},inf=${p.ccipInferMs},st2=${p.ccipStage2Ms})"
                            } else {
                                ""
                            }
                            val recMPart = if (p.escalatedLines > 0) {
                                " recM=${p.recMediumMs}ms(${p.escalatedLines}行/采${p.adoptedMedium})"
                            } else {
                                ""
                            }
                            appendLog(
                                "${p.label} [${snap.completedItems}/${snap.totalItems}] ${p.width}x${p.height} " +
                                    "${p.windowCount}w ${p.lineCount}L | ${"%.2f".format(p.pageTotalMs / 1000.0)}s | " +
                                    "ext=${p.extractMs}ms det=${detTotal}ms(${msPerWin}ms/w) cv=${p.cvLinesMs}ms " +
                                    "recS=${p.recSmallMs}ms$recMPart$ccipPart | ${currentMemMb()} | ETA $etaStr",
                            )

                            val denom = sumPageMs.coerceAtLeast(1L).toDouble()
                            val avgDetWin = if (sumWins > 0) sumDetMs / sumWins else 0L
                            profileSummary = buildString {
                                append("均 ${"%.2f".format(avgPageMs / 1000.0)}s/页 · det ${avgDetWin}ms/窗")
                                if (remainPages > 0) append(" · 预计剩 $etaStr")
                                append(" | ")
                                append("det ${(sumDetMs * 100.0 / denom).toInt()}% · ")
                                append("recS ${(sumRecSMs * 100.0 / denom).toInt()}%")
                                if (sumRecMMs > 0L) append(" · recM ${(sumRecMMs * 100.0 / denom).toInt()}%")
                                if (sumCcipMs > 0L) append(" · ccip ${(sumCcipMs * 100.0 / denom).toInt()}%")
                                if (sumExtractMs > 0L) append(" · 抽图 ${(sumExtractMs * 100.0 / denom).toInt()}%")
                            }

                            // Atomic streaming flush to disk so UI can preview/export completed pages immediately!
                            atomicWriteText(txtFile, snap.combinedMomoScript)
                            snap.moetalkArchiveJson?.let { atomicWriteText(jsonFile, it) }
                            atomicWriteText(
                                previewFile,
                                Json.encodeToString(ListSerializer(PagePreviewRecord.serializer()), snap.previewPages),
                            )
                            flushSeq++
                            pushProgress(
                                done = snap.completedItems,
                                total = snap.totalItems,
                                label = p.label,
                                detail = "已完成 ${snap.completedItems}/${snap.totalItems} · 本页 ${"%.2f".format(p.pageTotalMs / 1000.0)}s · ETA $etaStr",
                            )
                        },
                    ) { prog ->
                        val now = System.currentTimeMillis()
                        if (now - lastNotifMs >= 500L || prog.completedItems == prog.totalItems) {
                            lastNotifMs = now
                            pushProgress(
                                done = prog.completedItems,
                                total = prog.totalItems,
                                label = prog.currentLabel,
                                detail = prog.detail,
                            )
                            // Update the active foreground notification directly via NotificationManager
                            // (never re-call setForegroundAsync in background on Android 12-16!)
                            runCatching {
                                notifManager.notify(
                                    NOTIFICATION_ID,
                                    buildNotification(
                                        prog.completedItems,
                                        prog.totalItems.coerceAtLeast(1),
                                        spec.title,
                                        "[${prog.completedItems}/${prog.totalItems}] ${prog.currentLabel} · ${prog.detail}",
                                    ),
                                )
                            }
                        }
                    }
                } finally {
                    matcher?.close()
                }
            }

            atomicWriteText(txtFile, output.combinedMomoScript)
            atomicWriteText(jsonFile, output.moetalkArchiveJson)
            atomicWriteText(
                previewFile,
                Json.encodeToString(ListSerializer(PagePreviewRecord.serializer()), output.previewPages),
            )
            CheckpointStore.markCompleted(outDir)
            flushSeq++
            appendLog(
                "=== 全部完成: ${output.itemCount} 页 / ${output.totalWindows} 窗 / ${output.totalLines} 行 / " +
                    "总耗时 ${formatDurationMs(output.totalElapsedMs)} | $profileSummary ===",
            )

            Result.success(
                workDataOf(
                    KEY_JOB_ID to spec.jobId,
                    KEY_TITLE to output.title,
                    KEY_TXT_PATH to txtFile.absolutePath,
                    KEY_JSON_PATH to jsonFile.absolutePath,
                    KEY_PREVIEW_PATH to previewFile.absolutePath,
                    KEY_LOG_PATH to logFile.absolutePath,
                    KEY_PROFILE_SUMMARY to profileSummary,
                    KEY_ITEM_COUNT to output.itemCount,
                    KEY_TOTAL to output.itemCount,
                    KEY_WINDOW_COUNT to output.totalWindows,
                    KEY_LINE_COUNT to output.totalLines,
                    KEY_MATCHED_AVATARS to output.matchedOfficialAvatars,
                    KEY_MATCHED_VISUALS to output.matchedVisualAssets,
                    KEY_ELAPSED_MS to output.totalElapsedMs,
                    KEY_USED_BRANCHES to output.usedBranchPacking,
                ),
            )
        } catch (ce: CancellationException) {
            val doneCount = latestSnap?.completedItems ?: initialDone
            val reasonStr = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) "stopReason=$stopReason" else "cancelled"
            val modeLabel = if (CheckpointStore.isPaused(outDir)) "已暂停" else "已中断($reasonStr)"
            appendLog("=== 任务$modeLabel：已保存前 $doneCount 页断点，随时可继续 ===")
            throw ce
        } catch (e: Exception) {
            appendLog("!!! 运行异常: ${e.javaClass.simpleName}: ${e.message}")
            Result.failure(
                workDataOf(
                    KEY_ERROR to (e.message ?: "端侧转录失败"),
                    KEY_LOG_PATH to logFile.absolutePath,
                    KEY_PROFILE_SUMMARY to profileSummary,
                ),
            )
        } finally {
            runCatching {
                if (wakeLock?.isHeld == true) wakeLock.release()
            }
        }
    }

    override suspend fun getForegroundInfo(): ForegroundInfo {
        ensureNotificationChannel()
        return createForegroundInfo(0, 1, "MomoTranScript", "正在离线转录…")
    }

    private fun atomicWriteText(target: File, content: String) {
        val tmp = File(target.parentFile, "${target.name}.tmp")
        tmp.writeText(content, Charsets.UTF_8)
        if (!tmp.renameTo(target)) {
            target.writeText(content, Charsets.UTF_8)
            tmp.delete()
        }
    }

    private fun pruneOldJobs(jobsRoot: File, keepJobId: String, maxKeep: Int) {
        runCatching {
            val dirs = jobsRoot.listFiles { f -> f.isDirectory && f.name != keepJobId }
                ?.sortedByDescending { it.lastModified() }
                ?: return
            dirs.drop(maxKeep).forEach { it.deleteRecursively() }
        }
    }

    private fun formatDurationMs(ms: Long): String {
        val totalSec = (ms / 1000L).coerceAtLeast(0L)
        val h = totalSec / 3600L
        val m = (totalSec % 3600L) / 60L
        val s = totalSec % 60L
        return when {
            h > 0L -> "%dh%02dm%02ds".format(h, m, s)
            m > 0L -> "%dm%02ds".format(m, s)
            else -> "%ds".format(s)
        }
    }

    private fun ensureNotificationChannel() {
        val nm = applicationContext.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        val channel = NotificationChannel(
            CHANNEL_ID,
            "离线转录进度",
            NotificationManager.IMPORTANCE_LOW,
        ).apply {
            description = "显示批量图片与 PDF 离线转录进度"
        }
        nm.createNotificationChannel(channel)
    }

    private fun buildNotification(done: Int, total: Int, title: String, detail: String): Notification {
        val launchIntent = Intent(applicationContext, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        val piFlags = PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        val contentIntent = PendingIntent.getActivity(applicationContext, 0, launchIntent, piFlags)
        return NotificationCompat.Builder(applicationContext, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_stat_momotranscript)
            .setContentTitle("MomoTranScript · $title")
            .setContentText(detail)
            .setContentIntent(contentIntent)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setSilent(true)
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
            .setProgress(total, done.coerceIn(0, total), total <= 0)
            .build()
    }

    private fun createForegroundInfo(done: Int, total: Int, title: String, detail: String): ForegroundInfo {
        val notification = buildNotification(done, total, title, detail)
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            ForegroundInfo(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            ForegroundInfo(NOTIFICATION_ID, notification)
        }
    }

    companion object {
        private const val LOG_TAG = "MomoTranScript"
        const val UNIQUE_WORK_NAME = "mmt_transcription_job"
        const val KEY_JOB_ID = "job_id"
        const val KEY_SPEC_FILE = "spec_file"
        const val KEY_DONE = "done"
        const val KEY_TOTAL = "total"
        const val KEY_LABEL = "label"
        const val KEY_PROGRESS_TEXT = "progress_text"
        const val KEY_ERROR = "error"
        const val KEY_TITLE = "title"
        const val KEY_TXT_PATH = "txt_path"
        const val KEY_JSON_PATH = "json_path"
        const val KEY_PREVIEW_PATH = "preview_path"
        const val KEY_LOG_PATH = "log_path"
        const val KEY_FLUSH_SEQ = "flush_seq"
        const val KEY_LOG_SEQ = "log_seq"
        const val KEY_PROFILE_SUMMARY = "profile_summary"
        const val KEY_ITEM_COUNT = "item_count"
        const val KEY_WINDOW_COUNT = "window_count"
        const val KEY_LINE_COUNT = "line_count"
        const val KEY_MATCHED_AVATARS = "matched_avatars"
        const val KEY_MATCHED_VISUALS = "matched_visuals"
        const val KEY_ELAPSED_MS = "elapsed_ms"
        const val KEY_USED_BRANCHES = "used_branches"

        private const val CHANNEL_ID = "mmt_transcription_progress"
        private const val NOTIFICATION_ID = 1042

        fun buildRequest(context: Context, spec: TranscriptionJobSpec): OneTimeWorkRequest {
            val dir = File(context.filesDir, "jobs/${spec.jobId}").apply { mkdirs() }
            val specFile = File(dir, "spec.json")
            specFile.writeText(Json.encodeToString(TranscriptionJobSpec.serializer(), spec), Charsets.UTF_8)
            val data: Data = workDataOf(KEY_SPEC_FILE to specFile.absolutePath)
            return OneTimeWorkRequestBuilder<TranscriptionWorker>()
                .setInputData(data)
                .build()
        }

        fun buildResumeRequest(specFile: File): OneTimeWorkRequest {
            val data: Data = workDataOf(KEY_SPEC_FILE to specFile.absolutePath)
            return OneTimeWorkRequestBuilder<TranscriptionWorker>()
                .setInputData(data)
                .build()
        }
    }
}
