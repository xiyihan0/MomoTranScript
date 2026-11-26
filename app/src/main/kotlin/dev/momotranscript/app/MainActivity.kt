package dev.momotranscript.app

import android.Manifest
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.DocumentsContract
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Add
import androidx.compose.material.icons.rounded.Build
import androidx.compose.material.icons.rounded.CheckCircle
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.Edit
import androidx.compose.material.icons.rounded.Face
import androidx.compose.material.icons.rounded.Info
import androidx.compose.material.icons.rounded.KeyboardArrowDown
import androidx.compose.material.icons.rounded.KeyboardArrowUp
import androidx.compose.material.icons.rounded.List
import androidx.compose.material.icons.rounded.Menu
import androidx.compose.material.icons.rounded.PlayArrow
import androidx.compose.material.icons.rounded.Refresh
import androidx.compose.material.icons.rounded.Search
import androidx.compose.material.icons.rounded.Settings
import androidx.compose.material.icons.rounded.Share
import androidx.compose.material.icons.rounded.Star
import androidx.compose.material3.Badge
import androidx.compose.material3.BadgedBox
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.work.ExistingWorkPolicy
import androidx.work.WorkInfo
import androidx.work.WorkManager
import dev.momotranscript.app.avatar.OnnxAvatarMatcher
import dev.momotranscript.app.engine.BatchProgress
import dev.momotranscript.app.engine.OnnxTranscriptionEngine
import dev.momotranscript.app.input.BatchInputSummary
import dev.momotranscript.app.input.InputBatchScanner
import dev.momotranscript.app.input.NaturalSort
import dev.momotranscript.app.model.ModelDownloadProgress
import dev.momotranscript.app.model.ModelPackCatalog
import dev.momotranscript.app.model.ModelScopeConfig
import dev.momotranscript.app.model.ModelScopeDownloader
import dev.momotranscript.app.nativecore.NativeCore
import dev.momotranscript.app.ui.DetectionOverlayViewer
import dev.momotranscript.app.ui.ModelsTab
import dev.momotranscript.app.ui.PagePreviewRecord
import dev.momotranscript.app.ui.theme.MomoTranScriptTheme
import dev.momotranscript.app.work.TranscriptionJobSpec
import dev.momotranscript.app.work.TranscriptionWorker
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.Json
import java.io.File
import java.util.UUID

private sealed interface IncomingPayload {
    data class Images(val uris: List<Uri>) : IncomingPayload
    data class Pdf(val uri: Uri) : IncomingPayload
}

private data class CompletedJobSummary(
    val title: String,
    val txtPath: String,
    val jsonPath: String,
    val itemCount: Int,
    val totalItems: Int = itemCount,
    val isPartial: Boolean = false,
    val windowCount: Int,
    val lineCount: Int,
    val matchedAvatars: Int,
    val matchedVisuals: Int,
    val elapsedMs: Long,
    val usedBranches: Boolean,
)

private const val PREFS_NAME = "mmt_settings"
private const val PREF_REPO_ID = "ms_repo_id"
private const val PREF_REVISION = "ms_revision"
private const val PREF_PREFIX = "ms_prefix"
private const val PREF_TOKEN = "ms_token"
private const val PREF_BACKEND_MODE = "mnn_backend_mode"
private const val PREF_PRIVACY_CONSENT_V1 = "privacy_consent_v1"

class MainActivity : ComponentActivity() {
    private val incomingPayloadState = mutableStateOf<IncomingPayload?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        handleIncomingIntent(intent)

        setContent {
            MomoTranScriptTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    val sp = remember { getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE) }
                    var consentAccepted by remember { mutableStateOf(sp.getBoolean(PREF_PRIVACY_CONSENT_V1, false)) }
                    if (consentAccepted) {
                        MomoTranScriptApp(
                            incomingPayload = incomingPayloadState.value,
                        )
                    } else {
                        PrivacyConsentScreen(
                            onOpenUrl = { url ->
                                runCatching {
                                    startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url)))
                                }.onFailure {
                                    Toast.makeText(this, "无法打开链接：$url", Toast.LENGTH_SHORT).show()
                                }
                            },
                            onAccept = {
                                sp.edit().putBoolean(PREF_PRIVACY_CONSENT_V1, true).apply()
                                consentAccepted = true
                            },
                            onDecline = {
                                finishAffinity()
                            },
                        )
                    }
                }
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleIncomingIntent(intent)
    }

    private fun handleIncomingIntent(intent: Intent?) {
        if (intent == null) return
        val mime = intent.type.orEmpty()
        when (intent.action) {
            Intent.ACTION_VIEW -> {
                val uri = intent.data ?: return
                if (mime == "application/pdf" || NaturalSort.isPdfName(uri.lastPathSegment.orEmpty())) {
                    incomingPayloadState.value = IncomingPayload.Pdf(uri)
                }
            }
            Intent.ACTION_SEND -> {
                val uri = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    intent.getParcelableExtra(Intent.EXTRA_STREAM, Uri::class.java)
                } else {
                    @Suppress("DEPRECATION")
                    intent.getParcelableExtra(Intent.EXTRA_STREAM)
                } ?: return
                if (mime == "application/pdf" || NaturalSort.isPdfName(uri.lastPathSegment.orEmpty())) {
                    incomingPayloadState.value = IncomingPayload.Pdf(uri)
                } else {
                    incomingPayloadState.value = IncomingPayload.Images(listOf(uri))
                }
            }
            Intent.ACTION_SEND_MULTIPLE -> {
                val uris = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    intent.getParcelableArrayListExtra(Intent.EXTRA_STREAM, Uri::class.java)
                } else {
                    @Suppress("DEPRECATION")
                    intent.getParcelableArrayListExtra(Intent.EXTRA_STREAM)
                }.orEmpty()
                if (uris.isNotEmpty()) {
                    incomingPayloadState.value = IncomingPayload.Images(uris)
                }
            }
        }
    }
}

private fun loadModelScopeConfig(context: Context): ModelScopeConfig {
    val sp = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    return ModelScopeConfig(
        repoId = sp.getString(PREF_REPO_ID, ModelScopeConfig.DEFAULT_REPO_ID) ?: ModelScopeConfig.DEFAULT_REPO_ID,
        revision = sp.getString(PREF_REVISION, ModelScopeConfig.DEFAULT_REVISION) ?: ModelScopeConfig.DEFAULT_REVISION,
        pathPrefix = sp.getString(PREF_PREFIX, "") ?: "",
        accessToken = sp.getString(PREF_TOKEN, "") ?: "",
    )
}

private fun saveModelScopeConfig(context: Context, cfg: ModelScopeConfig) {
    context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).edit()
        .putString(PREF_REPO_ID, cfg.repoId)
        .putString(PREF_REVISION, cfg.revision)
        .putString(PREF_PREFIX, cfg.pathPrefix)
        .putString(PREF_TOKEN, cfg.accessToken)
        .apply()
}

private fun loadBackendMode(context: Context): Int {
    val sp = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    return sp.getInt(PREF_BACKEND_MODE, OnnxTranscriptionEngine.BACKEND_AUTO_GPU_CPU)
        .coerceIn(OnnxTranscriptionEngine.BACKEND_AUTO_GPU_CPU, OnnxTranscriptionEngine.BACKEND_CPU_FP16)
}

private fun saveBackendMode(context: Context, mode: Int) {
    context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).edit()
        .putInt(PREF_BACKEND_MODE, mode)
        .apply()
}

private fun importModelBundleFromTree(context: Context, treeUri: Uri, bundleRoot: File): Int {
    val wantedRelPaths = ModelPackCatalog.packs.flatMap { it.files }.map { it.relativePath }.toSet()
    val wantedByBasename = wantedRelPaths.associateBy { it.substringAfterLast('/') }
    var copied = 0

    fun walk(parentDocId: String, relPrefix: String) {
        val childrenUri = DocumentsContract.buildChildDocumentsUriUsingTree(treeUri, parentDocId)
        val projection = arrayOf(
            DocumentsContract.Document.COLUMN_DOCUMENT_ID,
            DocumentsContract.Document.COLUMN_DISPLAY_NAME,
            DocumentsContract.Document.COLUMN_MIME_TYPE,
        )
        context.contentResolver.query(childrenUri, projection, null, null, null)?.use { cursor ->
            val idCol = cursor.getColumnIndexOrThrow(DocumentsContract.Document.COLUMN_DOCUMENT_ID)
            val nameCol = cursor.getColumnIndexOrThrow(DocumentsContract.Document.COLUMN_DISPLAY_NAME)
            val mimeCol = cursor.getColumnIndexOrThrow(DocumentsContract.Document.COLUMN_MIME_TYPE)
            while (cursor.moveToNext()) {
                val docId = cursor.getString(idCol) ?: continue
                val name = cursor.getString(nameCol) ?: continue
                val mime = cursor.getString(mimeCol) ?: ""
                val rel = if (relPrefix.isEmpty()) name else "$relPrefix/$name"
                if (mime == DocumentsContract.Document.MIME_TYPE_DIR) {
                    walk(docId, rel)
                } else {
                    val targetRel = when {
                        rel in wantedRelPaths -> rel
                        name in wantedByBasename -> wantedByBasename[name]
                        else -> null
                    }
                    if (targetRel != null) {
                        val docUri = DocumentsContract.buildDocumentUriUsingTree(treeUri, docId)
                        val dst = File(bundleRoot, targetRel)
                        dst.parentFile?.mkdirs()
                        context.contentResolver.openInputStream(docUri)?.use { input ->
                            dst.outputStream().use { output -> input.copyTo(output) }
                        }
                        copied++
                    }
                }
            }
        }
    }

    walk(DocumentsContract.getTreeDocumentId(treeUri), "")
    return copied
}

@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
private fun MomoTranScriptApp(incomingPayload: IncomingPayload?) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val workManager = remember { WorkManager.getInstance(context) }
    val bundleRoot = remember { File(context.getExternalFilesDir(null) ?: context.filesDir, "bundles") }

    var currentTab by remember { mutableIntStateOf(0) } // 0=转录, 1=结果, 2=模型
    var hasUnreadResult by remember { mutableStateOf(false) }

    var packStatuses by remember { mutableStateOf(ModelPackCatalog.inspectInstalled(bundleRoot)) }
    val corePackReady = packStatuses.filter { it.spec.required }.all { it.isReady }
    val visualPackReady = OnnxAvatarMatcher.isAvailable(bundleRoot)

    var msConfig by remember { mutableStateOf(loadModelScopeConfig(context)) }
    var downloadJob by remember { mutableStateOf<Job?>(null) }
    var downloadingPackId by remember { mutableStateOf<String?>(null) }
    var downloadProgress by remember { mutableStateOf<ModelDownloadProgress?>(null) }
    var downloadError by remember { mutableStateOf<String?>(null) }

    var inputSummary by remember { mutableStateOf<BatchInputSummary?>(null) }
    var isScanningInput by remember { mutableStateOf(false) }
    var isImportingModels by remember { mutableStateOf(false) }
    var scanError by remember { mutableStateOf<String?>(null) }

    var useMoeTalkBranches by remember { mutableStateOf(true) }
    var useCcipVisualMatch by remember { mutableStateOf(true) }
    var selectedBackendMode by remember { mutableIntStateOf(loadBackendMode(context)) }

    var isJobRunning by remember { mutableStateOf(false) }
    var batchProgress by remember { mutableStateOf<BatchProgress?>(null) }
    var lastCompleted by remember { mutableStateOf<CompletedJobSummary?>(null) }
    var previewPages by remember { mutableStateOf<List<PagePreviewRecord>>(emptyList()) }
    var lastCompletedWorkId by remember { mutableStateOf<UUID?>(null) }
    var lastObservedFlushSeq by remember { mutableIntStateOf(-1) }
    var lastObservedLogSeq by remember { mutableIntStateOf(-1) }
    var activeLogPath by remember { mutableStateOf<String?>(null) }
    var liveLogText by remember { mutableStateOf("") }
    var liveProfileSummary by remember { mutableStateOf("") }
    var userEditedScript by remember { mutableStateOf(false) }
    var hasPendingNewerText by remember { mutableStateOf(false) }
    var runError by remember { mutableStateOf<String?>(null) }

    var scriptText by remember {
        mutableStateOf(
            "- 转录完成后的 MomoScript 正文会显示在这里\n" +
                "> 提示: 支持直接编辑、一键复制、导出 .txt 或导入 MoeTalk 的分支 .json 存档。",
        )
    }

    fun readLogTail(path: String?, maxChars: Int = 12000): String {
        if (path.isNullOrBlank()) return ""
        val f = File(path)
        if (!f.isFile) return ""
        val text = runCatching { f.readText(Charsets.UTF_8) }.getOrDefault("")
        return if (text.length > maxChars) "… (前略)\n" + text.takeLast(maxChars) else text
    }

    var coreVersion by remember { mutableStateOf(NativeCore.coreVersion()) }
    LaunchedEffect(Unit) {
        val mnnInfo = withContext(Dispatchers.IO) {
            runCatching { NativeCore.mnnDeviceInfo() }.getOrDefault("")
        }
        if (mnnInfo.isNotBlank()) {
            coreVersion = "${NativeCore.coreVersion()}\n$mnnInfo"
        }
    }
    val avifsPackFile = remember { File(bundleRoot, "avatar/stage2_templates.avifs.pack") }
    var packInspectNote by remember {
        mutableStateOf(
            if (avifsPackFile.isFile) NativeCore.inspectAvifsPack(avifsPackFile.absolutePath) else "",
        )
    }

    fun refreshPacks() {
        packStatuses = ModelPackCatalog.inspectInstalled(bundleRoot)
        packInspectNote = if (avifsPackFile.isFile) {
            NativeCore.inspectAvifsPack(avifsPackFile.absolutePath)
        } else {
            ""
        }
    }

    // Observe foreground WorkManager job (with live log & per-page streaming preview/export!)
    LaunchedEffect(workManager) {
        workManager.getWorkInfosForUniqueWorkFlow(TranscriptionWorker.UNIQUE_WORK_NAME).collect { infos ->
            val info = infos.firstOrNull() ?: return@collect
            when (info.state) {
                WorkInfo.State.ENQUEUED,
                WorkInfo.State.RUNNING,
                -> {
                    isJobRunning = true
                    runError = null
                    val p = info.progress
                    val done = p.getInt(TranscriptionWorker.KEY_DONE, 0)
                    val total = p.getInt(TranscriptionWorker.KEY_TOTAL, 0)
                    batchProgress = BatchProgress(
                        completedItems = done,
                        totalItems = total,
                        currentLabel = p.getString(TranscriptionWorker.KEY_LABEL) ?: "准备中",
                        detail = p.getString(TranscriptionWorker.KEY_PROGRESS_TEXT) ?: "正在初始化端侧引擎…",
                    )

                    val prof = p.getString(TranscriptionWorker.KEY_PROFILE_SUMMARY).orEmpty()
                    if (prof.isNotEmpty()) liveProfileSummary = prof

                    val logPath = p.getString(TranscriptionWorker.KEY_LOG_PATH)
                    val logSeq = p.getInt(TranscriptionWorker.KEY_LOG_SEQ, 0)
                    if (!logPath.isNullOrBlank()) {
                        activeLogPath = logPath
                        if (logSeq != lastObservedLogSeq) {
                            lastObservedLogSeq = logSeq
                            liveLogText = withContext(Dispatchers.IO) { readLogTail(logPath) }
                        }
                    }

                    val flushSeq = p.getInt(TranscriptionWorker.KEY_FLUSH_SEQ, 0)
                    val txtPath = p.getString(TranscriptionWorker.KEY_TXT_PATH).orEmpty()
                    val jsonPath = p.getString(TranscriptionWorker.KEY_JSON_PATH).orEmpty()
                    val previewPath = p.getString(TranscriptionWorker.KEY_PREVIEW_PATH).orEmpty()
                    if (done > 0 && flushSeq > 0 && flushSeq != lastObservedFlushSeq && txtPath.isNotBlank()) {
                        lastObservedFlushSeq = flushSeq
                        lastCompleted = CompletedJobSummary(
                            title = p.getString(TranscriptionWorker.KEY_TITLE) ?: "MomoTranScript",
                            txtPath = txtPath,
                            jsonPath = jsonPath,
                            itemCount = done,
                            totalItems = total.coerceAtLeast(done),
                            isPartial = total <= 0 || done < total,
                            windowCount = p.getInt(TranscriptionWorker.KEY_WINDOW_COUNT, 0),
                            lineCount = p.getInt(TranscriptionWorker.KEY_LINE_COUNT, 0),
                            matchedAvatars = p.getInt(TranscriptionWorker.KEY_MATCHED_AVATARS, 0),
                            matchedVisuals = p.getInt(TranscriptionWorker.KEY_MATCHED_VISUALS, 0),
                            elapsedMs = p.getLong(TranscriptionWorker.KEY_ELAPSED_MS, 0L),
                            usedBranches = p.getBoolean(TranscriptionWorker.KEY_USED_BRANCHES, false),
                        )
                        val (loadedTxt, loadedPreviews) = withContext(Dispatchers.IO) {
                            val txt = File(txtPath).takeIf { it.isFile }?.readText(Charsets.UTF_8)
                            val previews = File(previewPath).takeIf { it.isFile }?.let { pf ->
                                runCatching {
                                    Json { ignoreUnknownKeys = true }.decodeFromString(
                                        ListSerializer(PagePreviewRecord.serializer()),
                                        pf.readText(Charsets.UTF_8),
                                    )
                                }.getOrNull()
                            }.orEmpty()
                            txt to previews
                        }
                        if (loadedTxt != null) {
                            if (!userEditedScript) {
                                scriptText = loadedTxt
                                hasPendingNewerText = false
                            } else {
                                hasPendingNewerText = true
                            }
                        }
                        if (loadedPreviews.isNotEmpty()) {
                            previewPages = loadedPreviews
                        }
                        if (currentTab != 1) {
                            hasUnreadResult = true
                        }
                    }
                }
                WorkInfo.State.SUCCEEDED -> {
                    isJobRunning = false
                    if (lastCompletedWorkId != info.id) {
                        lastCompletedWorkId = info.id
                        val out = info.outputData
                        val txtPath = out.getString(TranscriptionWorker.KEY_TXT_PATH).orEmpty()
                        val jsonPath = out.getString(TranscriptionWorker.KEY_JSON_PATH).orEmpty()
                        val previewPath = out.getString(TranscriptionWorker.KEY_PREVIEW_PATH).orEmpty()
                        val logPath = out.getString(TranscriptionWorker.KEY_LOG_PATH)
                        if (!logPath.isNullOrBlank()) {
                            activeLogPath = logPath
                            liveLogText = withContext(Dispatchers.IO) { readLogTail(logPath) }
                        }
                        val prof = out.getString(TranscriptionWorker.KEY_PROFILE_SUMMARY).orEmpty()
                        if (prof.isNotEmpty()) liveProfileSummary = prof

                        val itemCount = out.getInt(TranscriptionWorker.KEY_ITEM_COUNT, 0)
                        val summary = CompletedJobSummary(
                            title = out.getString(TranscriptionWorker.KEY_TITLE) ?: "MomoTranScript",
                            txtPath = txtPath,
                            jsonPath = jsonPath,
                            itemCount = itemCount,
                            totalItems = itemCount,
                            isPartial = false,
                            windowCount = out.getInt(TranscriptionWorker.KEY_WINDOW_COUNT, 0),
                            lineCount = out.getInt(TranscriptionWorker.KEY_LINE_COUNT, 0),
                            matchedAvatars = out.getInt(TranscriptionWorker.KEY_MATCHED_AVATARS, 0),
                            matchedVisuals = out.getInt(TranscriptionWorker.KEY_MATCHED_VISUALS, 0),
                            elapsedMs = out.getLong(TranscriptionWorker.KEY_ELAPSED_MS, 0L),
                            usedBranches = out.getBoolean(TranscriptionWorker.KEY_USED_BRANCHES, false),
                        )
                        lastCompleted = summary
                        val (loadedTxt, loadedPreviews) = withContext(Dispatchers.IO) {
                            val txt = File(txtPath).takeIf { it.isFile }?.readText(Charsets.UTF_8)
                            val previews = File(previewPath).takeIf { it.isFile }?.let { pf ->
                                runCatching {
                                    Json { ignoreUnknownKeys = true }.decodeFromString(
                                        ListSerializer(PagePreviewRecord.serializer()),
                                        pf.readText(Charsets.UTF_8),
                                    )
                                }.getOrNull()
                            }.orEmpty()
                            txt to previews
                        }
                        if (loadedTxt != null) {
                            scriptText = loadedTxt
                            userEditedScript = false
                            hasPendingNewerText = false
                        }
                        previewPages = loadedPreviews
                        batchProgress = BatchProgress(
                            completedItems = summary.itemCount,
                            totalItems = summary.itemCount,
                            currentLabel = summary.title,
                            detail = "完成 (${"%.2f".format(summary.elapsedMs / 1000.0)}s)",
                        )
                        hasUnreadResult = true
                        currentTab = 1 // Jump smoothly to Result tab
                    }
                }
                WorkInfo.State.FAILED -> {
                    isJobRunning = false
                    val errLog = info.outputData.getString(TranscriptionWorker.KEY_LOG_PATH)
                    if (!errLog.isNullOrBlank()) {
                        activeLogPath = errLog
                        liveLogText = withContext(Dispatchers.IO) { readLogTail(errLog) }
                    }
                    runError = info.outputData.getString(TranscriptionWorker.KEY_ERROR) ?: "端侧转录失败"
                }
                WorkInfo.State.CANCELLED -> {
                    isJobRunning = false
                    activeLogPath?.let { lp ->
                        liveLogText = withContext(Dispatchers.IO) { readLogTail(lp) }
                    }
                    val partialDone = lastCompleted?.itemCount ?: 0
                    val partialTotal = lastCompleted?.totalItems ?: 0
                    runError = if (partialDone > 0) {
                        "已手动停止（已保留前 $partialDone/$partialTotal 页成果，可直接前往「结果」页预览或导出）"
                    } else {
                        "已手动停止转录任务"
                    }
                }
                else -> Unit
            }
        }
    }

    val notifPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
    ) { _ -> }

    LaunchedEffect(incomingPayload) {
        if (incomingPayload == null) return@LaunchedEffect
        currentTab = 0
        isScanningInput = true
        scanError = null
        try {
            inputSummary = withContext(Dispatchers.IO) {
                when (incomingPayload) {
                    is IncomingPayload.Images -> InputBatchScanner.probeImages(context, incomingPayload.uris)
                    is IncomingPayload.Pdf -> InputBatchScanner.scanPdfUri(context, incomingPayload.uri)
                }
            }
        } catch (e: Exception) {
            scanError = e.message ?: "读取输入失败"
        } finally {
            isScanningInput = false
        }
    }

    val photoPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickMultipleVisualMedia(maxItems = 50),
    ) { uris ->
        if (uris.isNotEmpty()) {
            scope.launch {
                isScanningInput = true
                scanError = null
                try {
                    inputSummary = withContext(Dispatchers.IO) {
                        InputBatchScanner.probeImages(context, uris)
                    }
                } catch (e: Exception) {
                    scanError = e.message ?: "读取图片失败"
                } finally {
                    isScanningInput = false
                }
            }
        }
    }

    val folderPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocumentTree(),
    ) { treeUri ->
        if (treeUri != null) {
            runCatching {
                context.contentResolver.takePersistableUriPermission(treeUri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            scope.launch {
                isScanningInput = true
                scanError = null
                try {
                    inputSummary = withContext(Dispatchers.IO) {
                        InputBatchScanner.scanFolderTree(context, treeUri)
                    }
                } catch (e: Exception) {
                    scanError = e.message ?: "扫描文件夹失败"
                } finally {
                    isScanningInput = false
                }
            }
        }
    }

    val pdfPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument(),
    ) { pdfUri ->
        if (pdfUri != null) {
            runCatching {
                context.contentResolver.takePersistableUriPermission(pdfUri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            scope.launch {
                isScanningInput = true
                scanError = null
                try {
                    inputSummary = withContext(Dispatchers.IO) {
                        InputBatchScanner.scanPdfUri(context, pdfUri)
                    }
                } catch (e: Exception) {
                    scanError = e.message ?: "流式扫描 PDF 失败"
                } finally {
                    isScanningInput = false
                }
            }
        }
    }

    val modelImportPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocumentTree(),
    ) { treeUri ->
        if (treeUri != null) {
            scope.launch {
                isImportingModels = true
                downloadError = null
                try {
                    val count = withContext(Dispatchers.IO) {
                        importModelBundleFromTree(context, treeUri, bundleRoot)
                    }
                    refreshPacks()
                    Toast.makeText(context, "已导入 $count 个模型/索引文件", Toast.LENGTH_SHORT).show()
                } catch (e: Exception) {
                    downloadError = "本地导入失败：${e.message}"
                } finally {
                    isImportingModels = false
                }
            }
        }
    }

    val exportTxtLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.CreateDocument("text/plain"),
    ) { uri ->
        if (uri != null) {
            scope.launch {
                runCatching {
                    withContext(Dispatchers.IO) {
                        context.contentResolver.openOutputStream(uri)?.use { out ->
                            out.write(scriptText.toByteArray(Charsets.UTF_8))
                        }
                    }
                }.onSuccess {
                    Toast.makeText(context, "已导出 MomoScript .txt", Toast.LENGTH_SHORT).show()
                }.onFailure { e ->
                    Toast.makeText(context, "导出失败：${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    val exportLogLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.CreateDocument("text/plain"),
    ) { uri ->
        if (uri != null) {
            scope.launch {
                runCatching {
                    withContext(Dispatchers.IO) {
                        val content = activeLogPath?.let { File(it).takeIf { f -> f.isFile }?.readText(Charsets.UTF_8) }
                            ?: liveLogText
                        context.contentResolver.openOutputStream(uri)?.use { out ->
                            out.write(content.toByteArray(Charsets.UTF_8))
                        }
                    }
                }.onSuccess {
                    Toast.makeText(context, "已导出运行日志 .log", Toast.LENGTH_SHORT).show()
                }.onFailure { e ->
                    Toast.makeText(context, "导出日志失败：${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    val exportMoeTalkLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.CreateDocument("application/json"),
    ) { uri ->
        val jsonPath = lastCompleted?.jsonPath
        if (uri != null && !jsonPath.isNullOrBlank()) {
            scope.launch {
                runCatching {
                    withContext(Dispatchers.IO) {
                        val jf = File(jsonPath)
                        require(jf.isFile) { "MoeTalk 存档尚未落盘，请稍候 1 页完成后重试" }
                        jf.inputStream().use { input ->
                            context.contentResolver.openOutputStream(uri)?.use { out ->
                                input.copyTo(out)
                            }
                        }
                    }
                }.onSuccess {
                    Toast.makeText(context, "已导出 MoeTalk 存档 .json", Toast.LENGTH_SHORT).show()
                }.onFailure { e ->
                    Toast.makeText(context, "导出失败：${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.background,
                ),
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(
                            modifier = Modifier
                                .size(30.dp)
                                .clip(RoundedCornerShape(9.dp))
                                .background(MaterialTheme.colorScheme.primary),
                            contentAlignment = Alignment.Center,
                        ) {
                            Text(
                                text = ">_",
                                color = MaterialTheme.colorScheme.onPrimary,
                                fontFamily = FontFamily.Monospace,
                                fontWeight = FontWeight.Bold,
                                style = MaterialTheme.typography.labelMedium,
                            )
                        }
                        Spacer(Modifier.width(10.dp))
                        Column {
                            Text(
                                text = "MomoTranScript",
                                fontWeight = FontWeight.Bold,
                                style = MaterialTheme.typography.titleLarge,
                            )
                            Text(
                                text = "v${BuildConfig.VERSION_NAME}",
                                style = MaterialTheme.typography.labelSmall,
                                fontFamily = FontFamily.Monospace,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                },
                actions = {
                    // Compact clickable status chip -> jumps straight to Models tab
                    val readyBg = Color(0xFFE6F6EC)
                    val readyFg = Color(0xFF1B7A43)
                    Surface(
                        color = if (corePackReady) readyBg else MaterialTheme.colorScheme.primaryContainer,
                        shape = RoundedCornerShape(999.dp),
                        modifier = Modifier
                            .padding(end = 12.dp)
                            .clip(RoundedCornerShape(999.dp))
                            .clickable { currentTab = 2 },
                    ) {
                        Row(
                            modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(4.dp),
                        ) {
                            Icon(
                                imageVector = if (corePackReady) Icons.Rounded.CheckCircle else Icons.Rounded.Add,
                                contentDescription = null,
                                tint = if (corePackReady) readyFg else MaterialTheme.colorScheme.onPrimaryContainer,
                                modifier = Modifier.size(14.dp),
                            )
                            Text(
                                text = if (corePackReady) "离线就绪" else "下载模型",
                                style = MaterialTheme.typography.labelSmall,
                                fontWeight = FontWeight.SemiBold,
                                color = if (corePackReady) readyFg else MaterialTheme.colorScheme.onPrimaryContainer,
                            )
                        }
                    }
                },
            )
        },
        bottomBar = {
            NavigationBar(
                containerColor = MaterialTheme.colorScheme.surfaceContainerLow,
                tonalElevation = 0.dp,
            ) {
                NavigationBarItem(
                    selected = currentTab == 0,
                    onClick = { currentTab = 0 },
                    icon = { Icon(Icons.Rounded.Star, contentDescription = "转录") },
                    label = { Text("转录") },
                )
                NavigationBarItem(
                    selected = currentTab == 1,
                    onClick = {
                        currentTab = 1
                        hasUnreadResult = false
                    },
                    icon = {
                        BadgedBox(
                            badge = { if (hasUnreadResult) Badge() },
                        ) {
                            Icon(Icons.Rounded.Edit, contentDescription = "结果")
                        }
                    },
                    label = { Text("结果") },
                )
                NavigationBarItem(
                    selected = currentTab == 2,
                    onClick = { currentTab = 2 },
                    icon = {
                        BadgedBox(
                            badge = { if (!corePackReady) Badge() },
                        ) {
                            Icon(Icons.Rounded.Settings, contentDescription = "模型")
                        }
                    },
                    label = { Text("模型") },
                )
            }
        },
    ) { innerPadding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding),
        ) {
            when (currentTab) {
                0 -> TranscribeTab(
                    corePackReady = corePackReady,
                    visualPackReady = visualPackReady,
                    inputSummary = inputSummary,
                    isScanningInput = isScanningInput,
                    scanError = scanError,
                    useMoeTalkBranches = useMoeTalkBranches,
                    onToggleBranches = { useMoeTalkBranches = it },
                    useCcipVisualMatch = useCcipVisualMatch,
                    onToggleCcip = { useCcipVisualMatch = it },
                    backendMode = selectedBackendMode,
                    onSelectBackendMode = { mode ->
                        selectedBackendMode = mode
                        saveBackendMode(context, mode)
                    },
                    isJobRunning = isJobRunning,
                    batchProgress = batchProgress,
                    liveCompletedCount = lastCompleted?.itemCount ?: 0,
                    liveProfileSummary = liveProfileSummary,
                    liveLogText = liveLogText,
                    runError = runError,
                    onPickImages = {
                        photoPicker.launch(
                            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly),
                        )
                    },
                    onPickFolder = { folderPicker.launch(null) },
                    onPickPdf = { pdfPicker.launch(arrayOf("application/pdf")) },
                    onClearInput = { inputSummary = null; scanError = null },
                    onGoToModels = { currentTab = 2 },
                    onPreviewPartialResult = {
                        hasUnreadResult = false
                        currentTab = 1
                    },
                    onCopyLog = {
                        scope.launch {
                            val fullLog = withContext(Dispatchers.IO) {
                                activeLogPath?.let { File(it).takeIf { f -> f.isFile }?.readText(Charsets.UTF_8) }
                                    ?: liveLogText
                            }
                            val cm = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                            cm.setPrimaryClip(ClipData.newPlainText("MomoTranScript Log", fullLog))
                            Toast.makeText(context, "已复制运行日志", Toast.LENGTH_SHORT).show()
                        }
                    },
                    onExportLog = {
                        val base = (lastCompleted?.title ?: "momotranscript").replace(Regex("[\\\\/:*?\"<>|]"), "_")
                        exportLogLauncher.launch("${base}_profile.log")
                    },
                    onStartTranscription = {
                        val currentInput = inputSummary ?: return@TranscribeTab
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
                            ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS)
                            != PackageManager.PERMISSION_GRANTED
                        ) {
                            notifPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
                        }

                        val jobId = UUID.randomUUID().toString().take(8)
                        val spec = when (currentInput) {
                            is BatchInputSummary.Images -> TranscriptionJobSpec(
                                jobId = jobId,
                                kind = "images",
                                title = currentInput.folderName ?: currentInput.items.firstOrNull()?.displayName ?: "图片批处理",
                                folderName = currentInput.folderName,
                                uris = currentInput.items.map { it.uri.toString() },
                                displayNames = currentInput.items.map { it.displayName },
                                useBranches = useMoeTalkBranches,
                                useCcipVisual = useCcipVisualMatch && visualPackReady,
                                backendMode = selectedBackendMode,
                            )
                            is BatchInputSummary.PdfDocument -> TranscriptionJobSpec(
                                jobId = jobId,
                                kind = "pdf",
                                title = currentInput.displayName.substringBeforeLast('.'),
                                uris = listOf(currentInput.uri.toString()),
                                displayNames = listOf(currentInput.displayName),
                                useBranches = useMoeTalkBranches,
                                useCcipVisual = useCcipVisualMatch && visualPackReady,
                                backendMode = selectedBackendMode,
                            )
                        }
                        runError = null
                        lastObservedFlushSeq = -1
                        lastObservedLogSeq = -1
                        liveLogText = ""
                        liveProfileSummary = ""
                        lastCompleted = null
                        previewPages = emptyList()
                        userEditedScript = false
                        hasPendingNewerText = false
                        val req = TranscriptionWorker.buildRequest(context, spec)
                        workManager.enqueueUniqueWork(
                            TranscriptionWorker.UNIQUE_WORK_NAME,
                            ExistingWorkPolicy.REPLACE,
                            req,
                        )
                    },
                    onStopTranscription = {
                        workManager.cancelUniqueWork(TranscriptionWorker.UNIQUE_WORK_NAME)
                    },
                )

                1 -> ResultTab(
                    scriptText = scriptText,
                    onScriptChange = {
                        scriptText = it
                        userEditedScript = true
                    },
                    isJobRunning = isJobRunning,
                    hasPendingNewerText = hasPendingNewerText,
                    onSyncLatestText = {
                        val txtPath = lastCompleted?.txtPath
                        if (!txtPath.isNullOrBlank()) {
                            scope.launch {
                                val loaded = withContext(Dispatchers.IO) {
                                    File(txtPath).takeIf { it.isFile }?.readText(Charsets.UTF_8)
                                }
                                if (loaded != null) {
                                    scriptText = loaded
                                    userEditedScript = false
                                    hasPendingNewerText = false
                                    Toast.makeText(context, "已同步至最新已完成页", Toast.LENGTH_SHORT).show()
                                }
                            }
                        }
                    },
                    lastCompleted = lastCompleted,
                    previewPages = previewPages,
                    onCopy = {
                        val cm = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                        cm.setPrimaryClip(ClipData.newPlainText("MomoScript", scriptText))
                        Toast.makeText(context, "已复制 MomoScript 正文", Toast.LENGTH_SHORT).show()
                    },
                    onExportTxt = {
                        val suffix = lastCompleted?.let { if (it.isPartial) "_p1-${it.itemCount}" else "" }.orEmpty()
                        val base = (lastCompleted?.title ?: "momoscript").replace(Regex("[\\\\/:*?\"<>|]"), "_")
                        exportTxtLauncher.launch("$base$suffix.txt")
                    },
                    onExportMoeTalk = {
                        val suffix = lastCompleted?.let { if (it.isPartial) "_p1-${it.itemCount}" else "" }.orEmpty()
                        val base = (lastCompleted?.title ?: "moetalk_archive").replace(Regex("[\\\\/:*?\"<>|]"), "_")
                        exportMoeTalkLauncher.launch("$base$suffix.json")
                    },
                )

                else -> ModelsTab(
                    packStatuses = packStatuses,
                    config = msConfig,
                    onSaveConfig = { updated ->
                        msConfig = updated
                        saveModelScopeConfig(context, updated)
                        Toast.makeText(context, "已保存 ModelScope 源设置", Toast.LENGTH_SHORT).show()
                    },
                    downloadingPackId = downloadingPackId,
                    downloadProgress = downloadProgress,
                    downloadError = downloadError,
                    isImportingLocal = isImportingModels,
                    isJobRunning = isJobRunning,
                    packInspectNote = packInspectNote,
                    coreVersion = coreVersion,
                    onDownloadPack = { pack ->
                        downloadError = null
                        downloadingPackId = pack.id
                        downloadProgress = ModelDownloadProgress(pack.id, 0, pack.files.size, "连接 ModelScope…", 0L, -1L, 0L)
                        downloadJob = scope.launch {
                            try {
                                ModelScopeDownloader.downloadPack(
                                    pack = pack,
                                    bundleRoot = bundleRoot,
                                    config = msConfig,
                                ) { prog ->
                                    downloadProgress = prog
                                }
                                refreshPacks()
                                Toast.makeText(context, "${pack.title} 下载并校验完成", Toast.LENGTH_SHORT).show()
                            } catch (_: CancellationException) {
                                downloadError = "已取消下载"
                            } catch (e: Exception) {
                                downloadError = e.message ?: "下载失败"
                            } finally {
                                downloadingPackId = null
                                downloadProgress = null
                                downloadJob = null
                            }
                        }
                    },
                    onCancelDownload = { downloadJob?.cancel() },
                    onDeletePack = { pack ->
                        scope.launch {
                            val removed = withContext(Dispatchers.IO) {
                                ModelScopeDownloader.deletePack(pack, bundleRoot)
                            }
                            refreshPacks()
                            Toast.makeText(context, "已删除 $removed 个文件", Toast.LENGTH_SHORT).show()
                        }
                    },
                    onImportLocalFolder = { modelImportPicker.launch(null) },
                    onRefresh = { refreshPacks() },
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun TranscribeTab(
    corePackReady: Boolean,
    visualPackReady: Boolean,
    inputSummary: BatchInputSummary?,
    isScanningInput: Boolean,
    scanError: String?,
    useMoeTalkBranches: Boolean,
    onToggleBranches: (Boolean) -> Unit,
    useCcipVisualMatch: Boolean,
    onToggleCcip: (Boolean) -> Unit,
    backendMode: Int,
    onSelectBackendMode: (Int) -> Unit,
    isJobRunning: Boolean,
    batchProgress: BatchProgress?,
    liveCompletedCount: Int,
    liveProfileSummary: String,
    liveLogText: String,
    runError: String?,
    onPickImages: () -> Unit,
    onPickFolder: () -> Unit,
    onPickPdf: () -> Unit,
    onClearInput: () -> Unit,
    onGoToModels: () -> Unit,
    onPreviewPartialResult: () -> Unit,
    onCopyLog: () -> Unit,
    onExportLog: () -> Unit,
    onStartTranscription: () -> Unit,
    onStopTranscription: () -> Unit,
) {
    var expandLogs by remember { mutableStateOf(true) }
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        // 1. 3 equal-width tactile source picker tiles
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            SourcePickerTile(
                icon = Icons.Rounded.Add,
                title = "图片",
                subtitle = "相册多选",
                tint = MaterialTheme.colorScheme.primary,
                containerColor = MaterialTheme.colorScheme.primaryContainer,
                enabled = !isJobRunning,
                onClick = onPickImages,
                modifier = Modifier.weight(1f),
            )
            SourcePickerTile(
                icon = Icons.Rounded.List,
                title = "文件夹",
                subtitle = "自然序批量",
                tint = MaterialTheme.colorScheme.secondary,
                containerColor = MaterialTheme.colorScheme.secondaryContainer,
                enabled = !isJobRunning,
                onClick = onPickFolder,
                modifier = Modifier.weight(1f),
            )
            SourcePickerTile(
                icon = Icons.Rounded.Menu,
                title = "PDF 合集",
                subtitle = "无损抽原图",
                tint = MaterialTheme.colorScheme.tertiary,
                containerColor = MaterialTheme.colorScheme.tertiaryContainer,
                enabled = !isJobRunning,
                onClick = onPickPdf,
                modifier = Modifier.weight(1f),
            )
        }

        // 2. Selected input preview card
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(20.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow),
            border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant),
        ) {
            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                if (isScanningInput) {
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                    Text(
                        text = "正在流式索引原图与切窗计划…",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                } else if (scanError != null) {
                    Text(
                        text = "读取失败：$scanError",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.error,
                    )
                } else when (inputSummary) {
                    null -> {
                        Text(
                            text = "尚未选择输入对象",
                            style = MaterialTheme.typography.titleSmall,
                            fontWeight = FontWeight.SemiBold,
                        )
                        Text(
                            text = "点击上方选择长图、图片目录或嵌入式 PDF；也可在 QQ / 相册 / 贴吧里直接「分享」给 MomoTranScript。",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    is BatchInputSummary.Images -> {
                        val totalWin = inputSummary.items.sumOf { it.plannedWindows }
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.SpaceBetween,
                        ) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    text = inputSummary.folderName?.let { "文件夹「$it」" } ?: "已选 ${inputSummary.items.size} 张图片",
                                    style = MaterialTheme.typography.titleSmall,
                                    fontWeight = FontWeight.SemiBold,
                                    color = MaterialTheme.colorScheme.primary,
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis,
                                )
                                Text(
                                    text = "共 ${inputSummary.items.size} 张 · 预计滑窗 $totalWin 窗",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                            if (!isJobRunning) {
                                IconButton(onClick = onClearInput) {
                                    Icon(Icons.Rounded.Close, contentDescription = "清除")
                                }
                            }
                        }
                        inputSummary.items.take(4).forEachIndexed { idx, item ->
                            Text(
                                text = "${idx + 1}. ${item.displayName} (${item.width}×${item.height})",
                                style = MaterialTheme.typography.bodySmall,
                                fontFamily = FontFamily.Monospace,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis,
                            )
                        }
                        if (inputSummary.items.size > 4) {
                            Text(
                                text = "… 其余 ${inputSummary.items.size - 4} 张已按自然序排好",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                    is BatchInputSummary.PdfDocument -> {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.SpaceBetween,
                        ) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    text = inputSummary.displayName,
                                    style = MaterialTheme.typography.titleSmall,
                                    fontWeight = FontWeight.SemiBold,
                                    color = MaterialTheme.colorScheme.primary,
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis,
                                )
                                Text(
                                    text = "${inputSummary.pageCount} 页 · 直提原图 ${inputSummary.directImageCount} 张 · 回退 ${inputSummary.fallbackPageCount} 页",
                                    style = MaterialTheme.typography.bodySmall,
                                )
                            }
                            if (!isJobRunning) {
                                IconButton(onClick = onClearInput) {
                                    Icon(Icons.Rounded.Close, contentDescription = "清除")
                                }
                            }
                        }
                        Text(
                            text = "首页 ${inputSummary.firstPageSize ?: "未知"} · 预计滑窗 ${inputSummary.estimatedTotalWindows} 窗（后台保活流式逐页处理）",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }
        }

        // 3. Clean options card
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(20.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow),
            border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant),
        ) {
            Column(modifier = Modifier.padding(horizontal = 16.dp, vertical = 10.dp)) {
                OptionToggleRow(
                    icon = Icons.Rounded.Build,
                    title = "MoeTalk 分支管理打包",
                    subtitle = "多图 / 多页生成根目录章节卡，导入不卡顿",
                    checked = useMoeTalkBranches,
                    enabled = !isJobRunning,
                    onCheckedChange = onToggleBranches,
                )
                HorizontalDivider(
                    modifier = Modifier.padding(vertical = 6.dp),
                    color = MaterialTheme.colorScheme.outlineVariant,
                )
                OptionToggleRow(
                    icon = Icons.Rounded.Face,
                    title = "视觉头像与差分匹配",
                    subtitle = if (visualPackReady) "外部视觉扩展包已就绪，自动匹配对应头像与差分索引"
                    else "未下载外部视觉扩展包（可选 +53 MB）",
                    checked = useCcipVisualMatch && visualPackReady,
                    enabled = !isJobRunning && visualPackReady,
                    onCheckedChange = onToggleCcip,
                )
                HorizontalDivider(
                    modifier = Modifier.padding(vertical = 6.dp),
                    color = MaterialTheme.colorScheme.outlineVariant,
                )
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 4.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    val backendDesc = when (backendMode) {
                        OnnxTranscriptionEngine.BACKEND_AUTO_GPU_CPU -> "det / CCIP 走 OpenCL GPU，动态宽 OCR 走 ARMv9 FP16 CPU"
                        OnnxTranscriptionEngine.BACKEND_OPENCL_ALL -> "全部 4 模型走 OpenCL GPU（首次会编译并缓存 Shader）"
                        OnnxTranscriptionEngine.BACKEND_VULKAN_ALL -> "全部 4 模型走 Vulkan Compute GPU（不支持时自动回退 CPU）"
                        else -> "全部 4 模型走 MNN ARMv9 FP16 CPU（4 线程基准模式）"
                    }
                    Column {
                        Text(
                            text = "MNN 3.6 推理后端",
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.Medium,
                        )
                        Text(
                            text = backendDesc,
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    val backendOptions = listOf(
                        OnnxTranscriptionEngine.BACKEND_AUTO_GPU_CPU to "自动",
                        OnnxTranscriptionEngine.BACKEND_OPENCL_ALL to "OpenCL",
                        OnnxTranscriptionEngine.BACKEND_VULKAN_ALL to "Vulkan",
                        OnnxTranscriptionEngine.BACKEND_CPU_FP16 to "CPU",
                    )
                    SingleChoiceSegmentedButtonRow(modifier = Modifier.fillMaxWidth()) {
                        backendOptions.forEachIndexed { idx, (modeVal, label) ->
                            SegmentedButton(
                                selected = backendMode == modeVal,
                                onClick = { onSelectBackendMode(modeVal) },
                                enabled = !isJobRunning,
                                shape = SegmentedButtonDefaults.itemShape(index = idx, count = backendOptions.size),
                                icon = {},
                            ) {
                                Text(label, maxLines = 1, softWrap = false, style = MaterialTheme.typography.labelMedium)
                            }
                        }
                    }
                }
            }
        }

        // 4. Live progress card (shown when running or recently errored)
        if (isJobRunning && batchProgress != null) {
            val prog = batchProgress
            val frac = if (prog.totalItems > 0) (prog.completedItems.toFloat() / prog.totalItems).coerceIn(0f, 1f) else null
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.45f)),
            ) {
                Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(
                            text = "正在离线转录 · ${prog.currentLabel}",
                            style = MaterialTheme.typography.titleSmall,
                            fontWeight = FontWeight.SemiBold,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            modifier = Modifier.weight(1f),
                        )
                        if (prog.totalItems > 0) {
                            Text(
                                text = "${prog.completedItems}/${prog.totalItems}",
                                style = MaterialTheme.typography.labelMedium,
                                fontFamily = FontFamily.Monospace,
                                color = MaterialTheme.colorScheme.primary,
                            )
                        }
                    }
                    if (frac != null) {
                        LinearProgressIndicator(progress = { frac }, modifier = Modifier.fillMaxWidth())
                    } else {
                        LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                    }
                    Text(
                        text = prog.detail,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    if (liveCompletedCount > 0) {
                        FilledTonalButton(
                            onClick = onPreviewPartialResult,
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(12.dp),
                            contentPadding = PaddingValues(horizontal = 12.dp, vertical = 8.dp),
                        ) {
                            Icon(Icons.Rounded.Search, contentDescription = null, modifier = Modifier.size(16.dp))
                            Spacer(Modifier.width(6.dp))
                            Text("即时预览 / 导出已完成 $liveCompletedCount 页", maxLines = 1, softWrap = false)
                        }
                    }
                }
            }
        }

        // 4b. Live Debug & Profiling Console Card (always visible once logs exist)
        if (liveLogText.isNotEmpty() || liveProfileSummary.isNotEmpty()) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant),
            ) {
                Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable { expandLogs = !expandLogs },
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = "运行日志与性能 Profile",
                                style = MaterialTheme.typography.titleSmall,
                                fontWeight = FontWeight.SemiBold,
                            )
                            if (liveProfileSummary.isNotEmpty()) {
                                Text(
                                    text = liveProfileSummary,
                                    style = MaterialTheme.typography.labelSmall,
                                    fontFamily = FontFamily.Monospace,
                                    color = MaterialTheme.colorScheme.secondary,
                                    maxLines = 2,
                                    overflow = TextOverflow.Ellipsis,
                                )
                            }
                        }
                        Icon(
                            imageVector = if (expandLogs) Icons.Rounded.KeyboardArrowUp else Icons.Rounded.KeyboardArrowDown,
                            contentDescription = "展开/收起日志",
                        )
                    }

                    AnimatedVisibility(visible = expandLogs) {
                        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            val logScroll = rememberScrollState()
                            LaunchedEffect(liveLogText.length) {
                                logScroll.scrollTo(logScroll.maxValue)
                            }
                            Surface(
                                color = Color(0xFF161922),
                                shape = RoundedCornerShape(12.dp),
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(250.dp),
                            ) {
                                Text(
                                    text = liveLogText.ifEmpty { "等待引擎输出…" },
                                    color = Color(0xFFE2E8F0),
                                    fontFamily = FontFamily.Monospace,
                                    fontSize = 11.sp,
                                    lineHeight = 15.sp,
                                    modifier = Modifier
                                        .fillMaxSize()
                                        .verticalScroll(logScroll)
                                        .padding(10.dp),
                                )
                            }
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                            ) {
                                OutlinedButton(
                                    onClick = onCopyLog,
                                    modifier = Modifier.weight(1f),
                                    shape = RoundedCornerShape(10.dp),
                                    contentPadding = PaddingValues(horizontal = 10.dp, vertical = 6.dp),
                                ) {
                                    Icon(Icons.Rounded.Share, contentDescription = null, modifier = Modifier.size(15.dp))
                                    Spacer(Modifier.width(4.dp))
                                    Text("复制日志", maxLines = 1, softWrap = false)
                                }
                                OutlinedButton(
                                    onClick = onExportLog,
                                    modifier = Modifier.weight(1f),
                                    shape = RoundedCornerShape(10.dp),
                                    contentPadding = PaddingValues(horizontal = 10.dp, vertical = 6.dp),
                                ) {
                                    Icon(Icons.Rounded.Info, contentDescription = null, modifier = Modifier.size(15.dp))
                                    Spacer(Modifier.width(4.dp))
                                    Text("导出 .log", maxLines = 1, softWrap = false)
                                }
                            }
                        }
                    }
                }
            }
        }

        runError?.let { err ->
            Surface(
                color = MaterialTheme.colorScheme.errorContainer,
                shape = RoundedCornerShape(14.dp),
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(
                    text = err,
                    color = MaterialTheme.colorScheme.onErrorContainer,
                    style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.padding(12.dp),
                )
            }
        }

        // 5. Prominent full-width primary CTA
        if (!corePackReady) {
            FilledTonalButton(
                onClick = onGoToModels,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(52.dp),
                shape = RoundedCornerShape(16.dp),
            ) {
                Icon(Icons.Rounded.Add, contentDescription = null)
                Spacer(Modifier.width(8.dp))
                Text("前往下载基础模型包（约 43 MiB）", fontWeight = FontWeight.SemiBold)
            }
        } else if (isJobRunning) {
            OutlinedButton(
                onClick = onStopTranscription,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(52.dp),
                shape = RoundedCornerShape(16.dp),
                colors = ButtonDefaults.outlinedButtonColors(contentColor = MaterialTheme.colorScheme.error),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.error),
            ) {
                Icon(Icons.Rounded.Close, contentDescription = null)
                Spacer(Modifier.width(8.dp))
                Text("停止转录任务", fontWeight = FontWeight.SemiBold)
            }
        } else {
            Button(
                onClick = onStartTranscription,
                enabled = inputSummary != null && !isScanningInput,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(52.dp),
                shape = RoundedCornerShape(16.dp),
            ) {
                Icon(Icons.Rounded.PlayArrow, contentDescription = null)
                Spacer(Modifier.width(8.dp))
                Text("开始离线转录", fontWeight = FontWeight.SemiBold)
            }
        }

        Spacer(Modifier.height(12.dp))
    }
}

@OptIn(ExperimentalLayoutApi::class, ExperimentalMaterial3Api::class)
@Composable
private fun ResultTab(
    scriptText: String,
    onScriptChange: (String) -> Unit,
    isJobRunning: Boolean,
    hasPendingNewerText: Boolean,
    onSyncLatestText: () -> Unit,
    lastCompleted: CompletedJobSummary?,
    previewPages: List<PagePreviewRecord>,
    onCopy: () -> Unit,
    onExportTxt: () -> Unit,
    onExportMoeTalk: () -> Unit,
) {
    var resultMode by remember { mutableIntStateOf(0) } // 0=正文编辑, 1=打框对照

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        SingleChoiceSegmentedButtonRow(modifier = Modifier.fillMaxWidth()) {
            SegmentedButton(
                selected = resultMode == 0,
                onClick = { resultMode = 0 },
                shape = SegmentedButtonDefaults.itemShape(index = 0, count = 2),
                icon = { Icon(Icons.Rounded.Edit, contentDescription = null, modifier = Modifier.size(16.dp)) },
            ) {
                Text("正文编辑")
            }
            SegmentedButton(
                selected = resultMode == 1,
                onClick = { resultMode = 1 },
                shape = SegmentedButtonDefaults.itemShape(index = 1, count = 2),
                icon = { Icon(Icons.Rounded.Search, contentDescription = null, modifier = Modifier.size(16.dp)) },
            ) {
                val boxCount = previewPages.sumOf { it.elements.size }
                Text(if (boxCount > 0) "打框对照 ($boxCount)" else "打框对照")
            }
        }

        if (isJobRunning || lastCompleted?.isPartial == true) {
            val done = lastCompleted?.itemCount ?: 0
            val total = lastCompleted?.totalItems ?: 0
            Surface(
                color = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.7f),
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier.fillMaxWidth(),
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 12.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Text(
                        text = if (isJobRunning) {
                            if (done > 0) "转录进行中 · 已同步 $done/$total 页，可直接预览或导出"
                            else "转录进行中 · 首页完成后将自动显示结果"
                        } else {
                            "部分完成 · 已保留前 $done/$total 页，可直接预览或导出"
                        },
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSecondaryContainer,
                        modifier = Modifier.weight(1f),
                    )
                    if (hasPendingNewerText) {
                        FilledTonalButton(
                            onClick = onSyncLatestText,
                            contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp),
                            shape = RoundedCornerShape(10.dp),
                        ) {
                            Icon(Icons.Rounded.Refresh, contentDescription = null, modifier = Modifier.size(14.dp))
                            Spacer(Modifier.width(4.dp))
                            Text("同步最新", maxLines = 1, softWrap = false)
                        }
                    }
                }
            }
        }

        lastCompleted?.let { out ->
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                val countLabel = if (out.isPartial && out.totalItems > out.itemCount) {
                    "${out.itemCount}/${out.totalItems} 页 · ${out.lineCount} 行"
                } else {
                    "${out.itemCount} 项 · ${out.lineCount} 行"
                }
                StatChip(countLabel)
                StatChip("${"%.1f".format(out.elapsedMs / 1000.0)}s")
                StatChip(if (out.usedBranches) "MoeTalk 分支打包" else "单流合并")
                if (out.matchedAvatars > 0 || out.matchedVisuals > 0) {
                    StatChip("头像 ${out.matchedAvatars} · 差分 ${out.matchedVisuals}")
                }
            }
        }

        if (resultMode == 0) {
            val compactPadding = PaddingValues(horizontal = 10.dp, vertical = 8.dp)
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                OutlinedButton(
                    onClick = onCopy,
                    modifier = Modifier.weight(1f),
                    shape = RoundedCornerShape(12.dp),
                    contentPadding = compactPadding,
                ) {
                    Icon(Icons.Rounded.Share, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(Modifier.width(4.dp))
                    Text("复制", maxLines = 1, softWrap = false)
                }
                OutlinedButton(
                    onClick = onExportTxt,
                    modifier = Modifier.weight(1f),
                    shape = RoundedCornerShape(12.dp),
                    contentPadding = compactPadding,
                ) {
                    Icon(Icons.Rounded.Info, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(Modifier.width(4.dp))
                    Text("导出 .txt", maxLines = 1, softWrap = false)
                }
                Button(
                    onClick = onExportMoeTalk,
                    enabled = lastCompleted != null,
                    modifier = Modifier.weight(1.25f),
                    shape = RoundedCornerShape(12.dp),
                    contentPadding = compactPadding,
                ) {
                    Icon(Icons.Rounded.CheckCircle, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(Modifier.width(4.dp))
                    Text("MoeTalk", maxLines = 1, softWrap = false)
                }
            }

            OutlinedTextField(
                value = scriptText,
                onValueChange = onScriptChange,
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
                shape = RoundedCornerShape(18.dp),
                textStyle = MaterialTheme.typography.bodyMedium.copy(fontFamily = FontFamily.Monospace),
            )
        } else {
            DetectionOverlayViewer(
                pages = previewPages,
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
            )
        }
    }
}

@Composable
private fun SourcePickerTile(
    icon: ImageVector,
    title: String,
    subtitle: String,
    tint: Color,
    containerColor: Color,
    enabled: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Card(
        modifier = modifier
            .clip(RoundedCornerShape(18.dp))
            .clickable(enabled = enabled, onClick = onClick),
        shape = RoundedCornerShape(18.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant),
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 16.dp, horizontal = 8.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Box(
                modifier = Modifier
                    .size(44.dp)
                    .clip(CircleShape)
                    .background(containerColor),
                contentAlignment = Alignment.Center,
            ) {
                Icon(imageVector = icon, contentDescription = title, tint = tint, modifier = Modifier.size(24.dp))
            }
            Text(text = title, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold, maxLines = 1)
            Text(
                text = subtitle,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 1,
            )
        }
    }
}

@Composable
private fun OptionToggleRow(
    icon: ImageVector,
    title: String,
    subtitle: String,
    checked: Boolean,
    enabled: Boolean,
    onCheckedChange: (Boolean) -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = if (enabled) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.size(22.dp),
        )
        Column(modifier = Modifier.weight(1f)) {
            Text(text = title, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Medium)
            Text(
                text = subtitle,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Switch(
            checked = checked,
            onCheckedChange = onCheckedChange,
            enabled = enabled,
        )
    }
}

@Composable
private fun StatChip(label: String) {
    Surface(
        color = MaterialTheme.colorScheme.secondaryContainer,
        shape = RoundedCornerShape(999.dp),
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSecondaryContainer,
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
        )
    }
}

@Composable
private fun PrivacyConsentScreen(
    onOpenUrl: (String) -> Unit,
    onAccept: () -> Unit,
    onDecline: () -> Unit,
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .padding(24.dp),
        contentAlignment = Alignment.Center,
    ) {
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(24.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow),
            border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant),
        ) {
            Column(
                modifier = Modifier
                    .padding(22.dp)
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        modifier = Modifier
                            .size(34.dp)
                            .clip(RoundedCornerShape(10.dp))
                            .background(MaterialTheme.colorScheme.primary),
                        contentAlignment = Alignment.Center,
                    ) {
                        Text(
                            text = ">_",
                            color = MaterialTheme.colorScheme.onPrimary,
                            fontFamily = FontFamily.Monospace,
                            fontWeight = FontWeight.Bold,
                            style = MaterialTheme.typography.labelMedium,
                        )
                    }
                    Spacer(Modifier.width(10.dp))
                    Column {
                        Text(
                            text = "用户协议与隐私保护指引",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                        )
                        Text(
                            text = "MomoTranScript v${BuildConfig.VERSION_NAME}",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }

                Text(
                    text = "欢迎使用 MomoTranScript！在使用本应用前，请您仔细阅读并了解我们的隐私保护承诺：",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurface,
                )

                Surface(
                    color = MaterialTheme.colorScheme.surfaceContainer,
                    shape = RoundedCornerShape(14.dp),
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Column(
                        modifier = Modifier.padding(12.dp),
                        verticalArrangement = Arrangement.spacedBy(6.dp),
                    ) {
                        Text(
                            text = "• 纯端侧离线处理：您选择的图片、文件夹与 PDF 均在手机本地完成版面分析与 OCR 识别，绝不上传至任何服务器。",
                            style = MaterialTheme.typography.bodySmall,
                        )
                        Text(
                            text = "• 最小权限原则：通过系统文件选择器读取您主动授权的文件，不申请全局存储、通讯录、位置或设备标识符权限。",
                            style = MaterialTheme.typography.bodySmall,
                        )
                        Text(
                            text = "• 网络仅用于下载模型：在您点击「同意」前不会发起任何网络请求；后续联网仅用于您主动从配置源下载离线识别模型包。",
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                }

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    OutlinedButton(
                        onClick = { onOpenUrl(BuildConfig.PRIVACY_POLICY_URL) },
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(12.dp),
                        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 6.dp),
                    ) {
                        Text("阅读《隐私政策》", style = MaterialTheme.typography.labelMedium, maxLines = 1)
                    }
                    OutlinedButton(
                        onClick = { onOpenUrl(BuildConfig.USER_AGREEMENT_URL) },
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(12.dp),
                        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 6.dp),
                    ) {
                        Text("阅读《用户协议》", style = MaterialTheme.typography.labelMedium, maxLines = 1)
                    }
                }

                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    OutlinedButton(
                        onClick = onDecline,
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(14.dp),
                        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 10.dp),
                    ) {
                        Text("不同意并退出", maxLines = 1, softWrap = false)
                    }
                    Button(
                        onClick = onAccept,
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(14.dp),
                        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 10.dp),
                    ) {
                        Text("同意并继续", fontWeight = FontWeight.SemiBold, maxLines = 1, softWrap = false)
                    }
                }
            }
        }
    }
}
