package dev.momotranscript.app.ui

import android.content.Intent
import android.net.Uri
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
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
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Add
import androidx.compose.material.icons.rounded.CheckCircle
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.Delete
import androidx.compose.material.icons.rounded.KeyboardArrowDown
import androidx.compose.material.icons.rounded.KeyboardArrowUp
import androidx.compose.material.icons.rounded.List
import androidx.compose.material.icons.rounded.Refresh
import androidx.compose.material.icons.rounded.Settings
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import dev.momotranscript.app.BuildConfig
import dev.momotranscript.app.model.ModelDownloadProgress
import dev.momotranscript.app.model.ModelPackSpec
import dev.momotranscript.app.model.ModelPackStatus
import dev.momotranscript.app.model.ModelScopeConfig

@Composable
fun ModelsTab(
    packStatuses: List<ModelPackStatus>,
    config: ModelScopeConfig,
    onSaveConfig: (ModelScopeConfig) -> Unit,
    downloadingPackId: String?,
    downloadProgress: ModelDownloadProgress?,
    downloadError: String?,
    isImportingLocal: Boolean,
    isJobRunning: Boolean,
    packInspectNote: String,
    coreVersion: String,
    onDownloadPack: (ModelPackSpec) -> Unit,
    onCancelDownload: () -> Unit,
    onDeletePack: (ModelPackSpec) -> Unit,
    onImportLocalFolder: () -> Unit,
    onRefresh: () -> Unit,
) {
    val context = LocalContext.current
    fun openExternalUrl(url: String) {
        runCatching {
            context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url)))
        }
    }

    var expandRepoConfig by remember { mutableStateOf(false) }
    var repoInput by remember(config.repoId) { mutableStateOf(config.repoId) }
    var revInput by remember(config.revision) { mutableStateOf(config.revision) }
    var prefixInput by remember(config.pathPrefix) { mutableStateOf(config.pathPrefix) }
    var tokenInput by remember(config.accessToken) { mutableStateOf(config.accessToken) }

    val installedMiB = packStatuses.sumOf { it.installedBytes }.toDouble() / (1024.0 * 1024.0)
    val readyCount = packStatuses.count { it.isReady }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        // 1. ModelScope repository config card
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(20.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow),
            border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant),
        ) {
            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.weight(1f),
                    ) {
                        Icon(
                            imageVector = Icons.Rounded.Settings,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.size(22.dp),
                        )
                        Spacer(Modifier.width(8.dp))
                        Column {
                            Text("ModelScope 在线源", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
                            Text(
                                text = "${config.repoId} (${config.revision})",
                                style = MaterialTheme.typography.bodySmall,
                                fontFamily = FontFamily.Monospace,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis,
                            )
                        }
                    }
                    IconButton(onClick = { expandRepoConfig = !expandRepoConfig }) {
                        Icon(
                            imageVector = if (expandRepoConfig) Icons.Rounded.KeyboardArrowUp else Icons.Rounded.KeyboardArrowDown,
                            contentDescription = "展开配置",
                        )
                    }
                }

                AnimatedVisibility(visible = expandRepoConfig) {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                        OutlinedTextField(
                            value = repoInput,
                            onValueChange = { repoInput = it },
                            label = { Text("ModelScope 仓库 ID") },
                            placeholder = { Text("例如 xiyihan/momotranscript-models") },
                            singleLine = true,
                            modifier = Modifier.fillMaxWidth(),
                        )
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            OutlinedTextField(
                                value = revInput,
                                onValueChange = { revInput = it },
                                label = { Text("分支 / Tag") },
                                singleLine = true,
                                modifier = Modifier.weight(1f),
                            )
                            OutlinedTextField(
                                value = prefixInput,
                                onValueChange = { prefixInput = it },
                                label = { Text("子目录（可选）") },
                                placeholder = { Text("如 20261125") },
                                singleLine = true,
                                modifier = Modifier.weight(1f),
                            )
                        }
                        OutlinedTextField(
                            value = tokenInput,
                            onValueChange = { tokenInput = it },
                            label = { Text("Access Token（私有库才需填写）") },
                            singleLine = true,
                            visualTransformation = PasswordVisualTransformation(),
                            modifier = Modifier.fillMaxWidth(),
                        )
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.End,
                        ) {
                            FilledTonalButton(
                                onClick = {
                                    onSaveConfig(
                                        ModelScopeConfig(
                                            repoId = repoInput.trim().ifEmpty { ModelScopeConfig.DEFAULT_REPO_ID },
                                            revision = revInput.trim().ifEmpty { ModelScopeConfig.DEFAULT_REVISION },
                                            pathPrefix = prefixInput.trim(),
                                            accessToken = tokenInput.trim(),
                                        ),
                                    )
                                    expandRepoConfig = false
                                },
                            ) {
                                Icon(Icons.Rounded.Settings, contentDescription = null, modifier = Modifier.size(16.dp))
                                Spacer(Modifier.width(6.dp))
                                Text("保存源设置")
                            }
                        }
                    }
                }

                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)

                // Clean equal-width local import & refresh bar (never overflows!)
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    OutlinedButton(
                        onClick = onImportLocalFolder,
                        enabled = !isImportingLocal && downloadingPackId == null && !isJobRunning,
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(12.dp),
                    ) {
                        Icon(Icons.Rounded.List, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(6.dp))
                        Text(if (isImportingLocal) "导入中…" else "本地目录导入", maxLines = 1)
                    }
                    OutlinedButton(
                        onClick = onRefresh,
                        enabled = downloadingPackId == null,
                        shape = RoundedCornerShape(12.dp),
                    ) {
                        Icon(Icons.Rounded.Refresh, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(4.dp))
                        Text("刷新", maxLines = 1)
                    }
                }

                Text(
                    text = "已就绪 $readyCount/${packStatuses.size} 个模块 · 占用 ${"%.1f".format(installedMiB)} MiB",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        if (isImportingLocal) {
            LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
        }

        downloadError?.let { err ->
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

        // 2. Modular pack cards
        packStatuses.forEach { st ->
            val isThisDownloading = downloadingPackId == st.spec.id
            val anyDownloading = downloadingPackId != null

            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow),
                border = BorderStroke(
                    width = 1.dp,
                    color = if (st.isReady) MaterialTheme.colorScheme.secondary.copy(alpha = 0.35f)
                    else MaterialTheme.colorScheme.outlineVariant,
                ),
            ) {
                Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween,
                    ) {
                        Text(
                            text = st.spec.title,
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold,
                            modifier = Modifier.weight(1f),
                        )
                        StatusBadge(
                            ready = st.isReady,
                            text = if (st.isReady) "已就绪" else "${st.installedFiles}/${st.totalFiles}",
                        )
                    }

                    Text(
                        text = "${st.spec.subtitle} · 约 ${"%.1f".format(st.spec.approxTotalMiB)} MiB",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )

                    if (isThisDownloading && downloadProgress != null) {
                        val prog = downloadProgress
                        val frac = if (prog.totalBytes > 0L) {
                            (prog.downloadedBytes.toFloat() / prog.totalBytes.toFloat()).coerceIn(0f, 1f)
                        } else {
                            null
                        }
                        if (frac != null) {
                            LinearProgressIndicator(progress = { frac }, modifier = Modifier.fillMaxWidth())
                        } else {
                            LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                        }
                        val doneMb = prog.downloadedBytes / (1024.0 * 1024.0)
                        val totalMb = prog.totalBytes / (1024.0 * 1024.0)
                        val spdMb = prog.speedBytesPerSec / (1024.0 * 1024.0)
                        val sizeText = if (prog.totalBytes > 0) {
                            "${"%.1f".format(doneMb)}/${"%.1f".format(totalMb)} MB"
                        } else {
                            "${"%.1f".format(doneMb)} MB"
                        }
                        Text(
                            text = "[${prog.fileIndex}/${prog.fileCount}] ${prog.fileName} · $sizeText · ${"%.1f".format(spdMb)} MB/s",
                            style = MaterialTheme.typography.labelSmall,
                            fontFamily = FontFamily.Monospace,
                            color = MaterialTheme.colorScheme.primary,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                        )
                    }

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        if (isThisDownloading) {
                            OutlinedButton(
                                onClick = onCancelDownload,
                                modifier = Modifier.weight(1f),
                                shape = RoundedCornerShape(12.dp),
                            ) {
                                Icon(Icons.Rounded.Close, contentDescription = null, modifier = Modifier.size(18.dp))
                                Spacer(Modifier.width(6.dp))
                                Text("取消下载")
                            }
                        } else if (!st.isReady) {
                            Button(
                                onClick = { onDownloadPack(st.spec) },
                                enabled = !anyDownloading && !isJobRunning,
                                modifier = Modifier.weight(1f),
                                shape = RoundedCornerShape(12.dp),
                            ) {
                                Icon(Icons.Rounded.Add, contentDescription = null, modifier = Modifier.size(18.dp))
                                Spacer(Modifier.width(6.dp))
                                Text("从 ModelScope 下载")
                            }
                        } else {
                            FilledTonalButton(
                                onClick = { onDownloadPack(st.spec) },
                                enabled = !anyDownloading && !isJobRunning,
                                modifier = Modifier.weight(1f),
                                shape = RoundedCornerShape(12.dp),
                            ) {
                                Icon(Icons.Rounded.Refresh, contentDescription = null, modifier = Modifier.size(18.dp))
                                Spacer(Modifier.width(6.dp))
                                Text("校验 / 更新")
                            }
                            IconButton(
                                onClick = { onDeletePack(st.spec) },
                                enabled = !anyDownloading && !isJobRunning,
                            ) {
                                Icon(
                                    imageVector = Icons.Rounded.Delete,
                                    contentDescription = "删除此模型包",
                                    tint = MaterialTheme.colorScheme.error,
                                )
                            }
                        }
                    }
                }
            }
        }

        // 3. Subtle technical engine info footer
        Surface(
            color = MaterialTheme.colorScheme.surfaceContainer,
            shape = RoundedCornerShape(16.dp),
            modifier = Modifier.fillMaxWidth(),
        ) {
            Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(
                    text = "端侧引擎信息 · v${BuildConfig.VERSION_NAME} (build ${BuildConfig.VERSION_CODE})",
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.SemiBold,
                )
                Text(
                    text = coreVersion,
                    style = MaterialTheme.typography.labelSmall,
                    fontFamily = FontFamily.Monospace,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                if (packInspectNote.isNotEmpty()) {
                    Text(
                        text = "MMTAVIF1: $packInspectNote",
                        style = MaterialTheme.typography.labelSmall,
                        fontFamily = FontFamily.Monospace,
                        color = MaterialTheme.colorScheme.secondary,
                    )
                }
            }
        }

        // 4. Compliance, privacy, MIIT filing & open-source notice card
        Surface(
            color = MaterialTheme.colorScheme.surfaceContainer,
            shape = RoundedCornerShape(16.dp),
            modifier = Modifier.fillMaxWidth(),
        ) {
            Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(
                    text = "关于与合规信息",
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.SemiBold,
                )
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    OutlinedButton(
                        onClick = { openExternalUrl(BuildConfig.PRIVACY_POLICY_URL) },
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(10.dp),
                        contentPadding = PaddingValues(horizontal = 6.dp, vertical = 4.dp),
                    ) {
                        Text("隐私政策", style = MaterialTheme.typography.labelSmall, maxLines = 1)
                    }
                    OutlinedButton(
                        onClick = { openExternalUrl(BuildConfig.USER_AGREEMENT_URL) },
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(10.dp),
                        contentPadding = PaddingValues(horizontal = 6.dp, vertical = 4.dp),
                    ) {
                        Text("用户协议", style = MaterialTheme.typography.labelSmall, maxLines = 1)
                    }
                    OutlinedButton(
                        onClick = { openExternalUrl(BuildConfig.HOMEPAGE_URL) },
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(10.dp),
                        contentPadding = PaddingValues(horizontal = 6.dp, vertical = 4.dp),
                    ) {
                        Text("应用主页", style = MaterialTheme.typography.labelSmall, maxLines = 1)
                    }
                }
                val filingLabel = if (BuildConfig.APP_ICP_NUMBER.isNotBlank()) {
                    "工信部 App 备案号：${BuildConfig.APP_ICP_NUMBER}（点击查询）"
                } else {
                    "域名备案：${BuildConfig.SITE_ICP_NUMBER} · App 备案申请中（点击查验）"
                }
                Text(
                    text = filingLabel,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.clickable { openExternalUrl("https://beian.miit.gov.cn/") },
                )
                Text(
                    text = "开发者反馈邮箱：${BuildConfig.CONTACT_EMAIL}",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Text(
                    text = "开源组件致谢：Alibaba MNN (Apache-2.0) · AOMedia libavif (BSD-2) · Apache PDFBox-Android · zstd-jni",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
        Spacer(Modifier.height(12.dp))
    }
}

@Composable
private fun StatusBadge(ready: Boolean, text: String) {
    val bg = if (ready) Color(0xFFE6F6EC) else MaterialTheme.colorScheme.primaryContainer
    val fg = if (ready) Color(0xFF1B7A43) else MaterialTheme.colorScheme.onPrimaryContainer
    Surface(
        color = bg,
        shape = RoundedCornerShape(999.dp),
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            if (ready) {
                Icon(
                    imageVector = Icons.Rounded.CheckCircle,
                    contentDescription = null,
                    tint = fg,
                    modifier = Modifier.size(14.dp),
                )
            }
            Text(
                text = text,
                style = MaterialTheme.typography.labelSmall,
                fontWeight = FontWeight.SemiBold,
                color = fg,
            )
        }
    }
}
