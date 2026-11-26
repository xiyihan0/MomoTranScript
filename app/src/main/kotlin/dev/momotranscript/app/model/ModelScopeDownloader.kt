package dev.momotranscript.app.model

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URI
import java.net.URL
import java.net.URLEncoder
import java.nio.charset.StandardCharsets
import java.security.MessageDigest

data class ModelDownloadProgress(
    val packId: String,
    val fileIndex: Int,
    val fileCount: Int,
    val fileName: String,
    val downloadedBytes: Long,
    val totalBytes: Long,
    val speedBytesPerSec: Long,
)

data class ModelScopeConfig(
    val repoId: String = DEFAULT_REPO_ID,
    val revision: String = DEFAULT_REVISION,
    val pathPrefix: String = "",
    val accessToken: String = "",
) {
    companion object {
        const val DEFAULT_REPO_ID = "xiyihan/momotranscript-models"
        const val DEFAULT_REVISION = "master"
    }
}

/**
 * Direct HTTP downloader for ModelScope model repositories:
 * `https://www.modelscope.cn/models/{owner}/{repo}/resolve/{revision}/{path}`
 *
 * Features:
 * - Supports public repos (no token) and private repos (optional Bearer token sent ONLY to
 *   `modelscope.cn` hosts, stripped on cross-domain CDN redirects).
 * - Automatically fetches optional `bundle_manifest.json` from the repo to verify SHA-256
 *   checksums before atomically promoting `.part` files into place.
 * - Fully cancellable via Kotlin coroutines.
 */
object ModelScopeDownloader {

    fun buildResolveUrl(config: ModelScopeConfig, relativePath: String): String {
        val repo = config.repoId.trim().trim('/')
        val rev = config.revision.trim().ifEmpty { ModelScopeConfig.DEFAULT_REVISION }
        val prefix = config.pathPrefix.trim().trim('/')
        val fullRel = if (prefix.isEmpty()) relativePath.trimStart('/') else "$prefix/${relativePath.trimStart('/')}"
        val encodedPath = fullRel.split('/').joinToString("/") { seg ->
            URLEncoder.encode(seg, StandardCharsets.UTF_8.name()).replace("+", "%20")
        }
        return "https://www.modelscope.cn/models/$repo/resolve/$rev/$encodedPath"
    }

    suspend fun downloadPack(
        pack: ModelPackSpec,
        bundleRoot: File,
        config: ModelScopeConfig,
        onProgress: (ModelDownloadProgress) -> Unit,
    ) = withContext(Dispatchers.IO) {
        val expectedHashes = fetchManifestHashes(config)
        val totalFiles = pack.files.size

        for ((idx, fileSpec) in pack.files.withIndex()) {
            currentCoroutineContext().ensureActive()
            val targetFile = File(bundleRoot, fileSpec.relativePath)
            targetFile.parentFile?.mkdirs()

            val expectedSha = expectedHashes[fileSpec.relativePath]
            if (targetFile.isFile && targetFile.length() > 0L) {
                if (expectedSha == null || computeSha256(targetFile).equals(expectedSha, ignoreCase = true)) {
                    onProgress(
                        ModelDownloadProgress(
                            packId = pack.id,
                            fileIndex = idx + 1,
                            fileCount = totalFiles,
                            fileName = fileSpec.relativePath.substringAfterLast('/'),
                            downloadedBytes = targetFile.length(),
                            totalBytes = targetFile.length(),
                            speedBytesPerSec = 0L,
                        ),
                    )
                    continue
                }
            }

            val urlStr = buildResolveUrl(config, fileSpec.relativePath)
            val partFile = File(targetFile.parentFile, "${targetFile.name}.part")
            val shortName = fileSpec.relativePath.substringAfterLast('/')

            try {
                val actualSha = downloadFileWithSha256(
                    urlStr = urlStr,
                    token = config.accessToken.trim(),
                    dstPart = partFile,
                ) { done, total, speed ->
                    onProgress(
                        ModelDownloadProgress(
                            packId = pack.id,
                            fileIndex = idx + 1,
                            fileCount = totalFiles,
                            fileName = shortName,
                            downloadedBytes = done,
                            totalBytes = total,
                            speedBytesPerSec = speed,
                        ),
                    )
                }

                if (expectedSha != null && !actualSha.equals(expectedSha, ignoreCase = true)) {
                    partFile.delete()
                    throw IOException("$shortName SHA-256 校验失败，请重试下载")
                }

                if (targetFile.exists()) targetFile.delete()
                if (!partFile.renameTo(targetFile)) {
                    partFile.copyTo(targetFile, overwrite = true)
                    partFile.delete()
                }
                for (legacy in fileSpec.legacyFallbackPaths) {
                    val lf = File(bundleRoot, legacy)
                    if (lf.isFile) lf.delete()
                }
            } catch (e: Exception) {
                partFile.delete()
                throw e
            }
        }
    }

    fun deletePack(pack: ModelPackSpec, bundleRoot: File): Int {
        var deleted = 0
        for (f in pack.files) {
            val allPaths = listOf(f.relativePath) + f.legacyFallbackPaths
            for (rel in allPaths) {
                val target = File(bundleRoot, rel)
                if (target.isFile && target.delete()) {
                    deleted++
                }
                File(target.parentFile, "${target.name}.part").delete()
            }
        }
        return deleted
    }

    private fun fetchManifestHashes(config: ModelScopeConfig): Map<String, String> {
        val out = HashMap<String, String>()
        for (manifestName in listOf("bundle_manifest.json", "manifest.json")) {
            runCatching {
                val url = buildResolveUrl(config, manifestName)
                val conn = openConnectionFollowingRedirects(url, config.accessToken.trim())
                try {
                    if (conn.responseCode !in 200..299) return@runCatching
                    val text = conn.inputStream.bufferedReader(StandardCharsets.UTF_8).use { it.readText() }
                    val root = Json.parseToJsonElement(text).jsonObject
                    val filesObj = root["files"]?.jsonObject ?: return@runCatching
                    for ((rel, elem) in filesObj) {
                        val sha = elem.jsonObject["sha256"]?.jsonPrimitive?.content
                        if (!sha.isNullOrBlank()) out[rel] = sha
                    }
                } finally {
                    conn.disconnect()
                }
            }
        }
        return out
    }

    private suspend fun downloadFileWithSha256(
        urlStr: String,
        token: String,
        dstPart: File,
        onBytes: (downloaded: Long, total: Long, speedBps: Long) -> Unit,
    ): String {
        val conn = openConnectionFollowingRedirects(urlStr, token)
        try {
            val code = conn.responseCode
            if (code !in 200..299) {
                val hint = when (code) {
                    401, 403 -> "无权限访问该 ModelScope 仓库（私有库请填写 Access Token）"
                    404 -> "仓库中未找到文件（请确认仓库 ID 与目录结构）"
                    else -> "HTTP $code"
                }
                throw IOException("下载失败：$hint ($urlStr)")
            }

            val total = conn.contentLengthLong.coerceAtLeast(-1L)
            val digest = MessageDigest.getInstance("SHA-256")
            val buf = ByteArray(64 * 1024)
            var downloaded = 0L
            val tStart = System.nanoTime()
            var lastEmit = 0L

            conn.inputStream.use { input ->
                dstPart.outputStream().use { output ->
                    while (true) {
                        currentCoroutineContext().ensureActive()
                        val n = input.read(buf)
                        if (n <= 0) break
                        output.write(buf, 0, n)
                        digest.update(buf, 0, n)
                        downloaded += n

                        val now = System.nanoTime()
                        if (now - lastEmit >= 120_000_000L) {
                            lastEmit = now
                            val elapsedSec = ((now - tStart) / 1e9).coerceAtLeast(0.05)
                            onBytes(downloaded, total, (downloaded / elapsedSec).toLong())
                        }
                    }
                }
            }
            val elapsedSec = ((System.nanoTime() - tStart) / 1e9).coerceAtLeast(0.05)
            onBytes(downloaded, total.coerceAtLeast(downloaded), (downloaded / elapsedSec).toLong())
            return digest.digest().joinToString("") { "%02x".format(it) }
        } finally {
            conn.disconnect()
        }
    }

    private fun openConnectionFollowingRedirects(initialUrl: String, token: String): HttpURLConnection {
        var current = initialUrl
        var redirects = 0
        while (redirects < 8) {
            val uri = URI(current)
            val conn = (URL(current).openConnection() as HttpURLConnection).apply {
                instanceFollowRedirects = false
                connectTimeout = 20_000
                readTimeout = 60_000
                requestMethod = "GET"
                setRequestProperty("User-Agent", "MomoTranScript-Android/0.1")
                if (token.isNotEmpty() && uri.host?.endsWith("modelscope.cn") == true) {
                    setRequestProperty("Authorization", "Bearer $token")
                }
            }
            val code = conn.responseCode
            if (code in listOf(301, 302, 303, 307, 308)) {
                val loc = conn.getHeaderField("Location")
                conn.disconnect()
                if (loc.isNullOrBlank()) throw IOException("重定向缺少 Location 头")
                current = uri.resolve(loc).toString()
                redirects++
                continue
            }
            return conn
        }
        throw IOException("重定向次数过多")
    }

    private fun computeSha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val buf = ByteArray(64 * 1024)
        file.inputStream().use { input ->
            while (true) {
                val n = input.read(buf)
                if (n <= 0) break
                digest.update(buf, 0, n)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
