package dev.momotranscript.app.model

import org.junit.Assert.assertEquals
import org.junit.Test

class ModelScopeDownloaderTest {

    @Test
    fun buildsCleanModelScopeResolveUrls() {
        val cfg = ModelScopeConfig(
            repoId = "xiyihan/momotranscript-models",
            revision = "master",
            pathPrefix = "",
        )
        assertEquals(
            "https://www.modelscope.cn/models/xiyihan/momotranscript-models/resolve/master/models/rec_small_fp16_mmw8.onnx",
            ModelScopeDownloader.buildResolveUrl(cfg, "models/rec_small_fp16_mmw8.onnx"),
        )

        val prefixed = cfg.copy(repoId = "/xiyihan/wenshuku/", pathPrefix = "mobile/20261125/")
        assertEquals(
            "https://www.modelscope.cn/models/xiyihan/wenshuku/resolve/master/mobile/20261125/avatar/catalog.json.zst",
            ModelScopeDownloader.buildResolveUrl(prefixed, "avatar/catalog.json.zst"),
        )
    }
}
