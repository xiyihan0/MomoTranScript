package dev.momotranscript.app

import dev.momotranscript.app.model.ModelPackCatalog
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class ModelPackCatalogTest {
    @get:Rule
    val tempFolder = TemporaryFolder()

    @Test
    fun catalogContainsExpectedModularPacks() {
        val ids = ModelPackCatalog.packs.map { it.id }
        assertEquals(listOf("core_text", "rec_medium", "moetalk_avatar", "moetalk_cfid6"), ids)
        val corePack = ModelPackCatalog.packs.first { it.id == "core_text" }
        assertTrue(corePack.required)
        assertTrue(corePack.approxTotalMiB in 35.0..41.0)
        val medPack = ModelPackCatalog.packs.first { it.id == "rec_medium" }
        assertTrue(medPack.approxTotalMiB in 15.0..22.0)
        val avatarPack = ModelPackCatalog.packs.first { it.id == "moetalk_avatar" }
        assertTrue(avatarPack.approxTotalMiB in 24.0..32.0)
        val cfid6Pack = ModelPackCatalog.packs.first { it.id == "moetalk_cfid6" }
        assertFalse(cfid6Pack.required)
        assertTrue(cfid6Pack.approxTotalMiB in 4.0..7.0)
    }

    @Test
    fun inspectInstalledDetectsReadyCorePack() {
        val root = tempFolder.newFolder("bundles")
        val initial = ModelPackCatalog.inspectInstalled(root)
        assertFalse(initial.first { it.spec.id == "core_text" }.isReady)

        for (fileSpec in ModelPackCatalog.packs.first { it.id == "core_text" }.files) {
            val target = File(root, fileSpec.relativePath)
            target.parentFile?.mkdirs()
            target.writeBytes(byteArrayOf(1, 2, 3))
        }
        val after = ModelPackCatalog.inspectInstalled(root)
        assertTrue(after.first { it.spec.id == "core_text" }.isReady)
        assertFalse(after.first { it.spec.id == "moetalk_avatar" }.isReady)
    }

    @Test
    fun inspectInstalledAcceptsLegacyFallbackFiles() {
        val root = tempFolder.newFolder("legacy_bundles")
        for (rel in listOf(
            ModelPackCatalog.DET_MODEL,
            ModelPackCatalog.REC_SMALL_LEGACY,
            "models/ocr_vocab.yml",
        )) {
            val f = File(root, rel)
            f.parentFile?.mkdirs()
            f.writeBytes(byteArrayOf(1, 2, 3))
        }
        val st = ModelPackCatalog.inspectInstalled(root).first { it.spec.id == "core_text" }
        assertTrue(st.isReady)
    }
}
