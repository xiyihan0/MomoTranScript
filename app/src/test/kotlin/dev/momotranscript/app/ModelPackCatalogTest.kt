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
        assertEquals(listOf("core_text", "rec_medium", "moetalk_avatar"), ids)
        val corePack = ModelPackCatalog.packs.first { it.id == "core_text" }
        assertTrue(corePack.required)
        assertTrue(corePack.approxTotalMiB in 40.0..45.0)
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
}
