package dev.momotranscript.app.input

import java.util.Locale

/**
 * Natural (human-friendly) numeric sorting and filename filtering for folder / batch inputs.
 *
 * Ensures e.g. `2-2.png` comes before `2-10.png`, and `第9话.jpg` comes before `第10话.jpg`,
 * without overflowing on arbitrarily long numeric sequences.
 */
object NaturalSort {
    private val supportedImageExtensions = setOf(
        "jpg",
        "jpeg",
        "png",
        "webp",
        "bmp",
        "gif",
        "heic",
        "heif",
        "avif",
        "jxl",
    )

    fun isSupportedImageName(name: String): Boolean {
        val dot = name.lastIndexOf('.')
        if (dot < 0 || dot == name.lastIndex) return false
        val ext = name.substring(dot + 1).lowercase(Locale.ROOT)
        return ext in supportedImageExtensions
    }

    fun isPdfName(name: String): Boolean =
        name.lowercase(Locale.ROOT).endsWith(".pdf")

    val comparator: Comparator<String> = Comparator { a, b -> compare(a, b) }

    fun compare(a: String, b: String): Int {
        var ia = 0
        var ib = 0
        val na = a.length
        val nb = b.length

        while (ia < na && ib < nb) {
            val ca = a[ia]
            val cb = b[ib]
            val da = ca in '0'..'9'
            val db = cb in '0'..'9'

            if (da && db) {
                val startA = ia
                while (ia < na && a[ia] in '0'..'9') ia++
                val startB = ib
                while (ib < nb && b[ib] in '0'..'9') ib++

                val cmp = compareDigitRuns(a, startA, ia, b, startB, ib)
                if (cmp != 0) return cmp
            } else {
                val la = ca.lowercaseChar()
                val lb = cb.lowercaseChar()
                if (la != lb) return la.compareTo(lb)
                ia++
                ib++
            }
        }
        val lenCmp = (na - ia).compareTo(nb - ib)
        if (lenCmp != 0) return lenCmp
        return a.compareTo(b)
    }

    private fun compareDigitRuns(
        a: String,
        startA: Int,
        endA: Int,
        b: String,
        startB: Int,
        endB: Int,
    ): Int {
        var za = startA
        while (za < endA && a[za] == '0') za++
        var zb = startB
        while (zb < endB && b[zb] == '0') zb++

        val sigLenA = endA - za
        val sigLenB = endB - zb
        if (sigLenA != sigLenB) return sigLenA.compareTo(sigLenB)

        for (i in 0 until sigLenA) {
            val diff = a[za + i].compareTo(b[zb + i])
            if (diff != 0) return diff
        }
        // Tie-break on leading-zero count so ordering is strictly deterministic.
        val rawLenA = endA - startA
        val rawLenB = endB - startB
        return rawLenA.compareTo(rawLenB)
    }
}
