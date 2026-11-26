# MomoTranScript Release R8 / ProGuard Rules

# 1. Preserve JNI entry points & native-called models in NativeCore
-keep class dev.momotranscript.app.nativecore.NativeCore {
    *;
}
-keep class dev.momotranscript.app.nativecore.NativeCore$* {
    *;
}
-keep class dev.momotranscript.app.nativecore.WindowBox { *; }
-keep class dev.momotranscript.app.nativecore.TextLineBox { *; }
-keep class dev.momotranscript.app.nativecore.NativeDetectedElement { *; }
-keep class dev.momotranscript.app.nativecore.NativeAssembledMessage { *; }
-keep class dev.momotranscript.app.nativecore.NativeAssemblyResult { *; }

# 2. Preserve Kotlinx Serialization models & generated serializers
-keepattributes *Annotation*, InnerClasses
-dontnote kotlinx.serialization.AnnotationsKt
-keepclassmembers class kotlinx.serialization.json.** {
    *** Companion;
}
-keepclasseswithmembers class kotlinx.serialization.json.** {
    kotlinx.serialization.KSerializer serializer(...);
}
-keep,includedescriptorclasses class dev.momotranscript.app.**$$serializer { *; }
-keepclassmembers class dev.momotranscript.app.** {
    *** Companion;
}
-keepclasseswithmembers class dev.momotranscript.app.** {
    kotlinx.serialization.KSerializer serializer(...);
}

# 3. Preserve third-party JNI wrappers (libavif, zstd-jni)
-keep class org.aomedia.avif.android.** { *; }
-keep class com.github.luben.zstd.** { *; }
-dontwarn com.github.luben.zstd.**

# 4. Preserve Jetpack WorkManager, Room (R8 fullMode!), & Startup Initializers
-keep class * extends androidx.room.RoomDatabase {
    public <init>();
}
-keep class androidx.work.impl.WorkDatabase_Impl {
    *;
}
-keep class * extends androidx.work.ListenableWorker {
    public <init>(android.content.Context, androidx.work.WorkerParameters);
}
-keep class * extends androidx.work.InputMerger {
    public <init>();
}
-keep class * implements androidx.startup.Initializer {
    public <init>();
}
-keep class dev.momotranscript.app.work.TranscriptionWorker {
    public <init>(android.content.Context, androidx.work.WorkerParameters);
}
-keep,includedescriptorclasses class dev.momotranscript.app.work.TranscriptionJobSpec { *; }
-keep,includedescriptorclasses class dev.momotranscript.app.work.TranscriptionCheckpoint { *; }
-keep,includedescriptorclasses class dev.momotranscript.app.work.CheckpointBranchInput { *; }
-keep,includedescriptorclasses class dev.momotranscript.app.ui.PagePreviewRecord { *; }
-keep,includedescriptorclasses class dev.momotranscript.app.avatar.AvatarCatalog { *; }
-keep,includedescriptorclasses class dev.momotranscript.app.avatar.CatalogChar { *; }
-keep,includedescriptorclasses class dev.momotranscript.app.avatar.CatalogPackV2 { *; }
-keep,includedescriptorclasses class dev.momotranscript.app.avatar.CatalogExpression { *; }
-keep,includedescriptorclasses class dev.momotranscript.app.avatar.CatalogSkin { *; }
-keep,includedescriptorclasses class dev.momotranscript.app.avatar.CatalogEmoji { *; }
-keep,includedescriptorclasses class dev.momotranscript.app.avatar.CatalogOptions { *; }
-keep,includedescriptorclasses class dev.momotranscript.app.avatar.AvifsPackSkinEntry { *; }
-keep,includedescriptorclasses class dev.momotranscript.app.avatar.AvifsPackEmojiChunk { *; }
-keep,includedescriptorclasses class dev.momotranscript.app.avatar.AvifsPackHeader { *; }

# 5. Apache PDFBox-Android (uses reflection/resources internally)
-keep class com.tom_roush.pdfbox.** { *; }
-keep class com.tom_roush.fontbox.** { *; }
-dontwarn com.tom_roush.pdfbox.**
-dontwarn com.tom_roush.fontbox.**
-dontwarn org.bouncycastle.**
-dontwarn javax.xml.**
-dontwarn java.awt.**
