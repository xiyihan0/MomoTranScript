import java.util.Properties

plugins {
  alias(libs.plugins.android.application)
  alias(libs.plugins.compose.compiler)
  alias(libs.plugins.kotlin.serialization)
}

val secretsFile = rootProject.file("secrets.properties")
val secretsProps = Properties().apply {
  if (secretsFile.isFile) {
    secretsFile.inputStream().use { load(it) }
  }
}

fun secretOrEnv(key: String): String? =
  System.getenv(key)?.takeIf { it.isNotBlank() }
    ?: secretsProps.getProperty(key)?.takeIf { it.isNotBlank() }

android {
  namespace = "dev.momotranscript.app"
  compileSdk = 36
  ndkVersion = "28.2.13676358"

  defaultConfig {
    applicationId = "dev.momotranscript.app"
    minSdk = 26
    targetSdk = 36
    versionCode = 26
    versionName = "0.6.8"

    buildConfigField("String", "HOMEPAGE_URL", "\"https://www.xiyihan.cn/momotranscript/\"")
    buildConfigField("String", "PRIVACY_POLICY_URL", "\"https://www.xiyihan.cn/momotranscript/privacy.html\"")
    buildConfigField("String", "USER_AGREEMENT_URL", "\"https://www.xiyihan.cn/momotranscript/terms.html\"")
    buildConfigField("String", "SITE_ICP_NUMBER", "\"湘ICP备2023031823号-1\"")
    buildConfigField("String", "APP_ICP_NUMBER", "\"\"")
    buildConfigField("String", "CONTACT_EMAIL", "\"momotranscript@xiyihan.cn\"")

    ndk {
      val arm64Only = providers.gradleProperty("arm64Only").orNull?.toBoolean() == true
      if (arm64Only) {
        abiFilters += listOf("arm64-v8a")
      } else {
        abiFilters += listOf("arm64-v8a", "x86_64")
      }
    }
    externalNativeBuild {
      cmake {
        cppFlags += listOf("-std=c++20", "-O3")
        arguments += listOf("-DANDROID_STL=c++_shared")
      }
    }
  }

  signingConfigs {
    val storePath = secretOrEnv("RELEASE_STORE_FILE")
    val storePw = secretOrEnv("RELEASE_STORE_PASSWORD")
    val keyAliasVal = secretOrEnv("RELEASE_KEY_ALIAS")
    val keyPw = secretOrEnv("RELEASE_KEY_PASSWORD")
    if (storePath != null && storePw != null && keyAliasVal != null && keyPw != null) {
      val ksFile = rootProject.file(storePath)
      if (ksFile.isFile) {
        create("release") {
          storeFile = ksFile
          storePassword = storePw
          keyAlias = keyAliasVal
          keyPassword = keyPw
          enableV1Signing = true
          enableV2Signing = true
          enableV3Signing = true
        }
      }
    }
  }

  externalNativeBuild {
    cmake {
      path = file("src/main/cpp/CMakeLists.txt")
      version = "3.22.1"
    }
  }

  buildTypes {
    release {
      isMinifyEnabled = true
      isShrinkResources = true
      proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
      signingConfigs.findByName("release")?.let { signingConfig = it }
    }
  }
  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_21
    targetCompatibility = JavaVersion.VERSION_21
  }
  buildFeatures {
    compose = true
    aidl = false
    buildConfig = true
    shaders = false
  }

  packaging {
    resources {
      excludes += listOf(
        "/META-INF/{AL2.0,LGPL2.1}",
        // Strip unused BouncyCastle post-quantum parameter tables pulled in transitively by pdfbox-android (~4.0 MiB)
        "org/bouncycastle/pqc/**",
      )
    }
  }
}

kotlin {
  jvmToolchain(21)
}

dependencies {
  val composeBom = platform(libs.androidx.compose.bom)
  implementation(composeBom)
  androidTestImplementation(composeBom)

  implementation(libs.androidx.core.ktx)
  implementation(libs.androidx.lifecycle.runtime.ktx)
  implementation(libs.androidx.activity.compose)
  implementation(libs.androidx.lifecycle.runtime.compose)
  implementation(libs.androidx.lifecycle.viewmodel.compose)

  implementation(libs.androidx.compose.ui)
  implementation(libs.androidx.compose.ui.tooling.preview)
  implementation(libs.androidx.compose.material3)
  implementation(libs.androidx.compose.material.icons)
  debugImplementation(libs.androidx.compose.ui.tooling)

  // Native MNN 3.6.2 (OpenCL GPU / Vulkan GPU / ARMv9 FP16 CPU) is linked directly via CMake + jniLibs
  implementation(libs.kotlinx.serialization.json)
  // Native Zstd decompression for compact avatar catalog (`catalog.json.zst`)
  implementation("com.github.luben:zstd-jni:1.5.6-9@aar")
  // Official AOMedia libavif + libdav1d decoder for 4:4:4 multi-frame `.avifs` templates
  implementation(libs.aom.avif)
  // Apache PDFBox Android port for robust streaming embedded-image PDF extraction
  implementation(libs.pdfbox.android)
  // Jetpack WorkManager for foreground background-safe long batch & 200-page PDF transcription
  implementation(libs.androidx.work.runtime.ktx)

  testImplementation(libs.junit)
  testImplementation(libs.kotlinx.coroutines.test)

  androidTestImplementation(libs.androidx.compose.ui.test.junit4)
  debugImplementation(libs.androidx.compose.ui.test.manifest)
  androidTestImplementation(libs.androidx.test.core)
  androidTestImplementation(libs.androidx.test.ext.junit)
  androidTestImplementation(libs.androidx.test.runner)
  androidTestImplementation(libs.androidx.test.espresso.core)
}
