<p align="center">
  <img src="docs/assets/app_icon_512.png" width="128" height="128" alt="MomoTranScript Icon" />
</p>

<h1 align="center">MomoTranScript</h1>

<p align="center">
  <strong>Android 端侧离线对话长图 OCR 转录与结构化排版工具</strong><br/>
  Kotlin + Jetpack Compose (Material 3) + C++20 NDK + Alibaba MNN 3.6.2 (OpenCL GPU / Vulkan GPU / ARMv8.2-FP16)
</p>

<p align="center">
  <a href="https://github.com/xiyihan0/MomoTranScript/releases">GitHub Releases 下载</a> ·
  <a href="https://modelscope.cn/models/xiyihan/momotranscript-models/resolve/master/releases/MomoTranScript-0.5.3-arm64-release.apk">国内高速镜像（ModelScope）</a> ·
  <a href="https://www.xiyihan.cn/momotranscript/">应用主页</a> ·
  <a href="https://www.xiyihan.cn/momotranscript/privacy.html">隐私政策</a>
</p>

---

**MomoTranScript** 是一款完全在手机本地运行的 Android 对话长图与文档图像转录工具：支持从相册、文件夹、嵌入式图像 PDF 或系统「分享」接收任意长度对话截图，在端侧离线完成版面结构定位、原图物理 CV 切行、PP-OCRv6 识别与 MomoScript 纯文本格式化，并支持可选外部视觉匹配扩展包以导出结构化分支 `.json` 存档。

## 核心特性

- **纯本地离线、零隐私上传**：无需账号、零商业统计/广告 SDK，图片与 PDF 全程在设备本地完成推理，绝不上传云端；通过 Android `PhotoPicker` / `SAF` 最小权限读写，不申请全局存储权限。
- **MNN 3.6.2 跨厂商 GPU + ARMv9 FP16 异构加速**：
  - 支持 **自动（OpenCL GPU + ARMv8.2-FP16 CPU）/ OpenCL / Vulkan / CPU** 四档运行时切换与 GPU Shader 磁盘缓存；
  - 对检测模型（23 层）与视觉特征模型（76 层）做了原生 `LayerNormalization` 图融合，彻底消除移动端 FP16 溢出；
  - 实测参考（`vivo V2302A` 天玑 9200+，`345×16384` 超长图 / 25 窗 / 331 行 / 全开视觉匹配）：整页从 `74.7s`（ORT CPU）加速至 **`~30.6s`**，正式签名 R8 Release APK 仅 **`~18 MiB`**。
- **任意长度对话长图不截断、不 OOM**：
  - 空白带自适应切窗（`windowing.cpp`）+ 原图坐标系融合（`postprocess.cpp`）+ 1600px 分块矢量打框预览（`DetectionOverlayViewer.kt`），支持 20,000+ px 超长图流畅对照；
  - 全部图像切窗、双极性 8 连通域物理切行（零 OpenCV 依赖）、张量预处理与模板匹配均下沉至纯 C++20 核心库。
- **多源批处理与后台保活**：支持相册多选、自然序图片文件夹、嵌入式图像 PDF（`pdfbox-android` 无损直提原图，非纯图页自动回退 `PdfRenderer`），通过 Jetpack `WorkManager` 前台任务锁屏保活，支持**逐页流式预览与中途导出**。
- **模块化按需模型包**：APK 本体不含模型权重与第三方素材；应用内「模型」页可一键从 ModelScope 拉取（支持 SHA-256 校验）或从本地目录离线导入。

## 端侧模型包规格（运行时按需下载，不入 Git）

| 模块 | 文件 | 体积 | 说明 |
| --- | --- | --- | --- |
| **基础转录包（必选）** | `mnn/det_boxes_order_ln_w8.mnn`<br>`mnn/rec_small_fp16.mnn`<br>`models/ocr_vocab.yml` | ~43 MiB | 11 类 PP-DocLayoutV3（23 层 Fused-LN, W8 对称量化）+ PP-OCRv6 Small FP16 + 词表 |
| **高精度级联包（可选）** | `mnn/rec_medium_fp16.mnn` | ~37 MiB | PP-OCRv6 Medium FP16，仅对弱字/低置信度行自动复核 |
| **视觉匹配扩展包（可选）** | `mnn/ccip_caformer_24_ln_w8.mnn`<br>`avatar/catalog.json.zst`<br>`avatar/stage1_embeddings_svd128.npz`<br>`avatar/stage2_templates.avifs.pack` | ~52 MiB | 76 层 Fused-LN CCIP W8 + 无中心 SVD-128 FP16 向量库 + `MMTAVIF2`（`libavif`/`libdav1d` 4:4:4）模板容器 |

## 仓库结构

```text
MomoTranScript/
├── app/
│   ├── src/main/kotlin/dev/momotranscript/app/  # Jetpack Compose UI、批处理管线、模型管理、导出器
│   ├── src/main/cpp/
│   │   ├── core/                                # 平台无关 C++20 核心（切窗、后处理、CV 切行、MomoScript 组装、视觉匹配）
│   │   ├── mnn_engine.{hpp,cpp}                 # MNN 3.6.2 (OpenCL / Vulkan / ARMv8.2-FP16) 原生引擎封装
│   │   ├── jni_bridge.cpp                       # JNI 桥接层 -> libmomotranscript_core.so
│   │   └── third_party/mnn/include/             # MNN 3.6.2 C++ 头文件
│   └── proguard-rules.pro                       # R8 Full Mode 混淆与保活规则
├── scripts/fetch_mnn_android.sh                 # 拉取官方 MNN 3.6.2 arm64-v8a 预编译动态库
├── docs/web/                                    # 合规静态页（主页 / 隐私政策 / 用户协议）
└── LICENSE                                      # Apache-2.0
```

## 从源码构建

前置依赖：JDK 21、Android SDK（API 36）、Android NDK 28、CMake 3.22+。

```bash
# 1. 拉取官方预编译 MNN 3.6.2 arm64-v8a 动态库（不随源码提交）
chmod +x scripts/fetch_mnn_android.sh
./scripts/fetch_mnn_android.sh

# 2. 运行 Linux 主机端 C++20 核心单元测试
cmake -S app/src/main/cpp/core -B build-host
cmake --build build-host
ctest --test-dir build-host --output-on-failure

# 3. 构建 Android Debug APK（仅 arm64-v8a）
./gradlew :app:assembleDebug -Parm64Only=true

# 4. 构建签名 Release + R8 APK（需本地配置 secrets.properties 或环境变量指定签名 keystore）
./gradlew :app:assembleRelease -Parm64Only=true
```

## 开源许可与第三方致谢

本项目源代码采用 [Apache-2.0](LICENSE) 许可协议开源。本仓库不包含任何签名私钥、第三方游戏版权素材、私人归档原图或预训练模型权重。

端侧运行依赖以下优秀开源项目：
- [Alibaba MNN](https://github.com/alibaba/MNN) (Apache-2.0)
- [AOMedia libavif](https://github.com/AOMediaCodec/libavif) / dav1d (BSD-2-Clause)
- [PDFBox-Android](https://github.com/TomRoush/PdfBox-Android) (Apache-2.0)
- [zstd-jni](https://github.com/luben/zstd-jni) (BSD-2-Clause)
