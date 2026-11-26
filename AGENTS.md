# MomoTranScript — Agent 工作约定

`MomoTranScript` 是一款完全离线的 Android 端对话长图转录与结构化还原应用：支持从相册、目录、嵌入式图像 PDF 或系统「分享」接收任意长度对话截图，端侧完成结构打框、原图 CV 切行、PP-OCRv6 仅识别、MomoScript 纯文本组装，以及可选外部视觉匹配与分支 `.json` 存档导出。

默认中文沟通。本项目定位为公开开源仓库：**绝不提交任何私钥（`*.jks`/`secrets.properties`）、凭据、私有归档原图、人工标注原稿、预编译 `.so` 或模型权重进 Git。**

## 技术架构与分层边界

- **UI 与系统集成（Kotlin + Jetpack Compose / Material 3）**：`app/src/main/kotlin/dev/momotranscript/app/`
  - 输入支持：单/多图 `PhotoPicker` 或 `ACTION_SEND*`、自然序图片文件夹（SAF `OpenDocumentTree`）、嵌入式图像 PDF（`pdfbox-android` 无损提取 `PDImageXObject`，回退 `PdfRenderer`）。
  - 后台保活：Jetpack `WorkManager` 前台任务（`TranscriptionWorker`，`FOREGROUND_SERVICE_DATA_SYNC`），支持逐页原子落盘、实时日志 Profile 与中途预览/导出。
  - 隐私与合规：首次启动隐私同意门禁（同意前零联网/零引擎加载）、`allowBackup="false"`、关于页备案与开源声明；APK 本体不硬编码第三方角色名或版权素材。
  - 严禁在 JVM 堆中反复拷贝整张超长图 RGB 数组；切窗、CV 切行、张量预处理、V2 后处理与模板对齐统一下沉至 C++20 原生层。
- **原生推理与核心管线（C++20 / NDK / JNI）**：`app/src/main/cpp/`（产物 `libmomotranscript_core.so`）
  - 推理引擎：**Alibaba MNN 3.6.2**（`libMNN.so` / `libMNN_Express.so` / `libMNN_CL.so` / `libMNN_Vulkan.so`，通过 `scripts/fetch_mnn_android.sh` 拉取，使用 `-DANDROID_STL=c++_shared`），支持运行时四档后端切换（自动 GPU+CPU / OpenCL / Vulkan / CPU FP16）与 GPU Shader 磁盘缓存。
  - 核心 C++20 模块位于 `app/src/main/cpp/core/`（`windowing`、`postprocess`、`cv_lines`、`tensor_prep`、`momoscript`、`visual_match`、`avifs_pack`），与 JNI 胶水隔离，可在 Linux 主机用 CTest 独立回归测试。
  - `postprocess.cpp`：11 类原生解码 + V2 同 query 最高类 + 本窗 TR（重叠面积 ≥ 10% 候选框面积）抑制 + 同类 IoU `0.55` 融合。
  - `momoscript.cpp`：左 `> 姓名: 正文`，右无显式姓名 `< 正文`，独立旁白 `-`，插图 `[image]`（归属说话人写 `> 姓名: [image]` 且续行省去重复姓名），回复 `@reply … @end`，附注 ` | 附注`。

## 模型与资产契约（不入 Git，按需下载/本地导入）

- 基础转录包（~43 MiB）：`mnn/det_boxes_order_ln_w8.mnn`（23 层 Fused-LN）、`mnn/rec_small_fp16.mnn`、`models/ocr_vocab.yml`
- 高精度级联包（~37 MiB）：`mnn/rec_medium_fp16.mnn`
- 视觉匹配扩展包（~52 MiB）：`mnn/ccip_caformer_24_ln_w8.mnn`（76 层 Fused-LN，无子图）、`avatar/catalog.json.zst`、`avatar/stage1_embeddings_svd128.npz`、`avatar/stage2_templates.avifs.pack`（`MMTAVIF1` / `schema_version=2`, `mask_dtype="uint8"`）
- 11 类结构类别顺序（严禁重排）：`0: avatar`, `1: speaker_name`, `2: bubble_left`, `3: bubble_right`, `4: narration`, `5: insert_image`, `6: reply`, `7: reply_item`, `8: bond`, `9: truncated_region`, `10: message_note`。

## 构建与验证约定

- `minSdk = 26`、`targetSdk = 36`、主 ABI `arm64-v8a`，Release 启用 R8 Full Mode + 资源裁剪（注意在 `app/proguard-rules.pro` 显式保留 `WorkDatabase_Impl`、`ListenableWorker`、`Initializer`、`NativeCore` 与 `@Serializable` 类构造函数）。
- 修改 C++20 核心后运行主机端单测：`cmake -S app/src/main/cpp/core -B build-host && cmake --build build-host && ctest --test-dir build-host --output-on-failure`
- 正式发版使用 `assembleRelease -Parm64Only=true` 并以 `apksigner verify` 验签。
