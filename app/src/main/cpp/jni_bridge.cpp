#include <jni.h>

#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "core/avifs_pack.hpp"
#include "core/cv_lines.hpp"
#include "core/mmt_types.hpp"
#include "core/momoscript.hpp"
#include "core/postprocess.hpp"
#include "core/tensor_prep.hpp"
#include "core/visual_match.hpp"
#include "core/windowing.hpp"
#include "mnn_engine.hpp"

namespace {

jintArray windows_to_jint_array(JNIEnv* env, const std::vector<mmt::Window>& windows) {
    std::vector<jint> flat;
    flat.reserve(windows.size() * 4);
    for (const auto& w : windows) {
        flat.push_back(w.x0);
        flat.push_back(w.y0);
        flat.push_back(w.x1);
        flat.push_back(w.y1);
    }
    jintArray arr = env->NewIntArray(static_cast<jsize>(flat.size()));
    if (arr != nullptr && !flat.empty()) {
        env->SetIntArrayRegion(arr, 0, static_cast<jsize>(flat.size()), flat.data());
    }
    return arr;
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    return out;
}

std::string jstring_to_std(JNIEnv* env, jstring js) {
    if (js == nullptr) return "";
    const char* ptr = env->GetStringUTFChars(js, nullptr);
    std::string s(ptr ? ptr : "");
    if (ptr != nullptr) env->ReleaseStringUTFChars(js, ptr);
    return s;
}

}  // namespace

extern "C" {

JNIEXPORT jstring JNICALL
Java_dev_momotranscript_app_nativecore_NativeCore_nativeCoreVersion(
    JNIEnv* env,
    jobject /* thiz */) {
    return env->NewStringUTF("MomoTranScript C++20 Core + MNN 3.6.2 (OpenCL / Vulkan / ARMv9 FP16)");
}

JNIEXPORT jintArray JNICALL
Java_dev_momotranscript_app_nativecore_NativeCore_nativePlanFixedWindows(
    JNIEnv* env,
    jobject /* thiz */,
    jint width,
    jint height) {
    if (width <= 0 || height <= 0) {
        return env->NewIntArray(0);
    }
    auto windows = mmt::large_only_windows(static_cast<int>(width), static_cast<int>(height));
    return windows_to_jint_array(env, windows);
}

JNIEXPORT jintArray JNICALL
Java_dev_momotranscript_app_nativecore_NativeCore_nativePlanAdaptiveWindows(
    JNIEnv* env,
    jobject /* thiz */,
    jbyteArray rgb_bytes,
    jint width,
    jint height) {
    if (rgb_bytes == nullptr || width <= 0 || height <= 0) {
        return env->NewIntArray(0);
    }
    const jsize expected = width * height * 3;
    if (env->GetArrayLength(rgb_bytes) < expected) {
        return env->NewIntArray(0);
    }
    jbyte* raw = env->GetByteArrayElements(rgb_bytes, nullptr);
    if (raw == nullptr) {
        return env->NewIntArray(0);
    }
    auto windows = mmt::adaptive_blank_windows(
        reinterpret_cast<const uint8_t*>(raw),
        static_cast<int>(width),
        static_cast<int>(height));
    env->ReleaseByteArrayElements(rgb_bytes, raw, JNI_ABORT);
    return windows_to_jint_array(env, windows);
}

JNIEXPORT jfloatArray JNICALL
Java_dev_momotranscript_app_nativecore_NativeCore_nativePrepareDetInput(
    JNIEnv* env,
    jobject /* thiz */,
    jbyteArray rgb_bytes,
    jint img_w,
    jint img_h,
    jint x0,
    jint y0,
    jint x1,
    jint y1,
    jint dst_size) {
    if (rgb_bytes == nullptr || img_w <= 0 || img_h <= 0 || dst_size <= 0) {
        return env->NewFloatArray(0);
    }
    jbyte* raw = env->GetByteArrayElements(rgb_bytes, nullptr);
    if (raw == nullptr) {
        return env->NewFloatArray(0);
    }
    const size_t total = static_cast<size_t>(3) * dst_size * dst_size;
    std::vector<float> buf(total);
    mmt::prepare_det_input_nchw(
        reinterpret_cast<const uint8_t*>(raw),
        static_cast<int>(img_w),
        static_cast<int>(img_h),
        mmt::Window{x0, y0, x1, y1},
        static_cast<int>(dst_size),
        buf.data());
    env->ReleaseByteArrayElements(rgb_bytes, raw, JNI_ABORT);

    jfloatArray out = env->NewFloatArray(static_cast<jsize>(total));
    if (out != nullptr) {
        env->SetFloatArrayRegion(out, 0, static_cast<jsize>(total), buf.data());
    }
    return out;
}

JNIEXPORT jstring JNICALL
Java_dev_momotranscript_app_nativecore_NativeCore_nativeDecodeAndFuseDetections(
    JNIEnv* env,
    jobject /* thiz */,
    jfloatArray bboxes_flat,
    jfloatArray logits_flat,
    jint num_queries,
    jintArray windows_flat,
    jint img_w,
    jint img_h) {
    if (bboxes_flat == nullptr || logits_flat == nullptr || windows_flat == nullptr || num_queries <= 0) {
        return env->NewStringUTF("[]");
    }
    const jsize win_len = env->GetArrayLength(windows_flat);
    const int num_windows = win_len / 4;
    if (num_windows <= 0) {
        return env->NewStringUTF("[]");
    }

    jfloat* bboxes = env->GetFloatArrayElements(bboxes_flat, nullptr);
    jfloat* logits = env->GetFloatArrayElements(logits_flat, nullptr);
    jint* wins = env->GetIntArrayElements(windows_flat, nullptr);

    std::vector<mmt::MappedCandidate> all_cands;
    for (int wi = 0; wi < num_windows; ++wi) {
        mmt::Window w{wins[wi * 4 + 0], wins[wi * 4 + 1], wins[wi * 4 + 2], wins[wi * 4 + 3]};
        const float* wb = bboxes + static_cast<size_t>(wi) * num_queries * 4;
        const float* wl = logits + static_cast<size_t>(wi) * num_queries * mmt::kNumClasses;
        auto cands = mmt::decode_window_candidates(wb, wl, num_queries, w, wi, img_w, img_h);
        all_cands.insert(all_cands.end(), cands.begin(), cands.end());
    }

    env->ReleaseFloatArrayElements(bboxes_flat, bboxes, JNI_ABORT);
    env->ReleaseFloatArrayElements(logits_flat, logits, JNI_ABORT);
    env->ReleaseIntArrayElements(windows_flat, wins, JNI_ABORT);

    auto elems = mmt::fuse_and_finalize(std::move(all_cands), img_w, img_h);

    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < elems.size(); ++i) {
        if (i > 0) oss << ",";
        const auto& e = elems[i];
        oss << "{\"id\":\"" << json_escape(e.id) << "\""
            << ",\"classId\":" << e.class_id
            << ",\"score\":" << e.score
            << ",\"box\":[" << e.box.x0 << "," << e.box.y0 << "," << e.box.x1 << "," << e.box.y1 << "]"
            << ",\"parentId\":\"" << json_escape(e.parent_id) << "\"}";
    }
    oss << "]";
    return env->NewStringUTF(oss.str().c_str());
}

JNIEXPORT jintArray JNICALL
Java_dev_momotranscript_app_nativecore_NativeCore_nativeSegmentLines(
    JNIEnv* env,
    jobject /* thiz */,
    jbyteArray rgb_bytes,
    jint img_w,
    jint img_h,
    jint x0,
    jint y0,
    jint x1,
    jint y1,
    jboolean tight) {
    if (rgb_bytes == nullptr || img_w <= 0 || img_h <= 0) {
        return env->NewIntArray(0);
    }
    jbyte* raw = env->GetByteArrayElements(rgb_bytes, nullptr);
    if (raw == nullptr) {
        return env->NewIntArray(0);
    }
    auto lines = mmt::segment_physical_lines(
        reinterpret_cast<const uint8_t*>(raw),
        static_cast<int>(img_w),
        static_cast<int>(img_h),
        mmt::BoxI{x0, y0, x1, y1},
        tight == JNI_TRUE);
    env->ReleaseByteArrayElements(rgb_bytes, raw, JNI_ABORT);

    std::vector<jint> flat;
    flat.reserve(lines.size() * 9);
    for (const auto& ln : lines) {
        flat.push_back(ln.ink_box.x0);
        flat.push_back(ln.ink_box.y0);
        flat.push_back(ln.ink_box.x1);
        flat.push_back(ln.ink_box.y1);
        flat.push_back(ln.box.x0);
        flat.push_back(ln.box.y0);
        flat.push_back(ln.box.x1);
        flat.push_back(ln.box.y1);
        flat.push_back(ln.is_light_fg ? 1 : 0);
    }
    jintArray out = env->NewIntArray(static_cast<jsize>(flat.size()));
    if (out != nullptr && !flat.empty()) {
        env->SetIntArrayRegion(out, 0, static_cast<jsize>(flat.size()), flat.data());
    }
    return out;
}

JNIEXPORT jfloatArray JNICALL
Java_dev_momotranscript_app_nativecore_NativeCore_nativePrepareRecLine(
    JNIEnv* env,
    jobject /* thiz */,
    jbyteArray rgb_bytes,
    jint img_w,
    jint img_h,
    jint x0,
    jint y0,
    jint x1,
    jint y1,
    jboolean is_light,
    jint target_pad_h,
    jint dst_h,
    jint max_w) {
    if (rgb_bytes == nullptr || img_w <= 0 || img_h <= 0 || dst_h <= 0) {
        return env->NewFloatArray(0);
    }
    jbyte* raw = env->GetByteArrayElements(rgb_bytes, nullptr);
    if (raw == nullptr) {
        return env->NewFloatArray(0);
    }
    mmt::TextLineProposal prop{
        mmt::BoxI{x0, y0, x1, y1},
        mmt::BoxI{x0, y0, x1, y1},
        is_light == JNI_TRUE,
    };
    std::vector<float> chw;
    int resized_w = 0;
    mmt::prepare_rec_line_chw(
        reinterpret_cast<const uint8_t*>(raw),
        static_cast<int>(img_w),
        static_cast<int>(img_h),
        prop,
        static_cast<int>(target_pad_h),
        static_cast<int>(dst_h),
        static_cast<int>(max_w),
        chw,
        resized_w);
    env->ReleaseByteArrayElements(rgb_bytes, raw, JNI_ABORT);

    jfloatArray out = env->NewFloatArray(static_cast<jsize>(chw.size()));
    if (out != nullptr && !chw.empty()) {
        env->SetFloatArrayRegion(out, 0, static_cast<jsize>(chw.size()), chw.data());
    }
    return out;
}

JNIEXPORT jstring JNICALL
Java_dev_momotranscript_app_nativecore_NativeCore_nativeAssembleMomoScript(
    JNIEnv* env,
    jobject /* thiz */,
    jobjectArray ids_arr,
    jintArray class_ids_arr,
    jfloatArray scores_arr,
    jintArray boxes_arr,
    jobjectArray parent_ids_arr,
    jobjectArray texts_arr,
    jint img_w,
    jint img_h) {
    const jsize n = ids_arr ? env->GetArrayLength(ids_arr) : 0;
    if (n <= 0) {
        return env->NewStringUTF(R"({"text":"","messages":[]})");
    }
    jint* cids = env->GetIntArrayElements(class_ids_arr, nullptr);
    jfloat* scs = env->GetFloatArrayElements(scores_arr, nullptr);
    jint* bxs = env->GetIntArrayElements(boxes_arr, nullptr);

    std::vector<mmt::RecognizedRegion> regs;
    regs.reserve(static_cast<size_t>(n));
    for (jsize i = 0; i < n; ++i) {
        auto id_s = jstring_to_std(env, static_cast<jstring>(env->GetObjectArrayElement(ids_arr, i)));
        auto pid_s = jstring_to_std(env, static_cast<jstring>(env->GetObjectArrayElement(parent_ids_arr, i)));
        auto txt_s = jstring_to_std(env, static_cast<jstring>(env->GetObjectArrayElement(texts_arr, i)));
        regs.push_back(mmt::RecognizedRegion{
            std::move(id_s),
            cids[i],
            scs[i],
            mmt::BoxI{bxs[i * 4 + 0], bxs[i * 4 + 1], bxs[i * 4 + 2], bxs[i * 4 + 3]},
            std::move(pid_s),
            std::move(txt_s),
        });
    }

    env->ReleaseIntArrayElements(class_ids_arr, cids, JNI_ABORT);
    env->ReleaseFloatArrayElements(scores_arr, scs, JNI_ABORT);
    env->ReleaseIntArrayElements(boxes_arr, bxs, JNI_ABORT);

    auto assembled = mmt::assemble_momoscript(regs, img_w, img_h);

    std::ostringstream oss;
    oss << "{\"text\":\"" << json_escape(assembled.momoscript_text) << "\",\"messages\":[";
    for (size_t i = 0; i < assembled.messages.size(); ++i) {
        if (i > 0) oss << ",";
        const auto& m = assembled.messages[i];
        oss << "{\"kind\":\"" << json_escape(m.kind) << "\""
            << ",\"speaker\":\"" << json_escape(m.speaker) << "\""
            << ",\"isRight\":" << (m.is_right ? "true" : "false")
            << ",\"isFirstTurn\":" << (m.is_first_turn ? "true" : "false")
            << ",\"content\":\"" << json_escape(m.content) << "\""
            << ",\"note\":\"" << json_escape(m.note) << "\""
            << ",\"box\":[" << m.box.x0 << "," << m.box.y0 << "," << m.box.x1 << "," << m.box.y1 << "]}";
    }
    oss << "]}";
    return env->NewStringUTF(oss.str().c_str());
}

JNIEXPORT jstring JNICALL
Java_dev_momotranscript_app_nativecore_NativeCore_nativeInspectAvifsPack(
    JNIEnv* env,
    jobject /* thiz */,
    jstring path_jstr) {
    if (path_jstr == nullptr) {
        return env->NewStringUTF(R"({"ok":false,"error":"null path"})");
    }
    std::string path = jstring_to_std(env, path_jstr);
    mmt::AvifsPackReader reader;
    std::string err;
    if (!reader.open(path, &err)) {
        std::string msg = R"({"ok":false,"error":")" + json_escape(err) + R"("})";
        return env->NewStringUTF(msg.c_str());
    }
    const auto& idx = reader.index();
    char buf[256];
    std::snprintf(
        buf,
        sizeof(buf),
        R"({"ok":true,"schema_version":%d,"tpl_size":%d,"skin_count":%d,"expression_count":%d,"emoji_count":%d})",
        idx.schema_version,
        idx.tpl_size,
        idx.skin_count,
        idx.expression_count,
        idx.emoji_count);
    return env->NewStringUTF(buf);
}

JNIEXPORT jintArray JNICALL
Java_dev_momotranscript_app_nativecore_NativeCore_nativeSplitAvatarBox(
    JNIEnv* env,
    jobject /* thiz */,
    jint x0,
    jint y0,
    jint x1,
    jint y1,
    jint max_members) {
    auto boxes = mmt::split_avatar_box(mmt::BoxI{x0, y0, x1, y1}, max_members);
    std::vector<jint> flat;
    flat.reserve(boxes.size() * 4);
    for (const auto& b : boxes) {
        flat.push_back(b.x0);
        flat.push_back(b.y0);
        flat.push_back(b.x1);
        flat.push_back(b.y1);
    }
    jintArray out = env->NewIntArray(static_cast<jsize>(flat.size()));
    if (out != nullptr && !flat.empty()) {
        env->SetIntArrayRegion(out, 0, static_cast<jsize>(flat.size()), flat.data());
    }
    return out;
}

JNIEXPORT jfloatArray JNICALL
Java_dev_momotranscript_app_nativecore_NativeCore_nativePrepareCcipInput(
    JNIEnv* env,
    jobject /* thiz */,
    jbyteArray rgb_bytes,
    jint img_w,
    jint img_h,
    jint x0,
    jint y0,
    jint x1,
    jint y1,
    jboolean circle_mask,
    jboolean trim_borders,
    jint dst_size,
    jboolean norm_half,
    jfloatArray out_meta) {
    if (rgb_bytes == nullptr || img_w <= 0 || img_h <= 0 || dst_size <= 0) {
        return env->NewFloatArray(0);
    }
    jbyte* raw = env->GetByteArrayElements(rgb_bytes, nullptr);
    if (raw == nullptr) return env->NewFloatArray(0);

    const size_t total = static_cast<size_t>(3) * dst_size * dst_size;
    std::vector<float> buf(total);
    float meta[2] = {0.0f, 1.0f};
    mmt::prepare_ccip_input_nchw(
        reinterpret_cast<const uint8_t*>(raw),
        img_w,
        img_h,
        mmt::BoxI{x0, y0, x1, y1},
        circle_mask == JNI_TRUE,
        trim_borders == JNI_TRUE,
        dst_size,
        buf.data(),
        &meta[0],
        &meta[1],
        norm_half == JNI_TRUE);
    env->ReleaseByteArrayElements(rgb_bytes, raw, JNI_ABORT);

    if (out_meta != nullptr && env->GetArrayLength(out_meta) >= 2) {
        env->SetFloatArrayRegion(out_meta, 0, 2, meta);
    }
    jfloatArray out = env->NewFloatArray(static_cast<jsize>(total));
    if (out != nullptr) {
        env->SetFloatArrayRegion(out, 0, static_cast<jsize>(total), buf.data());
    }
    return out;
}

JNIEXPORT jfloatArray JNICALL
Java_dev_momotranscript_app_nativecore_NativeCore_nativeComputeRgbThumbnail(
    JNIEnv* env,
    jobject /* thiz */,
    jbyteArray rgb_bytes,
    jint img_w,
    jint img_h,
    jint x0,
    jint y0,
    jint x1,
    jint y1,
    jboolean circle_mask,
    jint thumb_size) {
    if (rgb_bytes == nullptr || img_w <= 0 || img_h <= 0 || thumb_size <= 0) {
        return env->NewFloatArray(0);
    }
    jbyte* raw = env->GetByteArrayElements(rgb_bytes, nullptr);
    if (raw == nullptr) return env->NewFloatArray(0);

    const size_t total = static_cast<size_t>(thumb_size) * thumb_size * 3;
    std::vector<float> buf(total);
    mmt::compute_rgb_thumbnail(
        reinterpret_cast<const uint8_t*>(raw),
        img_w,
        img_h,
        mmt::BoxI{x0, y0, x1, y1},
        circle_mask == JNI_TRUE,
        thumb_size,
        buf.data());
    env->ReleaseByteArrayElements(rgb_bytes, raw, JNI_ABORT);

    jfloatArray out = env->NewFloatArray(static_cast<jsize>(total));
    if (out != nullptr) {
        env->SetFloatArrayRegion(out, 0, static_cast<jsize>(total), buf.data());
    }
    return out;
}

JNIEXPORT jfloatArray JNICALL
Java_dev_momotranscript_app_nativecore_NativeCore_nativeScoreStage2Skin(
    JNIEnv* env,
    jobject /* thiz */,
    jbyteArray rgb_bytes,
    jint img_w,
    jint img_h,
    jint x0,
    jint y0,
    jint x1,
    jint y1,
    jbyteArray stack_rgb,
    jint count,
    jint tpl_size,
    jfloatArray var_weights) {
    float res[3] = {0.0f, -1.0f, -1.0f};
    if (rgb_bytes != nullptr && stack_rgb != nullptr && var_weights != nullptr && count > 0 && tpl_size > 0) {
        jbyte* raw = env->GetByteArrayElements(rgb_bytes, nullptr);
        jbyte* stk = env->GetByteArrayElements(stack_rgb, nullptr);
        jfloat* vw = env->GetFloatArrayElements(var_weights, nullptr);
        if (raw != nullptr && stk != nullptr && vw != nullptr) {
            auto st = mmt::score_stage2_skin_templates(
                reinterpret_cast<const uint8_t*>(raw),
                img_w,
                img_h,
                mmt::BoxI{x0, y0, x1, y1},
                reinterpret_cast<const uint8_t*>(stk),
                count,
                tpl_size,
                vw);
            res[0] = static_cast<float>(st.best_index);
            res[1] = st.diff_zncc;
            res[2] = st.struct_zncc;
        }
        if (raw != nullptr) env->ReleaseByteArrayElements(rgb_bytes, raw, JNI_ABORT);
        if (stk != nullptr) env->ReleaseByteArrayElements(stack_rgb, stk, JNI_ABORT);
        if (vw != nullptr) env->ReleaseFloatArrayElements(var_weights, vw, JNI_ABORT);
    }
    jfloatArray out = env->NewFloatArray(3);
    if (out != nullptr) env->SetFloatArrayRegion(out, 0, 3, res);
    return out;
}

JNIEXPORT jfloat JNICALL
Java_dev_momotranscript_app_nativecore_NativeCore_nativeScoreEmojiStructure(
    JNIEnv* env,
    jobject /* thiz */,
    jbyteArray rgb_bytes,
    jint img_w,
    jint img_h,
    jint x0,
    jint y0,
    jint x1,
    jint y1,
    jbyteArray tpl_rgb,
    jint tpl_size) {
    if (rgb_bytes == nullptr || tpl_rgb == nullptr || tpl_size <= 0) return -1.0f;
    jbyte* raw = env->GetByteArrayElements(rgb_bytes, nullptr);
    jbyte* tpl = env->GetByteArrayElements(tpl_rgb, nullptr);
    float zncc = -1.0f;
    if (raw != nullptr && tpl != nullptr) {
        zncc = mmt::score_emoji_structure_zncc(
            reinterpret_cast<const uint8_t*>(raw),
            img_w,
            img_h,
            mmt::BoxI{x0, y0, x1, y1},
            reinterpret_cast<const uint8_t*>(tpl),
            tpl_size);
    }
    if (raw != nullptr) env->ReleaseByteArrayElements(rgb_bytes, raw, JNI_ABORT);
    if (tpl != nullptr) env->ReleaseByteArrayElements(tpl_rgb, tpl, JNI_ABORT);
    return zncc;
}

// ============================= MNN 3.6.2 JNI Bridge =============================

JNIEXPORT jstring JNICALL
Java_dev_momotranscript_app_nativecore_NativeCore_nativeMnnDeviceInfo(
    JNIEnv* env,
    jobject /* thiz */) {
    return env->NewStringUTF(mmt::mnn_probe_device_info().c_str());
}

JNIEXPORT jlong JNICALL
Java_dev_momotranscript_app_nativecore_NativeCore_nativeMnnDetCreate(
    JNIEnv* env,
    jobject /* thiz */,
    jstring model_path,
    jint backend_req,
    jint threads,
    jstring cache_path,
    jobjectArray out_info) {
    auto* sess = new mmt::MnnDetSession();
    std::string err;
    const bool ok = sess->load(
        jstring_to_std(env, model_path),
        static_cast<mmt::MnnBackendRequest>(backend_req),
        static_cast<int>(threads),
        jstring_to_std(env, cache_path),
        &err);
    if (!ok) {
        if (out_info != nullptr && env->GetArrayLength(out_info) >= 1) {
            env->SetObjectArrayElement(out_info, 0, env->NewStringUTF(err.c_str()));
        }
        delete sess;
        return 0L;
    }
    if (out_info != nullptr && env->GetArrayLength(out_info) >= 1) {
        env->SetObjectArrayElement(out_info, 0, env->NewStringUTF(sess->active_backend().c_str()));
    }
    return reinterpret_cast<jlong>(sess);
}

JNIEXPORT jboolean JNICALL
Java_dev_momotranscript_app_nativecore_NativeCore_nativeMnnDetRun(
    JNIEnv* env,
    jobject /* thiz */,
    jlong handle,
    jfloatArray input_nchw,
    jfloatArray out_bboxes,
    jfloatArray out_logits) {
    auto* sess = reinterpret_cast<mmt::MnnDetSession*>(handle);
    if (sess == nullptr || input_nchw == nullptr || out_bboxes == nullptr || out_logits == nullptr) {
        return JNI_FALSE;
    }
    jfloat* in_ptr = env->GetFloatArrayElements(input_nchw, nullptr);
    jfloat* b_ptr = env->GetFloatArrayElements(out_bboxes, nullptr);
    jfloat* l_ptr = env->GetFloatArrayElements(out_logits, nullptr);
    std::string err;
    const bool ok = sess->run_window(in_ptr, b_ptr, l_ptr, &err);
    env->ReleaseFloatArrayElements(input_nchw, in_ptr, JNI_ABORT);
    env->ReleaseFloatArrayElements(out_bboxes, b_ptr, ok ? 0 : JNI_ABORT);
    env->ReleaseFloatArrayElements(out_logits, l_ptr, ok ? 0 : JNI_ABORT);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_dev_momotranscript_app_nativecore_NativeCore_nativeMnnDetDestroy(
    JNIEnv* /* env */,
    jobject /* thiz */,
    jlong handle) {
    delete reinterpret_cast<mmt::MnnDetSession*>(handle);
}

JNIEXPORT jlong JNICALL
Java_dev_momotranscript_app_nativecore_NativeCore_nativeMnnRecCreate(
    JNIEnv* env,
    jobject /* thiz */,
    jstring model_path,
    jint backend_req,
    jint threads,
    jstring cache_path,
    jobjectArray out_info) {
    auto* sess = new mmt::MnnRecSession();
    std::string err;
    const bool ok = sess->load(
        jstring_to_std(env, model_path),
        static_cast<mmt::MnnBackendRequest>(backend_req),
        static_cast<int>(threads),
        jstring_to_std(env, cache_path),
        &err);
    if (!ok) {
        if (out_info != nullptr && env->GetArrayLength(out_info) >= 1) {
            env->SetObjectArrayElement(out_info, 0, env->NewStringUTF(err.c_str()));
        }
        delete sess;
        return 0L;
    }
    if (out_info != nullptr && env->GetArrayLength(out_info) >= 1) {
        env->SetObjectArrayElement(out_info, 0, env->NewStringUTF(sess->active_backend().c_str()));
    }
    return reinterpret_cast<jlong>(sess);
}

JNIEXPORT jint JNICALL
Java_dev_momotranscript_app_nativecore_NativeCore_nativeMnnRecRun(
    JNIEnv* env,
    jobject /* thiz */,
    jlong handle,
    jfloatArray input_nchw,
    jint batch_size,
    jint width,
    jobjectArray out_pair) {
    auto* sess = reinterpret_cast<mmt::MnnRecSession*>(handle);
    if (sess == nullptr || input_nchw == nullptr || out_pair == nullptr || env->GetArrayLength(out_pair) < 2) {
        return -1;
    }
    jfloat* in_ptr = env->GetFloatArrayElements(input_nchw, nullptr);
    std::vector<int32_t> idxs;
    std::vector<float> vals;
    int seq_len = 0;
    std::string err;
    const bool ok = sess->run_batch(in_ptr, static_cast<int>(batch_size), static_cast<int>(width), idxs, vals, seq_len, &err);
    env->ReleaseFloatArrayElements(input_nchw, in_ptr, JNI_ABORT);
    if (!ok || seq_len <= 0) {
        return -1;
    }
    jintArray jidx = env->NewIntArray(static_cast<jsize>(idxs.size()));
    jfloatArray jval = env->NewFloatArray(static_cast<jsize>(vals.size()));
    if (jidx != nullptr && !idxs.empty()) {
        env->SetIntArrayRegion(jidx, 0, static_cast<jsize>(idxs.size()), reinterpret_cast<const jint*>(idxs.data()));
    }
    if (jval != nullptr && !vals.empty()) {
        env->SetFloatArrayRegion(jval, 0, static_cast<jsize>(vals.size()), vals.data());
    }
    env->SetObjectArrayElement(out_pair, 0, jidx);
    env->SetObjectArrayElement(out_pair, 1, jval);
    return static_cast<jint>(seq_len);
}

JNIEXPORT void JNICALL
Java_dev_momotranscript_app_nativecore_NativeCore_nativeMnnRecDestroy(
    JNIEnv* /* env */,
    jobject /* thiz */,
    jlong handle) {
    delete reinterpret_cast<mmt::MnnRecSession*>(handle);
}

JNIEXPORT jlong JNICALL
Java_dev_momotranscript_app_nativecore_NativeCore_nativeMnnCcipCreate(
    JNIEnv* env,
    jobject /* thiz */,
    jstring model_path,
    jint backend_req,
    jint threads,
    jstring cache_path,
    jobjectArray out_info) {
    auto* sess = new mmt::MnnCcipSession();
    std::string err;
    const bool ok = sess->load(
        jstring_to_std(env, model_path),
        static_cast<mmt::MnnBackendRequest>(backend_req),
        static_cast<int>(threads),
        jstring_to_std(env, cache_path),
        &err);
    if (!ok) {
        if (out_info != nullptr && env->GetArrayLength(out_info) >= 1) {
            env->SetObjectArrayElement(out_info, 0, env->NewStringUTF(err.c_str()));
        }
        delete sess;
        return 0L;
    }
    if (out_info != nullptr) {
        const jsize len = env->GetArrayLength(out_info);
        if (len >= 1) env->SetObjectArrayElement(out_info, 0, env->NewStringUTF(sess->active_backend().c_str()));
        if (len >= 2) env->SetObjectArrayElement(out_info, 1, env->NewStringUTF(std::to_string(sess->input_size()).c_str()));
        if (len >= 3) env->SetObjectArrayElement(out_info, 2, env->NewStringUTF(std::to_string(sess->id_dim()).c_str()));
        if (len >= 4) env->SetObjectArrayElement(out_info, 3, env->NewStringUTF(std::to_string(sess->expr_dim()).c_str()));
    }
    return reinterpret_cast<jlong>(sess);
}

JNIEXPORT jboolean JNICALL
Java_dev_momotranscript_app_nativecore_NativeCore_nativeMnnCcipRun(
    JNIEnv* env,
    jobject /* thiz */,
    jlong handle,
    jfloatArray input_nchw,
    jfloatArray out_id,
    jfloatArray out_expr_logits) {
    auto* sess = reinterpret_cast<mmt::MnnCcipSession*>(handle);
    if (sess == nullptr || input_nchw == nullptr || out_id == nullptr) {
        return JNI_FALSE;
    }
    jfloat* in_ptr = env->GetFloatArrayElements(input_nchw, nullptr);
    jfloat* o_ptr = env->GetFloatArrayElements(out_id, nullptr);
    jfloat* e_ptr = out_expr_logits != nullptr ? env->GetFloatArrayElements(out_expr_logits, nullptr) : nullptr;
    std::string err;
    const bool ok = sess->run_single(in_ptr, o_ptr, e_ptr, &err);
    env->ReleaseFloatArrayElements(input_nchw, in_ptr, JNI_ABORT);
    env->ReleaseFloatArrayElements(out_id, o_ptr, ok ? 0 : JNI_ABORT);
    if (e_ptr != nullptr) {
        env->ReleaseFloatArrayElements(out_expr_logits, e_ptr, ok ? 0 : JNI_ABORT);
    }
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_dev_momotranscript_app_nativecore_NativeCore_nativeMnnCcipDestroy(
    JNIEnv* /* env */,
    jobject /* thiz */,
    jlong handle) {
    delete reinterpret_cast<mmt::MnnCcipSession*>(handle);
}

}  // extern "C"
