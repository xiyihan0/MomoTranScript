#include "mnn_engine.hpp"

#include <MNN/Interpreter.hpp>
#include <MNN/MNNForwardType.h>
#include <MNN/Tensor.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>

#include <algorithm>
#include <cstring>
#include <sstream>

namespace mmt {
namespace {

MNNForwardType to_mnn_forward(MnnBackendRequest req) {
    switch (req) {
        case MnnBackendRequest::kOpenCl: return MNN_FORWARD_OPENCL;
        case MnnBackendRequest::kVulkan: return MNN_FORWARD_VULKAN;
        case MnnBackendRequest::kCpuFp16:
        default: return MNN_FORWARD_CPU;
    }
}

std::string forward_label(MNNForwardType ft, MNN::BackendConfig::PrecisionMode prec) {
    const char* ptag = (prec == MNN::BackendConfig::Precision_High) ? "FP32" : "FP16";
    switch (ft) {
        case MNN_FORWARD_OPENCL: return std::string("OpenCL(") + ptag + ")";
        case MNN_FORWARD_VULKAN: return std::string("Vulkan(") + ptag + ")";
        case MNN_FORWARD_CPU_EXTENSION: return std::string("CPU(ARMv8.2-") + ptag + ")";
        case MNN_FORWARD_CPU:
        default: return std::string("CPU(") + ptag + ")";
    }
}

void fill_schedule_config(
    MNN::ScheduleConfig& scfg,
    MNN::BackendConfig& bcfg,
    MNNForwardType ftype,
    int threads,
    MNN::BackendConfig::PrecisionMode prec = MNN::BackendConfig::Precision_Low) {
    bcfg.memory = MNN::BackendConfig::Memory_Normal;
    bcfg.power = MNN::BackendConfig::Power_High;
    bcfg.precision = prec;

    scfg.type = ftype;
    scfg.backupType = MNN_FORWARD_CPU;
    scfg.backendConfig = &bcfg;
    if (ftype == MNN_FORWARD_OPENCL) {
        scfg.mode = MNN_GPU_TUNING_FAST | MNN_GPU_MEMORY_BUFFER;
    } else if (ftype == MNN_FORWARD_VULKAN) {
        scfg.mode = MNN_GPU_TUNING_WIDE | MNN_GPU_RECORD_BATCH;
    } else {
        scfg.numThread = std::max(1, threads);
    }
}

std::string query_session_backend_label(
    MNN::Interpreter* net,
    MNN::Session* sess,
    MNNForwardType req,
    MNN::BackendConfig::PrecisionMode prec) {
    int backends[2] = {0, 0};
    if (net->getSessionInfo(sess, MNN::Interpreter::BACKENDS, backends)) {
        auto primary = static_cast<MNNForwardType>(backends[0]);
        bool matched = (primary == req) ||
                       (req == MNN_FORWARD_CPU && primary == MNN_FORWARD_CPU_EXTENSION);
        if (matched) return forward_label(primary, prec);
        return forward_label(primary, prec) + "(fb)";
    }
    return forward_label(req, prec);
}

}  // namespace

std::string mnn_probe_device_info() {
    std::ostringstream oss;
    oss << "MNN " << (MNN::getVersion() ? MNN::getVersion() : "3.6.2");

    MNN::ScheduleConfig probe_cfg;
    probe_cfg.type = MNN_FORWARD_CPU;
    std::shared_ptr<MNN::Express::Executor::RuntimeManager> probe_mgr(
        MNN::Express::Executor::RuntimeManager::createRuntimeManager(probe_cfg),
        MNN::Express::Executor::RuntimeManager::destroy);
    if (probe_mgr) {
        auto sup = probe_mgr->isBackendSupport({MNN_FORWARD_OPENCL, MNN_FORWARD_VULKAN});
        oss << " | OpenCL=" << ((sup.size() > 0 && sup[0]) ? "yes" : "no");
        oss << " | Vulkan=" << ((sup.size() > 1 && sup[1]) ? "yes" : "no");
    } else {
        oss << " | OpenCL=? | Vulkan=?";
    }

    auto exec = MNN::Express::Executor::getGlobalExecutor();
    if (exec) {
        int fp16 = exec->getCurrentRuntimeStatus(MNN::STATUS_SUPPORT_FP16);
        int dot = exec->getCurrentRuntimeStatus(MNN::STATUS_SUPPORT_DOT_PRODUCT);
        oss << " | CPU_FP16=" << fp16 << " DOT=" << dot;
    }
    return oss.str();
}

// ============================= MnnDetSession =============================

struct MnnDetSession::Impl {
    MNN::Interpreter* net = nullptr;
    MNN::Session* session = nullptr;
    MNN::Tensor* input = nullptr;
    MNN::Tensor* out_bboxes = nullptr;
    MNN::Tensor* out_logits = nullptr;
    bool use_gpu_cache = false;
    bool cache_updated = false;

    ~Impl() {
        if (net != nullptr) {
            if (session != nullptr) {
                net->releaseSession(session);
            }
            MNN::Interpreter::destroy(net);
        }
    }
};

MnnDetSession::MnnDetSession() = default;
MnnDetSession::~MnnDetSession() = default;

bool MnnDetSession::load(
    const std::string& model_path,
    MnnBackendRequest req_backend,
    int threads,
    const std::string& cache_path,
    std::string* err_out) {
    impl_ = std::make_unique<Impl>();
    impl_->net = MNN::Interpreter::createFromFile(model_path.c_str());
    if (impl_->net == nullptr) {
        if (err_out) *err_out = "MNN 无法加载检测模型: " + model_path;
        impl_.reset();
        return false;
    }

    impl_->net->setSessionMode(MNN::Interpreter::Session_Release);
    impl_->net->setSessionMode(MNN::Interpreter::Session_Memory_Collect);
    impl_->net->setSessionHint(MNN::Interpreter::WINOGRAD_MEMORY_LEVEL, 1);

    MNNForwardType ftype = to_mnn_forward(req_backend);
    if ((ftype == MNN_FORWARD_OPENCL || ftype == MNN_FORWARD_VULKAN) && !cache_path.empty()) {
        impl_->net->setCacheFile(cache_path.c_str());
        impl_->use_gpu_cache = true;
    }

    MNN::ScheduleConfig scfg;
    MNN::BackendConfig bcfg;
    fill_schedule_config(scfg, bcfg, ftype, threads);

    impl_->session = impl_->net->createSession(scfg);
    if (impl_->session == nullptr && ftype != MNN_FORWARD_CPU) {
        // Automatic fallback to ARMv9 CPU FP16
        ftype = MNN_FORWARD_CPU;
        impl_->use_gpu_cache = false;
        fill_schedule_config(scfg, bcfg, ftype, threads);
        impl_->session = impl_->net->createSession(scfg);
    }
    if (impl_->session == nullptr) {
        if (err_out) *err_out = "MNN 创建检测 Session 失败";
        impl_.reset();
        return false;
    }

    impl_->input = impl_->net->getSessionInput(impl_->session, nullptr);
    if (impl_->input == nullptr) {
        if (err_out) *err_out = "MNN 检测模型缺少输入张量";
        impl_.reset();
        return false;
    }

    impl_->net->resizeTensor(impl_->input, {1, 3, 672, 672});
    impl_->net->resizeSession(impl_->session);

    auto outs = impl_->net->getSessionOutputAll(impl_->session);
    for (const auto& kv : outs) {
        auto sh = kv.second->shape();
        if (sh.size() == 3 && sh[0] == 1 && sh[1] == 300 && sh[2] == 4) {
            impl_->out_bboxes = kv.second;
        } else if (sh.size() == 3 && sh[0] == 1 && sh[1] == 300 && sh[2] == 11) {
            impl_->out_logits = kv.second;
        }
    }
    if (impl_->out_bboxes == nullptr || impl_->out_logits == nullptr) {
        if (err_out) *err_out = "MNN 检测模型输出张量形状不匹配 (期望 1x300x4 和 1x300x11)";
        impl_.reset();
        return false;
    }

    active_backend_ = query_session_backend_label(
        impl_->net, impl_->session, ftype, MNN::BackendConfig::Precision_Low);
    return true;
}

bool MnnDetSession::run_window(
    const float* nchw_672,
    float* out_bboxes_1200,
    float* out_logits_3300,
    std::string* err_out) {
    if (!impl_ || !impl_->session) {
        if (err_out) *err_out = "MnnDetSession 未初始化";
        return false;
    }

    std::unique_ptr<MNN::Tensor> host_in(
        MNN::Tensor::create({1, 3, 672, 672}, halide_type_of<float>(), const_cast<float*>(nchw_672), MNN::Tensor::CAFFE));
    impl_->input->copyFromHostTensor(host_in.get());

    auto err = impl_->net->runSession(impl_->session);
    if (err != MNN::NO_ERROR) {
        if (err_out) *err_out = "MNN 检测推理失败, code=" + std::to_string(static_cast<int>(err));
        return false;
    }

    if (impl_->use_gpu_cache && !impl_->cache_updated) {
        impl_->net->updateCacheFile(impl_->session);
        impl_->cache_updated = true;
    }

    std::unique_ptr<MNN::Tensor> host_b(
        MNN::Tensor::create({1, 300, 4}, halide_type_of<float>(), out_bboxes_1200, MNN::Tensor::CAFFE));
    std::unique_ptr<MNN::Tensor> host_l(
        MNN::Tensor::create({1, 300, 11}, halide_type_of<float>(), out_logits_3300, MNN::Tensor::CAFFE));
    impl_->out_bboxes->copyToHostTensor(host_b.get());
    impl_->out_logits->copyToHostTensor(host_l.get());
    return true;
}

// ============================= MnnRecSession =============================

struct MnnRecSession::Impl {
    MNN::Interpreter* net = nullptr;
    MNN::Session* session = nullptr;
    MNN::Tensor* input = nullptr;
    MNN::Tensor* out_idx = nullptr;
    MNN::Tensor* out_val = nullptr;
    int last_batch = -1;
    int last_width = -1;

    ~Impl() {
        if (net != nullptr) {
            if (session != nullptr) {
                net->releaseSession(session);
            }
            MNN::Interpreter::destroy(net);
        }
    }
};

MnnRecSession::MnnRecSession() = default;
MnnRecSession::~MnnRecSession() = default;

bool MnnRecSession::load(
    const std::string& model_path,
    MnnBackendRequest req_backend,
    int threads,
    const std::string& cache_path,
    std::string* err_out) {
    impl_ = std::make_unique<Impl>();
    impl_->net = MNN::Interpreter::createFromFile(model_path.c_str());
    if (impl_->net == nullptr) {
        if (err_out) *err_out = "MNN 无法加载识别模型: " + model_path;
        impl_.reset();
        return false;
    }

    impl_->net->setSessionMode(MNN::Interpreter::Session_Release);
    impl_->net->setSessionMode(MNN::Interpreter::Session_Memory_Collect);

    MNNForwardType ftype = to_mnn_forward(req_backend);
    if ((ftype == MNN_FORWARD_OPENCL || ftype == MNN_FORWARD_VULKAN) && !cache_path.empty()) {
        impl_->net->setCacheFile(cache_path.c_str());
    }

    MNN::ScheduleConfig scfg;
    MNN::BackendConfig bcfg;
    fill_schedule_config(scfg, bcfg, ftype, threads);

    impl_->session = impl_->net->createSession(scfg);
    if (impl_->session == nullptr && ftype != MNN_FORWARD_CPU) {
        ftype = MNN_FORWARD_CPU;
        fill_schedule_config(scfg, bcfg, ftype, threads);
        impl_->session = impl_->net->createSession(scfg);
    }
    if (impl_->session == nullptr) {
        if (err_out) *err_out = "MNN 创建识别 Session 失败";
        impl_.reset();
        return false;
    }

    impl_->input = impl_->net->getSessionInput(impl_->session, nullptr);
    impl_->out_idx = impl_->net->getSessionOutput(impl_->session, "argmax_idx");
    impl_->out_val = impl_->net->getSessionOutput(impl_->session, "argmax_val");
    if (impl_->input == nullptr || impl_->out_idx == nullptr || impl_->out_val == nullptr) {
        if (err_out) *err_out = "MNN 识别模型缺少输入或 argmax_idx/argmax_val 输出";
        impl_.reset();
        return false;
    }

    active_backend_ = query_session_backend_label(
        impl_->net, impl_->session, ftype, MNN::BackendConfig::Precision_Low);
    return true;
}

bool MnnRecSession::run_batch(
    const float* nchw_batch,
    int batch_size,
    int width,
    std::vector<int32_t>& out_idx,
    std::vector<float>& out_val,
    int& out_seq_len,
    std::string* err_out) {
    if (!impl_ || !impl_->session || batch_size <= 0 || width <= 0) {
        if (err_out) *err_out = "MnnRecSession 参数无效或未初始化";
        return false;
    }

    if (batch_size != impl_->last_batch || width != impl_->last_width) {
        impl_->net->resizeTensor(impl_->input, {batch_size, 3, 48, width});
        impl_->net->resizeSession(impl_->session);
        impl_->last_batch = batch_size;
        impl_->last_width = width;
    }

    std::unique_ptr<MNN::Tensor> host_in(
        MNN::Tensor::create({batch_size, 3, 48, width}, halide_type_of<float>(), const_cast<float*>(nchw_batch), MNN::Tensor::CAFFE));
    impl_->input->copyFromHostTensor(host_in.get());

    auto err = impl_->net->runSession(impl_->session);
    if (err != MNN::NO_ERROR) {
        if (err_out) *err_out = "MNN 识别推理失败, code=" + std::to_string(static_cast<int>(err));
        return false;
    }

    auto sh = impl_->out_idx->shape();
    if (sh.size() < 2 || sh[0] != batch_size || sh[1] <= 0) {
        if (err_out) *err_out = "MNN 识别输出形状异常";
        return false;
    }
    out_seq_len = sh[1];
    const size_t total = static_cast<size_t>(batch_size) * static_cast<size_t>(out_seq_len);
    out_idx.resize(total);
    out_val.resize(total);

    std::unique_ptr<MNN::Tensor> host_idx(
        MNN::Tensor::create(sh, halide_type_of<int32_t>(), out_idx.data(), MNN::Tensor::CAFFE));
    std::unique_ptr<MNN::Tensor> host_val(
        MNN::Tensor::create(sh, halide_type_of<float>(), out_val.data(), MNN::Tensor::CAFFE));
    impl_->out_idx->copyToHostTensor(host_idx.get());
    impl_->out_val->copyToHostTensor(host_val.get());
    return true;
}

// ============================= MnnCcipSession =============================
// Uses static 76-LayerNorm-fused `ccip_caformer_24_ln_w8.mnn` (subgraphs=0).
// Every fused MNN LayerNorm kernel computes mean/variance in FP32 internally even
// under Precision_Low (OpenCL/ARMv8.2 FP16), eliminating the Pow(x-mean,2)=1.17e7
// FP16 overflow while running all 188 MatMuls & 152 Convs at full FP16 GPU speed!

struct MnnCcipSession::Impl {
    MNN::Interpreter* net = nullptr;
    MNN::Session* session = nullptr;
    MNN::Tensor* input = nullptr;
    MNN::Tensor* output = nullptr;
    bool use_gpu_cache = false;
    bool cache_updated = false;

    ~Impl() {
        if (net != nullptr) {
            if (session != nullptr) {
                net->releaseSession(session);
            }
            MNN::Interpreter::destroy(net);
        }
    }
};

MnnCcipSession::MnnCcipSession() = default;
MnnCcipSession::~MnnCcipSession() = default;

bool MnnCcipSession::load(
    const std::string& model_path,
    MnnBackendRequest req_backend,
    int threads,
    const std::string& cache_path,
    std::string* err_out) {
    impl_ = std::make_unique<Impl>();
    impl_->net = MNN::Interpreter::createFromFile(model_path.c_str());
    if (impl_->net == nullptr) {
        if (err_out) *err_out = "MNN 无法加载 CCIP 模型: " + model_path;
        impl_.reset();
        return false;
    }

    impl_->net->setSessionMode(MNN::Interpreter::Session_Release);
    impl_->net->setSessionMode(MNN::Interpreter::Session_Memory_Collect);

    MNNForwardType ftype = to_mnn_forward(req_backend);
    if ((ftype == MNN_FORWARD_OPENCL || ftype == MNN_FORWARD_VULKAN) && !cache_path.empty()) {
        impl_->net->setCacheFile(cache_path.c_str());
        impl_->use_gpu_cache = true;
    }

    MNN::ScheduleConfig scfg;
    MNN::BackendConfig bcfg;
    fill_schedule_config(scfg, bcfg, ftype, threads, MNN::BackendConfig::Precision_Low);

    impl_->session = impl_->net->createSession(scfg);
    if (impl_->session == nullptr && ftype != MNN_FORWARD_CPU) {
        ftype = MNN_FORWARD_CPU;
        impl_->use_gpu_cache = false;
        fill_schedule_config(scfg, bcfg, ftype, threads, MNN::BackendConfig::Precision_Low);
        impl_->session = impl_->net->createSession(scfg);
    }
    if (impl_->session == nullptr) {
        if (err_out) *err_out = "MNN 创建 CCIP Session 失败";
        impl_.reset();
        return false;
    }

    impl_->input = impl_->net->getSessionInput(impl_->session, nullptr);
    impl_->output = impl_->net->getSessionOutput(impl_->session, nullptr);
    if (impl_->input == nullptr || impl_->output == nullptr) {
        if (err_out) *err_out = "MNN CCIP 模型缺少输入或输出张量";
        impl_.reset();
        return false;
    }

    impl_->net->resizeTensor(impl_->input, {1, 3, 384, 384});
    impl_->net->resizeSession(impl_->session);

    active_backend_ = query_session_backend_label(
        impl_->net, impl_->session, ftype, MNN::BackendConfig::Precision_Low);
    return true;
}

bool MnnCcipSession::run_single(
    const float* nchw_384,
    float* out_768,
    std::string* err_out) {
    if (!impl_ || !impl_->session) {
        if (err_out) *err_out = "MnnCcipSession 未初始化";
        return false;
    }

    std::unique_ptr<MNN::Tensor> host_in(
        MNN::Tensor::create({1, 3, 384, 384}, halide_type_of<float>(), const_cast<float*>(nchw_384), MNN::Tensor::CAFFE));
    impl_->input->copyFromHostTensor(host_in.get());

    auto err = impl_->net->runSession(impl_->session);
    if (err != MNN::NO_ERROR) {
        if (err_out) *err_out = "MNN CCIP 推理失败, code=" + std::to_string(static_cast<int>(err));
        return false;
    }

    if (impl_->use_gpu_cache && !impl_->cache_updated) {
        impl_->net->updateCacheFile(impl_->session);
        impl_->cache_updated = true;
    }

    auto sh = impl_->output->shape();
    int total = 1;
    for (int d : sh) total *= d;
    if (total < 768) {
        if (err_out) *err_out = "MNN CCIP 输出维度异常";
        return false;
    }
    std::vector<float> tmp(static_cast<size_t>(total));
    std::unique_ptr<MNN::Tensor> host_out(
        MNN::Tensor::create(sh, halide_type_of<float>(), tmp.data(), MNN::Tensor::CAFFE));
    impl_->output->copyToHostTensor(host_out.get());
    std::memcpy(out_768, tmp.data(), 768 * sizeof(float));
    return true;
}

}  // namespace mmt
