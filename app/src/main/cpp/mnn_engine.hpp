#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mmt {

enum class MnnBackendRequest : int {
    kCpuFp16 = 0,
    kOpenCl = 1,
    kVulkan = 2,
};

struct MnnSessionMeta {
    std::string active_backend;
    std::string device_info;
};

/// Query MNN version & available GPU/CPU hardware capabilities on this Android device.
std::string mnn_probe_device_info();

class MnnDetSession {
public:
    MnnDetSession();
    ~MnnDetSession();

    bool load(
        const std::string& model_path,
        MnnBackendRequest req_backend,
        int threads,
        const std::string& cache_path,
        std::string* err_out = nullptr);

    bool run_window(
        const float* nchw_672,
        float* out_bboxes_1200,
        float* out_logits_3300,
        std::string* err_out = nullptr);

    const std::string& active_backend() const { return active_backend_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string active_backend_;
};

class MnnRecSession {
public:
    MnnRecSession();
    ~MnnRecSession();

    bool load(
        const std::string& model_path,
        MnnBackendRequest req_backend,
        int threads,
        const std::string& cache_path,
        std::string* err_out = nullptr);

    bool run_batch(
        const float* nchw_batch,
        int batch_size,
        int width,
        std::vector<int32_t>& out_idx,
        std::vector<float>& out_val,
        int& out_seq_len,
        std::string* err_out = nullptr);

    const std::string& active_backend() const { return active_backend_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string active_backend_;
};

class MnnCcipSession {
public:
    MnnCcipSession();
    ~MnnCcipSession();

    bool load(
        const std::string& model_path,
        MnnBackendRequest req_backend,
        int threads,
        const std::string& cache_path,
        std::string* err_out = nullptr);

    bool run_single(
        const float* nchw,
        float* out_id,
        float* out_expr_logits = nullptr,
        std::string* err_out = nullptr);

    const std::string& active_backend() const { return active_backend_; }
    int input_size() const { return input_size_; }
    int id_dim() const { return id_dim_; }
    int expr_dim() const { return expr_dim_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string active_backend_;
    int input_size_ = 384;
    int id_dim_ = 768;
    int expr_dim_ = 0;
};

}  // namespace mmt
