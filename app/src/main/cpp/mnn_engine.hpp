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
        const float* nchw_384,
        float* out_768,
        std::string* err_out = nullptr);

    const std::string& active_backend() const { return active_backend_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string active_backend_;
};

}  // namespace mmt
