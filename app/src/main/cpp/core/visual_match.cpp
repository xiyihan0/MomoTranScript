#include "visual_match.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace mmt {
namespace {

constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};

inline uint8_t rgb_to_gray(uint8_t r, uint8_t g, uint8_t b) {
    return static_cast<uint8_t>((77 * r + 150 * g + 29 * b + 128) >> 8);
}

inline float sample_bilinear_u8(
    const uint8_t* src,
    int sw,
    int sh,
    float sx,
    float sy,
    int c) {
    sx = std::clamp(sx, 0.0f, static_cast<float>(sw - 1));
    sy = std::clamp(sy, 0.0f, static_cast<float>(sh - 1));
    int x0 = static_cast<int>(sx);
    int y0 = static_cast<int>(sy);
    int x1 = std::min(x0 + 1, sw - 1);
    int y1 = std::min(y0 + 1, sh - 1);
    float dx = sx - static_cast<float>(x0);
    float dy = sy - static_cast<float>(y0);

    const uint8_t* r0 = src + static_cast<size_t>(y0) * sw * 3;
    const uint8_t* r1 = src + static_cast<size_t>(y1) * sw * 3;
    float v00 = static_cast<float>(r0[x0 * 3 + c]);
    float v10 = static_cast<float>(r0[x1 * 3 + c]);
    float v01 = static_cast<float>(r1[x0 * 3 + c]);
    float v11 = static_cast<float>(r1[x1 * 3 + c]);
    float top = v00 + (v10 - v00) * dx;
    float bot = v01 + (v11 - v01) * dx;
    return top + (bot - top) * dy;
}

void extract_crop_rgb(
    const uint8_t* rgb,
    int img_w,
    int img_h,
    const BoxI& box,
    std::vector<uint8_t>& out,
    int& cw,
    int& ch) {
    int x0 = std::clamp(box.x0, 0, img_w);
    int y0 = std::clamp(box.y0, 0, img_h);
    int x1 = std::clamp(box.x1, 0, img_w);
    int y1 = std::clamp(box.y1, 0, img_h);
    cw = std::max(1, x1 - x0);
    ch = std::max(1, y1 - y0);
    out.resize(static_cast<size_t>(cw) * ch * 3);
    if (x1 <= x0 || y1 <= y0) {
        std::fill(out.begin(), out.end(), 255);
        return;
    }
    for (int y = 0; y < ch; ++y) {
        const uint8_t* src = rgb + (static_cast<size_t>(y0 + y) * img_w + x0) * 3;
        std::copy_n(src, cw * 3, out.data() + static_cast<size_t>(y) * cw * 3);
    }
}

void trim_light_border_inplace(std::vector<uint8_t>& crop, int& w, int& h) {
    if (w < 16 || h < 16) return;
    auto is_row_pad = [&](int y, int x0, int x1) -> bool {
        int white_cnt = 0;
        int len = std::max(1, x1 - x0);
        const uint8_t* row = crop.data() + static_cast<size_t>(y) * w * 3;
        for (int x = x0; x < x1; ++x) {
            uint8_t mn = std::min({row[x * 3], row[x * 3 + 1], row[x * 3 + 2]});
            if (mn >= 242) white_cnt++;
        }
        return static_cast<float>(white_cnt) / static_cast<float>(len) >= 0.96f;
    };
    auto is_col_pad = [&](int x, int y0, int y1) -> bool {
        int white_cnt = 0;
        int len = std::max(1, y1 - y0);
        for (int y = y0; y < y1; ++y) {
            const uint8_t* p = crop.data() + (static_cast<size_t>(y) * w + x) * 3;
            uint8_t mn = std::min({p[0], p[1], p[2]});
            if (mn >= 242) white_cnt++;
        }
        return static_cast<float>(white_cnt) / static_cast<float>(len) >= 0.96f;
    };

    int y0 = 0, y1 = h, x0 = 0, x1 = w;
    while (y0 < y1 - 16 && is_row_pad(y0, x0, x1)) ++y0;
    while (y1 > y0 + 16 && is_row_pad(y1 - 1, x0, x1)) --y1;
    while (x0 < x1 - 16 && is_col_pad(x0, y0, y1)) ++x0;
    while (x1 > x0 + 16 && is_col_pad(x1 - 1, y0, y1)) --x1;

    if (x0 == 0 && y0 == 0 && x1 == w && y1 == h) return;
    int nw = x1 - x0;
    int nh = y1 - y0;
    std::vector<uint8_t> trimmed(static_cast<size_t>(nw) * nh * 3);
    for (int y = 0; y < nh; ++y) {
        const uint8_t* src = crop.data() + (static_cast<size_t>(y0 + y) * w + x0) * 3;
        std::copy_n(src, nw * 3, trimmed.data() + static_cast<size_t>(y) * nw * 3);
    }
    crop = std::move(trimmed);
    w = nw;
    h = nh;
}

void center_square_inplace(std::vector<uint8_t>& crop, int& w, int& h, bool circle_mask) {
    int side = std::min(w, h);
    int x0 = (w - side) / 2;
    int y0 = (h - side) / 2;
    std::vector<uint8_t> sq(static_cast<size_t>(side) * side * 3, 255);
    const float c = static_cast<float>(side - 1) * 0.5f;
    const float r2 = (static_cast<float>(side) * 0.47f) * (static_cast<float>(side) * 0.47f);

    for (int y = 0; y < side; ++y) {
        const float dy = static_cast<float>(y) - c;
        const uint8_t* src = crop.data() + (static_cast<size_t>(y0 + y) * w + x0) * 3;
        uint8_t* dst = sq.data() + static_cast<size_t>(y) * side * 3;
        for (int x = 0; x < side; ++x) {
            if (circle_mask) {
                const float dx = static_cast<float>(x) - c;
                if (dx * dx + dy * dy > r2) continue;
            }
            dst[x * 3 + 0] = src[x * 3 + 0];
            dst[x * 3 + 1] = src[x * 3 + 1];
            dst[x * 3 + 2] = src[x * 3 + 2];
        }
    }
    crop = std::move(sq);
    w = side;
    h = side;
}

float compute_rgb_std(const std::vector<uint8_t>& rgb) {
    if (rgb.empty()) return 0.0f;
    double sum = 0.0;
    double sum_sq = 0.0;
    for (uint8_t v : rgb) {
        double d = static_cast<double>(v);
        sum += d;
        sum_sq += d * d;
    }
    double n = static_cast<double>(rgb.size());
    double mean = sum / n;
    double var = std::max(0.0, sum_sq / n - mean * mean);
    return static_cast<float>(std::sqrt(var));
}

void resize_rgb_float(
    const uint8_t* src,
    int sw,
    int sh,
    int dw,
    int dh,
    std::vector<float>& dst_rgb) {
    dst_rgb.resize(static_cast<size_t>(dw) * dh * 3);
    const float sx_scale = static_cast<float>(sw) / static_cast<float>(dw);
    const float sy_scale = static_cast<float>(sh) / static_cast<float>(dh);
    for (int y = 0; y < dh; ++y) {
        float sy = (static_cast<float>(y) + 0.5f) * sy_scale - 0.5f;
        for (int x = 0; x < dw; ++x) {
            float sx = (static_cast<float>(x) + 0.5f) * sx_scale - 0.5f;
            size_t off = (static_cast<size_t>(y) * dw + x) * 3;
            dst_rgb[off + 0] = sample_bilinear_u8(src, sw, sh, sx, sy, 0);
            dst_rgb[off + 1] = sample_bilinear_u8(src, sw, sh, sx, sy, 1);
            dst_rgb[off + 2] = sample_bilinear_u8(src, sw, sh, sx, sy, 2);
        }
    }
}

void scale_and_shift_rgb(
    const std::vector<float>& src,
    int sz,
    float scale,
    int dx,
    int dy,
    std::vector<float>& dst) {
    dst.resize(static_cast<size_t>(sz) * sz * 3);
    const float half = static_cast<float>(sz - 1) * 0.5f;
    const float inv_s = 1.0f / scale;
    for (int y = 0; y < sz; ++y) {
        float sy = (static_cast<float>(y - dy) - half) * inv_s + half;
        sy = std::clamp(sy, 0.0f, static_cast<float>(sz - 1));
        int y0 = static_cast<int>(sy);
        int y1 = std::min(y0 + 1, sz - 1);
        float fy = sy - static_cast<float>(y0);
        for (int x = 0; x < sz; ++x) {
            float sx = (static_cast<float>(x - dx) - half) * inv_s + half;
            sx = std::clamp(sx, 0.0f, static_cast<float>(sz - 1));
            int x0 = static_cast<int>(sx);
            int x1 = std::min(x0 + 1, sz - 1);
            float fx = sx - static_cast<float>(x0);
            size_t doff = (static_cast<size_t>(y) * sz + x) * 3;
            for (int c = 0; c < 3; ++c) {
                float v00 = src[(static_cast<size_t>(y0) * sz + x0) * 3 + c];
                float v10 = src[(static_cast<size_t>(y0) * sz + x1) * 3 + c];
                float v01 = src[(static_cast<size_t>(y1) * sz + x0) * 3 + c];
                float v11 = src[(static_cast<size_t>(y1) * sz + x1) * 3 + c];
                float top = v00 + (v10 - v00) * fx;
                float bot = v01 + (v11 - v01) * fx;
                dst[doff + c] = top + (bot - top) * fy;
            }
        }
    }
}

}  // namespace

std::vector<BoxI> split_avatar_box(const BoxI& box, int max_members) {
    const int w = box.width();
    const int h = box.height();
    if (w <= 0 || h <= 0) return {};
    const float aspect = static_cast<float>(w) / static_cast<float>(h);
    if (aspect >= 1.0f / 1.35f && aspect <= 1.35f) {
        return {box};
    }
    std::vector<BoxI> out;
    if (aspect > 1.35f) {
        int count = std::clamp(static_cast<int>(std::lround(aspect * 1.15f)), 2, max_members);
        float step = static_cast<float>(w - h) / static_cast<float>(std::max(1, count - 1));
        for (int i = 0; i < count; ++i) {
            int sx = static_cast<int>(std::lround(box.x0 + i * step));
            out.push_back(BoxI{sx, box.y0, sx + h, box.y1});
        }
    } else {
        int count = std::clamp(static_cast<int>(std::lround((static_cast<float>(h) / static_cast<float>(w)) * 1.15f)), 2, max_members);
        float step = static_cast<float>(h - w) / static_cast<float>(std::max(1, count - 1));
        for (int i = 0; i < count; ++i) {
            int sy = static_cast<int>(std::lround(box.y0 + i * step));
            out.push_back(BoxI{box.x0, sy, box.x1, sy + w});
        }
    }
    return out;
}

void prepare_ccip_input_nchw(
    const uint8_t* rgb,
    int img_w,
    int img_h,
    const BoxI& box,
    bool circle_mask,
    bool trim_borders,
    int dst_size,
    float* out_nchw,
    float* out_std,
    float* out_aspect) {
    std::vector<uint8_t> crop;
    int cw = 0, ch = 0;
    extract_crop_rgb(rgb, img_w, img_h, box, crop, cw, ch);
    if (trim_borders) {
        trim_light_border_inplace(crop, cw, ch);
    }
    if (out_aspect != nullptr) {
        *out_aspect = static_cast<float>(cw) / static_cast<float>(std::max(1, ch));
    }
    center_square_inplace(crop, cw, ch, circle_mask);
    if (out_std != nullptr) {
        *out_std = compute_rgb_std(crop);
    }

    const size_t plane = static_cast<size_t>(dst_size) * static_cast<size_t>(dst_size);
    const float sx_scale = static_cast<float>(cw) / static_cast<float>(dst_size);
    const float sy_scale = static_cast<float>(ch) / static_cast<float>(dst_size);

    for (int dy = 0; dy < dst_size; ++dy) {
        float sy = (static_cast<float>(dy) + 0.5f) * sy_scale - 0.5f;
        size_t row = static_cast<size_t>(dy) * dst_size;
        for (int dx = 0; dx < dst_size; ++dx) {
            float sx = (static_cast<float>(dx) + 0.5f) * sx_scale - 0.5f;
            for (int c = 0; c < 3; ++c) {
                float v = sample_bilinear_u8(crop.data(), cw, ch, sx, sy, c) * (1.0f / 255.0f);
                out_nchw[c * plane + row + dx] = (v - kMean[c]) / kStd[c];
            }
        }
    }
}

void compute_rgb_thumbnail(
    const uint8_t* rgb,
    int img_w,
    int img_h,
    const BoxI& box,
    bool circle_mask,
    int thumb_size,
    float* out_thumb_rgb) {
    std::vector<uint8_t> crop;
    int cw = 0, ch = 0;
    extract_crop_rgb(rgb, img_w, img_h, box, crop, cw, ch);
    center_square_inplace(crop, cw, ch, circle_mask);
    std::vector<float> tmp;
    resize_rgb_float(crop.data(), cw, ch, thumb_size, thumb_size, tmp);
    std::copy(tmp.begin(), tmp.end(), out_thumb_rgb);
}

Stage2SkinScore score_stage2_skin_templates(
    const uint8_t* rgb,
    int img_w,
    int img_h,
    const BoxI& box,
    const uint8_t* stack_rgb,
    int count,
    int tpl_size,
    const float* var_weights) {
    Stage2SkinScore result;
    if (rgb == nullptr || stack_rgb == nullptr || var_weights == nullptr || count <= 0 || tpl_size <= 0) {
        return result;
    }

    std::vector<uint8_t> crop;
    int cw = 0, ch = 0;
    extract_crop_rgb(rgb, img_w, img_h, box, crop, cw, ch);
    center_square_inplace(crop, cw, ch, false);

    std::vector<float> q_rs;
    resize_rgb_float(crop.data(), cw, ch, tpl_size, tpl_size, q_rs);

    const size_t npix = static_cast<size_t>(tpl_size) * static_cast<size_t>(tpl_size);
    std::vector<float> mean_g(npix, 0.0f);
    for (int t = 0; t < count; ++t) {
        const uint8_t* tpl = stack_rgb + static_cast<size_t>(t) * npix * 3;
        for (size_t i = 0; i < npix; ++i) {
            mean_g[i] += static_cast<float>(rgb_to_gray(tpl[i * 3], tpl[i * 3 + 1], tpl[i * 3 + 2]));
        }
    }
    const float inv_count = 1.0f / static_cast<float>(count);
    for (size_t i = 0; i < npix; ++i) mean_g[i] *= inv_count;

    float vmax = 0.0f;
    for (size_t i = 0; i < npix; ++i) {
        if (var_weights[i] > vmax) vmax = var_weights[i];
    }
    const int border = std::max(3, tpl_size / 16);
    std::vector<float> stable(npix, 0.0f);
    float s_sum = 0.0f;
    const float low_var_thr = std::max(0.08f, vmax * 0.20f);
    for (int y = border; y < tpl_size - border; ++y) {
        for (int x = border; x < tpl_size - border; ++x) {
            size_t idx = static_cast<size_t>(y) * tpl_size + x;
            if (var_weights[idx] > 0.0f && var_weights[idx] < low_var_thr) {
                stable[idx] = 1.0f;
                s_sum += 1.0f;
            }
        }
    }
    if (s_sum < 30.0f) {
        s_sum = 0.0f;
        std::fill(stable.begin(), stable.end(), 0.0f);
        for (int y = border; y < tpl_size - border; ++y) {
            for (int x = border; x < tpl_size - border; ++x) {
                size_t idx = static_cast<size_t>(y) * tpl_size + x;
                if (var_weights[idx] > 0.0f) {
                    stable[idx] = 1.0f;
                    s_sum += 1.0f;
                }
            }
        }
    }
    if (s_sum < 30.0f) {
        s_sum = 0.0f;
        std::fill(stable.begin(), stable.end(), 0.0f);
        for (int y = border; y < tpl_size - border; ++y) {
            for (int x = border; x < tpl_size - border; ++x) {
                stable[static_cast<size_t>(y) * tpl_size + x] = 1.0f;
                s_sum += 1.0f;
            }
        }
    }

    double m_mean = 0.0;
    for (size_t i = 0; i < npix; ++i) m_mean += mean_g[i] * stable[i];
    m_mean /= std::max(1.0f, s_sum);
    std::vector<float> m_z(npix, 0.0f);
    double m_norm_sq = 0.0;
    for (size_t i = 0; i < npix; ++i) {
        float v = static_cast<float>((mean_g[i] - m_mean) * stable[i]);
        m_z[i] = v;
        m_norm_sq += static_cast<double>(v) * v;
    }
    const double m_norm = std::sqrt(m_norm_sq) + 1e-6;

    float best_sc = -1e9f;
    int best_dx = 0, best_dy = 0;
    float best_scale = 1.0f;
    std::vector<float> q_scaled;

    for (float scale : {0.97f, 1.0f, 1.03f}) {
        scale_and_shift_rgb(q_rs, tpl_size, scale, 0, 0, q_scaled);
        std::vector<float> qs_g(npix);
        for (size_t i = 0; i < npix; ++i) {
            qs_g[i] = 0.299f * q_scaled[i * 3] + 0.587f * q_scaled[i * 3 + 1] + 0.114f * q_scaled[i * 3 + 2];
        }
        for (int dy = -3; dy <= 3; ++dy) {
            for (int dx = -3; dx <= 3; ++dx) {
                double q_mean = 0.0;
                for (int y = border; y < tpl_size - border; ++y) {
                    int sy = std::clamp(y - dy, 0, tpl_size - 1);
                    for (int x = border; x < tpl_size - border; ++x) {
                        size_t idx = static_cast<size_t>(y) * tpl_size + x;
                        if (stable[idx] == 0.0f) continue;
                        int sx = std::clamp(x - dx, 0, tpl_size - 1);
                        q_mean += qs_g[static_cast<size_t>(sy) * tpl_size + sx];
                    }
                }
                q_mean /= std::max(1.0f, s_sum);
                double dot = 0.0;
                double q_norm_sq = 0.0;
                for (int y = border; y < tpl_size - border; ++y) {
                    int sy = std::clamp(y - dy, 0, tpl_size - 1);
                    for (int x = border; x < tpl_size - border; ++x) {
                        size_t idx = static_cast<size_t>(y) * tpl_size + x;
                        if (stable[idx] == 0.0f) continue;
                        int sx = std::clamp(x - dx, 0, tpl_size - 1);
                        double qz = qs_g[static_cast<size_t>(sy) * tpl_size + sx] - q_mean;
                        dot += qz * m_z[idx];
                        q_norm_sq += qz * qz;
                    }
                }
                float sc = static_cast<float>(dot / ((std::sqrt(q_norm_sq) + 1e-6) * m_norm));
                if (sc > best_sc) {
                    best_sc = sc;
                    best_dx = dx;
                    best_dy = dy;
                    best_scale = scale;
                }
            }
        }
    }

    std::vector<float> q_aligned;
    scale_and_shift_rgb(q_rs, tpl_size, best_scale, best_dx, best_dy, q_aligned);

    // Weighted ZNCC across all expressions in the skin.
    double w_sum = 0.0;
    std::vector<float> sw(npix);
    for (size_t i = 0; i < npix; ++i) {
        float w = std::max(0.0f, var_weights[i]);
        w_sum += w;
        sw[i] = std::sqrt(w);
    }
    w_sum = std::max(1e-6, w_sum);

    double qm[3] = {0.0, 0.0, 0.0};
    for (size_t i = 0; i < npix; ++i) {
        float w = std::max(0.0f, var_weights[i]);
        qm[0] += q_aligned[i * 3 + 0] * w;
        qm[1] += q_aligned[i * 3 + 1] * w;
        qm[2] += q_aligned[i * 3 + 2] * w;
    }
    qm[0] /= w_sum; qm[1] /= w_sum; qm[2] /= w_sum;

    std::vector<float> qz(npix * 3);
    double qn_sq = 0.0;
    for (size_t i = 0; i < npix; ++i) {
        float s = sw[i];
        for (int c = 0; c < 3; ++c) {
            float v = static_cast<float>((q_aligned[i * 3 + c] - qm[c]) * s);
            qz[i * 3 + c] = v;
            qn_sq += static_cast<double>(v) * v;
        }
    }
    const double qn = std::sqrt(qn_sq) + 1e-6;

    int best_idx = 0;
    float best_ex = -2.0f;
    for (int t = 0; t < count; ++t) {
        const uint8_t* tpl = stack_rgb + static_cast<size_t>(t) * npix * 3;
        double tm[3] = {0.0, 0.0, 0.0};
        for (size_t i = 0; i < npix; ++i) {
            float w = std::max(0.0f, var_weights[i]);
            tm[0] += tpl[i * 3 + 0] * w;
            tm[1] += tpl[i * 3 + 1] * w;
            tm[2] += tpl[i * 3 + 2] * w;
        }
        tm[0] /= w_sum; tm[1] /= w_sum; tm[2] /= w_sum;

        double dot = 0.0;
        double tn_sq = 0.0;
        for (size_t i = 0; i < npix; ++i) {
            float s = sw[i];
            for (int c = 0; c < 3; ++c) {
                double tv = (static_cast<double>(tpl[i * 3 + c]) - tm[c]) * s;
                dot += tv * qz[i * 3 + c];
                tn_sq += tv * tv;
            }
        }
        float zncc = static_cast<float>(dot / ((std::sqrt(tn_sq) + 1e-6) * qn));
        if (zncc > best_ex) {
            best_ex = zncc;
            best_idx = t;
        }
    }

    result.best_index = best_idx;
    result.diff_zncc = best_ex;
    result.struct_zncc = best_sc;
    return result;
}

float score_emoji_structure_zncc(
    const uint8_t* rgb,
    int img_w,
    int img_h,
    const BoxI& box,
    const uint8_t* tpl_rgb,
    int tpl_size) {
    if (rgb == nullptr || tpl_rgb == nullptr || tpl_size <= 0) return -1.0f;
    std::vector<uint8_t> crop;
    int cw = 0, ch = 0;
    extract_crop_rgb(rgb, img_w, img_h, box, crop, cw, ch);
    center_square_inplace(crop, cw, ch, false);

    std::vector<float> q;
    resize_rgb_float(crop.data(), cw, ch, tpl_size, tpl_size, q);

    const size_t npix = static_cast<size_t>(tpl_size) * static_cast<size_t>(tpl_size);
    double qm[3] = {0.0, 0.0, 0.0};
    double tm[3] = {0.0, 0.0, 0.0};
    for (size_t i = 0; i < npix; ++i) {
        for (int c = 0; c < 3; ++c) {
            qm[c] += q[i * 3 + c];
            tm[c] += static_cast<double>(tpl_rgb[i * 3 + c]);
        }
    }
    for (int c = 0; c < 3; ++c) {
        qm[c] /= static_cast<double>(npix);
        tm[c] /= static_cast<double>(npix);
    }

    double dot = 0.0, qn2 = 0.0, tn2 = 0.0;
    for (size_t i = 0; i < npix; ++i) {
        for (int c = 0; c < 3; ++c) {
            double qv = q[i * 3 + c] - qm[c];
            double tv = static_cast<double>(tpl_rgb[i * 3 + c]) - tm[c];
            dot += qv * tv;
            qn2 += qv * qv;
            tn2 += tv * tv;
        }
    }
    double denom = std::sqrt(qn2) * std::sqrt(tn2);
    return denom > 1e-6 ? static_cast<float>(dot / denom) : 0.0f;
}

}  // namespace mmt
