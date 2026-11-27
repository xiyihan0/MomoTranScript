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
    const int max_border_y = std::max(4, static_cast<int>(std::lround(h * 0.06f)));
    const int max_border_x = std::max(4, static_cast<int>(std::lround(w * 0.06f)));
    auto is_row_pad = [&](int y, int x0, int x1, int depth) -> bool {
        int white_cnt = 0;
        int neutral_cnt = 0;
        double sum = 0.0, sum_sq = 0.0;
        int len = std::max(1, x1 - x0);
        const uint8_t* row = crop.data() + static_cast<size_t>(y) * w * 3;
        for (int x = x0; x < x1; ++x) {
            uint8_t r = row[x * 3], g = row[x * 3 + 1], b = row[x * 3 + 2];
            uint8_t mn = std::min({r, g, b});
            uint8_t mx = std::max({r, g, b});
            if (mn >= 242) white_cnt++;
            if (mn >= 208 && (mx - mn) <= 14) neutral_cnt++;
            sum += mn;
            sum_sq += static_cast<double>(mn) * mn;
        }
        if (static_cast<float>(white_cnt) / static_cast<float>(len) >= 0.96f) return true;
        if (depth < max_border_y && static_cast<float>(neutral_cnt) / static_cast<float>(len) >= 0.94f) {
            double mean = sum / len;
            double var = std::max(0.0, sum_sq / len - mean * mean);
            return std::sqrt(var) < 18.0;
        }
        return false;
    };
    auto is_col_pad = [&](int x, int y0, int y1, int depth) -> bool {
        int white_cnt = 0;
        int neutral_cnt = 0;
        double sum = 0.0, sum_sq = 0.0;
        int len = std::max(1, y1 - y0);
        for (int y = y0; y < y1; ++y) {
            const uint8_t* p = crop.data() + (static_cast<size_t>(y) * w + x) * 3;
            uint8_t r = p[0], g = p[1], b = p[2];
            uint8_t mn = std::min({r, g, b});
            uint8_t mx = std::max({r, g, b});
            if (mn >= 242) white_cnt++;
            if (mn >= 208 && (mx - mn) <= 14) neutral_cnt++;
            sum += mn;
            sum_sq += static_cast<double>(mn) * mn;
        }
        if (static_cast<float>(white_cnt) / static_cast<float>(len) >= 0.96f) return true;
        if (depth < max_border_x && static_cast<float>(neutral_cnt) / static_cast<float>(len) >= 0.94f) {
            double mean = sum / len;
            double var = std::max(0.0, sum_sq / len - mean * mean);
            return std::sqrt(var) < 18.0;
        }
        return false;
    };

    int y0 = 0, y1 = h, x0 = 0, x1 = w;
    while (y0 < y1 - 16 && is_row_pad(y0, x0, x1, y0)) ++y0;
    while (y1 > y0 + 16 && is_row_pad(y1 - 1, x0, x1, h - y1)) --y1;
    while (x0 < x1 - 16 && is_col_pad(x0, y0, y1, x0)) ++x0;
    while (x1 > x0 + 16 && is_col_pad(x1 - 1, y0, y1, w - x1)) --x1;

    if (x0 == 0 && y0 == 0 && x1 == w && y1 == h) return;

    // Match Python trim_asymmetric_padding: preserve natural symmetric white margins of normal
    // white-bg CharFaces; only trim all 4 sides if a light card frame wraps a dark-bg portrait,
    // or trim only the excess side on large asymmetric blank padding.
    const int top_pad = y0;
    const int bot_pad = h - y1;
    const int left_pad = x0;
    const int right_pad = w - x1;
    const bool asym_x = std::abs(left_pad - right_pad) >= std::max(16, static_cast<int>(0.14f * w));
    const bool asym_y = std::abs(top_pad - bot_pad) >= std::max(16, static_cast<int>(0.14f * h));

    int dark_ring = 0;
    int ring_total = 0;
    for (int x = x0; x < x1; ++x) {
        const uint8_t* pt = crop.data() + (static_cast<size_t>(y0) * w + x) * 3;
        const uint8_t* pb = crop.data() + (static_cast<size_t>(y1 - 1) * w + x) * 3;
        if (rgb_to_gray(pt[0], pt[1], pt[2]) < 48) dark_ring++;
        if (rgb_to_gray(pb[0], pb[1], pb[2]) < 48) dark_ring++;
        ring_total += 2;
    }
    for (int y = y0; y < y1; ++y) {
        const uint8_t* pl = crop.data() + (static_cast<size_t>(y) * w + x0) * 3;
        const uint8_t* pr = crop.data() + (static_cast<size_t>(y) * w + (x1 - 1)) * 3;
        if (rgb_to_gray(pl[0], pl[1], pl[2]) < 48) dark_ring++;
        if (rgb_to_gray(pr[0], pr[1], pr[2]) < 48) dark_ring++;
        ring_total += 2;
    }
    const bool dark_bg_framed = (ring_total > 0) &&
        (static_cast<float>(dark_ring) / static_cast<float>(ring_total) >= 0.35f);

    int ny0 = 0, ny1 = h, nx0 = 0, nx1 = w;
    if (dark_bg_framed) {
        ny0 = y0; ny1 = y1; nx0 = x0; nx1 = x1;
    } else if (asym_x || asym_y) {
        const int sym_x = asym_x ? std::min(left_pad, right_pad) : 0;
        const int sym_y = asym_y ? std::min(top_pad, bot_pad) : 0;
        ny0 = asym_y ? std::max(0, y0 - sym_y) : 0;
        ny1 = asym_y ? std::min(h, y1 + sym_y) : h;
        nx0 = asym_x ? std::max(0, x0 - sym_x) : 0;
        nx1 = asym_x ? std::min(w, x1 + sym_x) : w;
    } else {
        return;
    }

    if (nx0 == 0 && ny0 == 0 && nx1 == w && ny1 == h) return;
    int nw = nx1 - nx0;
    int nh = ny1 - ny0;
    std::vector<uint8_t> trimmed(static_cast<size_t>(nw) * nh * 3);
    for (int y = 0; y < nh; ++y) {
        const uint8_t* src = crop.data() + (static_cast<size_t>(ny0 + y) * w + nx0) * 3;
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

float compute_rgb_spatial_std(const std::vector<uint8_t>& rgb, int w, int h, bool inner_circle_only) {
    if (rgb.empty() || w <= 0 || h <= 0) return 0.0f;
    double sum[3] = {0.0, 0.0, 0.0};
    double sum_sq[3] = {0.0, 0.0, 0.0};
    size_t count = 0;
    const float cx = static_cast<float>(w - 1) * 0.5f;
    const float cy = static_cast<float>(h - 1) * 0.5f;
    const float r = static_cast<float>(std::min(w, h)) * 0.40f;
    const float r2 = r * r;
    for (int y = 0; y < h; ++y) {
        const float dy = static_cast<float>(y) - cy;
        const uint8_t* row = rgb.data() + static_cast<size_t>(y) * w * 3;
        for (int x = 0; x < w; ++x) {
            if (inner_circle_only) {
                const float dx = static_cast<float>(x) - cx;
                if (dx * dx + dy * dy > r2) continue;
            }
            for (int c = 0; c < 3; ++c) {
                double d = static_cast<double>(row[x * 3 + c]);
                sum[c] += d;
                sum_sq[c] += d * d;
            }
            ++count;
        }
    }
    if (count == 0) return 0.0f;
    const double n = static_cast<double>(count);
    double max_std = 0.0;
    for (int c = 0; c < 3; ++c) {
        double mean = sum[c] / n;
        double var = std::max(0.0, sum_sq[c] / n - mean * mean);
        max_std = std::max(max_std, std::sqrt(var));
    }
    return static_cast<float>(max_std);
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
    float* out_aspect,
    bool norm_half) {
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
        *out_std = compute_rgb_spatial_std(crop, cw, ch, circle_mask);
    }

    const size_t plane = static_cast<size_t>(dst_size) * static_cast<size_t>(dst_size);
    const float sx_scale = static_cast<float>(cw) / static_cast<float>(dst_size);
    const float sy_scale = static_cast<float>(ch) / static_cast<float>(dst_size);
    const float mean_c[3] = {
        norm_half ? 0.5f : kMean[0],
        norm_half ? 0.5f : kMean[1],
        norm_half ? 0.5f : kMean[2],
    };
    const float inv_std_c[3] = {
        norm_half ? 2.0f : (1.0f / kStd[0]),
        norm_half ? 2.0f : (1.0f / kStd[1]),
        norm_half ? 2.0f : (1.0f / kStd[2]),
    };

    for (int dy = 0; dy < dst_size; ++dy) {
        float sy = (static_cast<float>(dy) + 0.5f) * sy_scale - 0.5f;
        size_t row = static_cast<size_t>(dy) * dst_size;
        for (int dx = 0; dx < dst_size; ++dx) {
            float sx = (static_cast<float>(dx) + 0.5f) * sx_scale - 0.5f;
            for (int c = 0; c < 3; ++c) {
                float v = sample_bilinear_u8(crop.data(), cw, ch, sx, sy, c) * (1.0f / 255.0f);
                out_nchw[c * plane + row + dx] = (v - mean_c[c]) * inv_std_c[c];
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
    trim_light_border_inplace(crop, cw, ch);
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

    for (float scale : {0.94f, 0.97f, 1.0f, 1.03f, 1.06f}) {
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
    trim_light_border_inplace(crop, cw, ch);
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
