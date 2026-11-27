#include "tensor_prep.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace mmt {
namespace {

inline float bilinear_sample(
    const uint8_t* src,
    int src_w,
    int src_h,
    int stride_bytes,
    float sx,
    float sy,
    int channel) {
    sx = std::clamp(sx, 0.0f, static_cast<float>(src_w - 1));
    sy = std::clamp(sy, 0.0f, static_cast<float>(src_h - 1));
    const int x0 = static_cast<int>(sx);
    const int y0 = static_cast<int>(sy);
    const int x1 = std::min(x0 + 1, src_w - 1);
    const int y1 = std::min(y0 + 1, src_h - 1);
    const float dx = sx - static_cast<float>(x0);
    const float dy = sy - static_cast<float>(y0);

    const uint8_t* row0 = src + static_cast<size_t>(y0) * stride_bytes;
    const uint8_t* row1 = src + static_cast<size_t>(y1) * stride_bytes;

    const float v00 = static_cast<float>(row0[x0 * 3 + channel]);
    const float v10 = static_cast<float>(row0[x1 * 3 + channel]);
    const float v01 = static_cast<float>(row1[x0 * 3 + channel]);
    const float v11 = static_cast<float>(row1[x1 * 3 + channel]);

    const float top = v00 + (v10 - v00) * dx;
    const float bot = v01 + (v11 - v01) * dx;
    return top + (bot - top) * dy;
}

uint8_t channel_median(const std::vector<uint8_t>& rgb, int channel) {
    std::array<int, 256> hist{};
    const size_t n = rgb.size() / 3;
    for (size_t i = 0; i < n; ++i) {
        hist[rgb[i * 3 + channel]]++;
    }
    const int half = static_cast<int>(n / 2);
    int acc = 0;
    for (int v = 0; v < 256; ++v) {
        acc += hist[v];
        if (acc > half) return static_cast<uint8_t>(v);
    }
    return 255;
}

}  // namespace

void prepare_det_input_nchw(
    const uint8_t* rgb,
    int img_w,
    int img_h,
    const Window& win,
    int dst_size,
    float* out_nchw) {
    const int x0 = std::clamp(win.x0, 0, img_w);
    const int y0 = std::clamp(win.y0, 0, img_h);
    const int x1 = std::clamp(win.x1, 0, img_w);
    const int y1 = std::clamp(win.y1, 0, img_h);
    const int cw = std::max(1, x1 - x0);
    const int ch = std::max(1, y1 - y0);

    const uint8_t* crop_ptr = rgb + (static_cast<size_t>(y0) * img_w + x0) * 3;
    const int stride_bytes = img_w * 3;
    const size_t plane = static_cast<size_t>(dst_size) * static_cast<size_t>(dst_size);

    const bool pad_bottom = ch < cw;
    const int eff_h = pad_bottom ? cw : ch;
    std::array<float, 3> bg_norm{1.0f, 1.0f, 1.0f};
    if (pad_bottom) {
        std::array<std::array<int, 256>, 3> hist{};
        int total_px = 0;
        const int b_rows = std::min(2, ch);
        const int b_cols = std::min(2, cw);
        auto add_px = [&](const uint8_t* px) {
            hist[0][px[0]]++;
            hist[1][px[1]]++;
            hist[2][px[2]]++;
            total_px++;
        };
        for (int y = ch - b_rows; y < ch; ++y) {
            const uint8_t* row = crop_ptr + static_cast<size_t>(y) * stride_bytes;
            for (int x = 0; x < cw; ++x) add_px(row + x * 3);
        }
        for (int y = 0; y < ch; ++y) {
            const uint8_t* row = crop_ptr + static_cast<size_t>(y) * stride_bytes;
            for (int x = 0; x < b_cols; ++x) add_px(row + x * 3);
            for (int x = cw - b_cols; x < cw; ++x) add_px(row + x * 3);
        }
        const int half = total_px / 2;
        for (int c = 0; c < 3; ++c) {
            int acc = 0;
            uint8_t med = 255;
            for (int v = 0; v < 256; ++v) {
                acc += hist[c][v];
                if (acc > half) {
                    med = static_cast<uint8_t>(v);
                    break;
                }
            }
            bg_norm[c] = static_cast<float>(med) * (1.0f / 255.0f);
        }
    }

    const float scale_x = static_cast<float>(cw) / static_cast<float>(dst_size);
    const float scale_y = static_cast<float>(eff_h) / static_cast<float>(dst_size);
    const float last_row_y = static_cast<float>(ch - 1);
    const float ch_f = static_cast<float>(ch);

    for (int dy = 0; dy < dst_size; ++dy) {
        const float sy = (static_cast<float>(dy) + 0.5f) * scale_y - 0.5f;
        const size_t row_off = static_cast<size_t>(dy) * dst_size;
        if (pad_bottom && sy >= ch_f) {
            for (int dx = 0; dx < dst_size; ++dx) {
                out_nchw[0 * plane + row_off + dx] = bg_norm[0];
                out_nchw[1 * plane + row_off + dx] = bg_norm[1];
                out_nchw[2 * plane + row_off + dx] = bg_norm[2];
            }
            continue;
        }
        const bool blend_pad = pad_bottom && sy > last_row_y;
        const float pad_w = blend_pad ? (sy - last_row_y) : 0.0f;
        for (int dx = 0; dx < dst_size; ++dx) {
            const float sx = (static_cast<float>(dx) + 0.5f) * scale_x - 0.5f;
            for (int c = 0; c < 3; ++c) {
                float val = bilinear_sample(crop_ptr, cw, ch, stride_bytes, sx, sy, c) * (1.0f / 255.0f);
                if (blend_pad) {
                    val = val * (1.0f - pad_w) + bg_norm[c] * pad_w;
                }
                out_nchw[static_cast<size_t>(c) * plane + row_off + dx] = val;
            }
        }
    }
}

void prepare_rec_line_chw(
    const uint8_t* rgb,
    int img_w,
    int img_h,
    const TextLineProposal& line,
    int target_pad_h,
    int dst_h,
    int max_w,
    std::vector<float>& out_chw,
    int& out_resized_w) {
    const int x0 = std::clamp(line.box.x0, 0, img_w);
    const int y0 = std::clamp(line.box.y0, 0, img_h);
    const int x1 = std::clamp(line.box.x1, 0, img_w);
    const int y1 = std::clamp(line.box.y1, 0, img_h);
    const int cw = std::max(1, x1 - x0);
    const int ch = std::max(1, y1 - y0);

    std::vector<uint8_t> crop(static_cast<size_t>(cw) * ch * 3);
    if (line.is_light_fg) {
        std::vector<uint8_t> inv(static_cast<size_t>(cw) * ch);
        uint8_t mn = 255, mx = 0;
        for (int y = 0; y < ch; ++y) {
            const uint8_t* src = rgb + (static_cast<size_t>(y0 + y) * img_w + x0) * 3;
            for (int x = 0; x < cw; ++x) {
                uint8_t gy = static_cast<uint8_t>((77 * src[x * 3] + 150 * src[x * 3 + 1] + 29 * src[x * 3 + 2] + 128) >> 8);
                uint8_t iv = static_cast<uint8_t>(255 - gy);
                inv[static_cast<size_t>(y) * cw + x] = iv;
                if (iv < mn) mn = iv;
                if (iv > mx) mx = iv;
            }
        }
        const int span = std::max(1, static_cast<int>(mx) - static_cast<int>(mn));
        for (size_t i = 0; i < inv.size(); ++i) {
            uint8_t norm = static_cast<uint8_t>(((static_cast<int>(inv[i]) - mn) * 255) / span);
            crop[i * 3 + 0] = norm;
            crop[i * 3 + 1] = norm;
            crop[i * 3 + 2] = norm;
        }
    } else {
        for (int y = 0; y < ch; ++y) {
            const uint8_t* src = rgb + (static_cast<size_t>(y0 + y) * img_w + x0) * 3;
            uint8_t* dst = crop.data() + static_cast<size_t>(y) * cw * 3;
            std::copy_n(src, cw * 3, dst);
        }
    }

    const uint8_t bg_r = channel_median(crop, 0);
    const uint8_t bg_g = channel_median(crop, 1);
    const uint8_t bg_b = channel_median(crop, 2);

    int pad_top = 3;
    int pad_bot = 3;
    if (target_pad_h > ch) {
        int extra = target_pad_h - ch;
        pad_top += extra / 2;
        pad_bot += extra - extra / 2;
    }
    const int pad_x = 4;
    const int pw = cw + pad_x * 2;
    const int ph = ch + pad_top + pad_bot;

    std::vector<uint8_t> padded(static_cast<size_t>(pw) * ph * 3);
    for (size_t i = 0; i < static_cast<size_t>(pw) * ph; ++i) {
        padded[i * 3 + 0] = bg_r;
        padded[i * 3 + 1] = bg_g;
        padded[i * 3 + 2] = bg_b;
    }
    for (int y = 0; y < ch; ++y) {
        const uint8_t* src = crop.data() + static_cast<size_t>(y) * cw * 3;
        uint8_t* dst = padded.data() + (static_cast<size_t>(y + pad_top) * pw + pad_x) * 3;
        std::copy_n(src, cw * 3, dst);
    }

    const float ratio = static_cast<float>(pw) / static_cast<float>(ph);
    out_resized_w = std::clamp(static_cast<int>(std::ceil(static_cast<float>(dst_h) * ratio)), 1, max_w);
    const size_t plane = static_cast<size_t>(dst_h) * static_cast<size_t>(out_resized_w);
    out_chw.assign(plane * 3, 0.0f);

    const float scale_x = static_cast<float>(pw) / static_cast<float>(out_resized_w);
    const float scale_y = static_cast<float>(ph) / static_cast<float>(dst_h);
    const int stride_bytes = pw * 3;

    // PP-OCRv6 expects BGR input normalized to [-1, 1]: (v / 255.0 - 0.5) / 0.5.
    for (int dy = 0; dy < dst_h; ++dy) {
        const float sy = (static_cast<float>(dy) + 0.5f) * scale_y - 0.5f;
        const size_t row_off = static_cast<size_t>(dy) * out_resized_w;
        for (int dx = 0; dx < out_resized_w; ++dx) {
            const float sx = (static_cast<float>(dx) + 0.5f) * scale_x - 0.5f;
            const float r = bilinear_sample(padded.data(), pw, ph, stride_bytes, sx, sy, 0);
            const float g = bilinear_sample(padded.data(), pw, ph, stride_bytes, sx, sy, 1);
            const float b = bilinear_sample(padded.data(), pw, ph, stride_bytes, sx, sy, 2);
            out_chw[0 * plane + row_off + dx] = (b * (1.0f / 255.0f) - 0.5f) * 2.0f;
            out_chw[1 * plane + row_off + dx] = (g * (1.0f / 255.0f) - 0.5f) * 2.0f;
            out_chw[2 * plane + row_off + dx] = (r * (1.0f / 255.0f) - 0.5f) * 2.0f;
        }
    }
}

}  // namespace mmt
