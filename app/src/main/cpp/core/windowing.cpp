#include "windowing.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <unordered_set>

namespace mmt {
namespace {

int py_round(double v) {
    return static_cast<int>(std::nearbyint(v));
}

uint8_t median_u8_channel(const std::vector<uint8_t>& vals) {
    if (vals.empty()) return 0;
    std::vector<uint8_t> copy = vals;
    const size_t mid = copy.size() / 2;
    std::nth_element(copy.begin(), copy.begin() + static_cast<std::ptrdiff_t>(mid), copy.end());
    if (copy.size() % 2 == 1) {
        return copy[mid];
    }
    const uint8_t upper = copy[mid];
    const uint8_t lower = *std::max_element(copy.begin(), copy.begin() + static_cast<std::ptrdiff_t>(mid));
    return static_cast<uint8_t>((static_cast<int>(lower) + static_cast<int>(upper)) / 2);
}

}  // namespace

std::vector<Window> large_only_windows(int width, int height) {
    const int size = std::min(height, std::max(1, py_round(2.2 * width)));
    const int stride = std::max(1, py_round(1.1 * width));
    std::vector<int> starts;
    for (int top = 0; top <= height - size; top += stride) {
        starts.push_back(top);
    }
    if (starts.empty() || starts.back() != height - size) {
        starts.push_back(std::max(0, height - size));
    }
    std::vector<Window> windows;
    windows.reserve(starts.size());
    for (int top : starts) {
        windows.push_back(Window{0, top, width, std::min(height, top + size)});
    }
    return windows;
}

std::vector<Window> adaptive_blank_windows(
    const uint8_t* rgb,
    int width,
    int height,
    float min_ratio,
    float max_ratio) {
    const int win_h = std::min(height, std::max(1, py_round(static_cast<double>(max_ratio) * width)));
    const int stride = std::max(1, py_round(static_cast<double>(min_ratio) * width));
    if (rgb == nullptr || height <= win_h || width < 16) {
        return large_only_windows(width, height);
    }

    // Estimate page canvas RGB from outer 2% left/right strips (subsampled vertically if huge).
    const int edge_w = std::max(2, width / 50);
    const int row_step = std::max(1, height / 2048);
    std::vector<uint8_t> r_vals, g_vals, b_vals;
    const size_t approx = static_cast<size_t>((height / row_step + 1) * edge_w * 2);
    r_vals.reserve(approx);
    g_vals.reserve(approx);
    b_vals.reserve(approx);

    for (int y = 0; y < height; y += row_step) {
        const uint8_t* row = rgb + static_cast<size_t>(y) * width * 3;
        for (int x = 0; x < edge_w; ++x) {
            r_vals.push_back(row[x * 3 + 0]);
            g_vals.push_back(row[x * 3 + 1]);
            b_vals.push_back(row[x * 3 + 2]);
        }
        for (int x = width - edge_w; x < width; ++x) {
            r_vals.push_back(row[x * 3 + 0]);
            g_vals.push_back(row[x * 3 + 1]);
            b_vals.push_back(row[x * 3 + 2]);
        }
    }
    const int canvas_r = median_u8_channel(r_vals);
    const int canvas_g = median_u8_channel(g_vals);
    const int canvas_b = median_u8_channel(b_vals);

    const int margin_x = std::max(2, py_round(0.03 * width));
    const int x_end = width - margin_x;
    std::vector<uint8_t> is_blank(static_cast<size_t>(height), 0);

    for (int y = 0; y < height; ++y) {
        const uint8_t* row = rgb + static_cast<size_t>(y) * width * 3;
        int r_min = 255, g_min = 255, b_min = 255;
        int r_max = 0, g_max = 0, b_max = 0;
        for (int x = margin_x; x < x_end; ++x) {
            const int r = row[x * 3 + 0];
            const int g = row[x * 3 + 1];
            const int b = row[x * 3 + 2];
            if (r < r_min) r_min = r;
            if (r > r_max) r_max = r;
            if (g < g_min) g_min = g;
            if (g > g_max) g_max = g;
            if (b < b_min) b_min = b;
            if (b > b_max) b_max = b;
        }
        const int row_range = std::max({r_max - r_min, g_max - g_min, b_max - b_min});
        if (row_range > 12) continue;
        const int med_r = (r_min + r_max) / 2;
        const int med_g = (g_min + g_max) / 2;
        const int med_b = (b_min + b_max) / 2;
        const int diff = std::max({
            std::abs(med_r - canvas_r),
            std::abs(med_g - canvas_g),
            std::abs(med_b - canvas_b),
        });
        if (diff <= 16) {
            is_blank[static_cast<size_t>(y)] = 1;
        }
    }

    const int min_band = std::max(4, py_round(0.008 * width));
    std::vector<int> cuts;
    std::unordered_set<int> cut_set;
    int y = 0;
    while (y < height) {
        if (!is_blank[static_cast<size_t>(y)]) {
            ++y;
            continue;
        }
        const int start = y;
        while (y < height && is_blank[static_cast<size_t>(y)]) {
            ++y;
        }
        if (y - start >= min_band) {
            const int c = (start + y) / 2;
            cuts.push_back(c);
            cut_set.insert(c);
        }
    }

    std::vector<Window> windows;
    int top = 0;
    while (top < height) {
        if (height - top <= win_h) {
            if (top > 0 && ((height - top) < stride || cut_set.find(top) == cut_set.end())) {
                const int tail_top = std::max(0, height - win_h);
                int chosen = tail_top;
                for (int c : cuts) {
                    if (c >= tail_top && c <= top) {
                        chosen = c;
                        break;
                    }
                }
                windows.push_back(Window{0, chosen, width, height});
            } else {
                windows.push_back(Window{0, top, width, height});
            }
            break;
        }
        const int lo = top + stride;
        const int hi = top + win_h;
        int best_cut = -1;
        for (int c : cuts) {
            if (c >= lo && c <= hi) {
                best_cut = c;
            }
        }
        if (best_cut >= 0) {
            windows.push_back(Window{0, top, width, best_cut});
            top = best_cut;
        } else {
            windows.push_back(Window{0, top, width, top + win_h});
            top = top + stride;
        }
    }
    return windows;
}

}  // namespace mmt
