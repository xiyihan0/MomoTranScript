#include "cv_lines.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace mmt {
namespace {

struct CompStat {
    int min_x = 0;
    int min_y = 0;
    int max_x = 0;
    int max_y = 0;
    int pixels = 0;
};

struct GlyphBox {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
};

inline uint8_t rgb_to_gray(uint8_t r, uint8_t g, uint8_t b) {
    return static_cast<uint8_t>((77 * r + 150 * g + 29 * b + 128) >> 8);
}

int find_root(std::vector<int>& parent, int x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

void unite(std::vector<int>& parent, int a, int b) {
    a = find_root(parent, a);
    b = find_root(parent, b);
    if (a != b) {
        if (a < b) {
            parent[b] = a;
        } else {
            parent[a] = b;
        }
    }
}

/// Fast two-pass 8-connected components with bounding-box stats.
int connected_components_8(
    const std::vector<uint8_t>& mask,
    int w,
    int h,
    std::vector<int>& labels,
    std::vector<CompStat>& stats) {
    const int n = w * h;
    labels.assign(static_cast<size_t>(n), 0);
    std::vector<int> parent;
    parent.push_back(0);  // label 0 = background
    int next_label = 1;

    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            if (!mask[row + x]) continue;
            int neighbors[4];
            int ncnt = 0;
            if (x > 0 && labels[row + x - 1] > 0) neighbors[ncnt++] = labels[row + x - 1];
            if (y > 0) {
                const int prow = (y - 1) * w;
                if (x > 0 && labels[prow + x - 1] > 0) neighbors[ncnt++] = labels[prow + x - 1];
                if (labels[prow + x] > 0) neighbors[ncnt++] = labels[prow + x];
                if (x + 1 < w && labels[prow + x + 1] > 0) neighbors[ncnt++] = labels[prow + x + 1];
            }
            if (ncnt == 0) {
                labels[row + x] = next_label;
                parent.push_back(next_label);
                ++next_label;
            } else {
                int best = find_root(parent, neighbors[0]);
                for (int k = 1; k < ncnt; ++k) {
                    int r = find_root(parent, neighbors[k]);
                    if (r < best) best = r;
                }
                labels[row + x] = best;
                for (int k = 0; k < ncnt; ++k) {
                    unite(parent, best, neighbors[k]);
                }
            }
        }
    }

    std::vector<int> remap(static_cast<size_t>(next_label), 0);
    int comp_count = 1;
    for (int i = 1; i < next_label; ++i) {
        int r = find_root(parent, i);
        if (remap[r] == 0) {
            remap[r] = comp_count++;
        }
        remap[i] = remap[r];
    }

    stats.assign(static_cast<size_t>(comp_count), CompStat{w, h, -1, -1, 0});
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            int lb = labels[row + x];
            if (lb == 0) continue;
            int c = remap[lb];
            labels[row + x] = c;
            auto& st = stats[c];
            if (x < st.min_x) st.min_x = x;
            if (x > st.max_x) st.max_x = x;
            if (y < st.min_y) st.min_y = y;
            if (y > st.max_y) st.max_y = y;
            st.pixels++;
        }
    }
    return comp_count;
}

float percentile_sorted(const std::vector<int>& sorted, float p) {
    if (sorted.empty()) return 8.0f;
    if (sorted.size() == 1) return static_cast<float>(sorted[0]);
    float pos = p * static_cast<float>(sorted.size() - 1);
    int lo = static_cast<int>(std::floor(pos));
    int hi = static_cast<int>(std::ceil(pos));
    float frac = pos - static_cast<float>(lo);
    return static_cast<float>(sorted[lo]) * (1.0f - frac) + static_cast<float>(sorted[hi]) * frac;
}

std::vector<std::pair<int, int>> find_runs(const std::vector<uint8_t>& flags) {
    std::vector<std::pair<int, int>> runs;
    int start = -1;
    for (int i = 0; i < static_cast<int>(flags.size()); ++i) {
        if (flags[i]) {
            if (start < 0) start = i;
        } else if (start >= 0) {
            runs.emplace_back(start, i);
            start = -1;
        }
    }
    if (start >= 0) {
        runs.emplace_back(start, static_cast<int>(flags.size()));
    }
    return runs;
}

}  // namespace

std::vector<TextLineProposal> segment_physical_lines(
    const uint8_t* rgb,
    int img_w,
    int img_h,
    const BoxI& roi,
    bool tight) {
    std::vector<TextLineProposal> out;
    if (rgb == nullptr || img_w <= 0 || img_h <= 0) return out;

    const int x0 = std::clamp(roi.x0, 0, img_w);
    const int y0 = std::clamp(roi.y0, 0, img_h);
    const int x1 = std::clamp(roi.x1, 0, img_w);
    const int y1 = std::clamp(roi.y1, 0, img_h);
    const int w = x1 - x0;
    const int h = y1 - y0;
    if (w < 2 || h < 2) return out;

    const size_t npix = static_cast<size_t>(w) * static_cast<size_t>(h);
    std::vector<uint8_t> gray(npix);
    std::vector<uint8_t> neutral(npix, 1);

    // Collect luminance histogram for fast O(N) median background estimation.
    std::array<int, 256> hist{};
    for (int y = 0; y < h; ++y) {
        const uint8_t* src_row = rgb + (static_cast<size_t>(y0 + y) * img_w + x0) * 3;
        const size_t dst_row = static_cast<size_t>(y) * w;
        for (int x = 0; x < w; ++x) {
            const uint8_t r = src_row[x * 3 + 0];
            const uint8_t g = src_row[x * 3 + 1];
            const uint8_t b = src_row[x * 3 + 2];
            const uint8_t gy = rgb_to_gray(r, g, b);
            gray[dst_row + x] = gy;
            hist[gy]++;
            if (!tight) {
                const int mx = std::max({r, g, b});
                const int mn = std::min({r, g, b});
                neutral[dst_row + x] = (mx - mn <= 48) ? 1 : 0;
            }
        }
    }

    int bg = 255;
    int half = static_cast<int>(npix / 2);
    int acc = 0;
    for (int v = 0; v < 256; ++v) {
        acc += hist[v];
        if (acc > half) {
            bg = v;
            break;
        }
    }

    struct PolarityChoice {
        int score = 0;
        bool has_ink = false;
        bool light = false;
        std::vector<uint8_t> ink;
        std::vector<int> heights;
        std::vector<GlyphBox> glyph_boxes;
    };

    PolarityChoice best;
    const int rim_y = std::max(4, std::min(7, static_cast<int>(std::lround(h * 0.08))));

    for (int p = 0; p < 2; ++p) {
        const bool light = (p == 1);
        std::vector<uint8_t> cand(npix, 0);
        for (size_t i = 0; i < npix; ++i) {
            const bool contrast = light ? (static_cast<int>(gray[i]) >= bg + 24)
                                        : (static_cast<int>(gray[i]) <= bg - 24);
            if (contrast && (tight || neutral[i])) {
                cand[i] = 1;
            }
        }

        std::vector<int> labels;
        std::vector<CompStat> stats;
        const int comp_count = connected_components_8(cand, w, h, labels, stats);

        std::vector<uint8_t> valid(static_cast<size_t>(comp_count), 0);
        std::vector<int> rim_candidates;
        bool any_valid = false;

        for (int i = 1; i < comp_count; ++i) {
            const auto& st = stats[i];
            const int cw = st.max_x - st.min_x + 1;
            const int ch = st.max_y - st.min_y + 1;
            if (st.pixels < 2 || (!tight && static_cast<double>(st.pixels) > static_cast<double>(h) * w * 0.45)) {
                continue;
            }
            if (!tight) {
                const bool is_rim =
                    (ch <= 4 && cw >= w * 0.72 && (st.min_y <= rim_y || st.max_y + 1 >= h - rim_y)) ||
                    (cw <= 4 && ch >= h * 0.72 && (st.min_x <= 4 || st.max_x + 1 >= w - 4)) ||
                    (cw >= w * 0.80 && ch >= h * 0.65 && st.min_y <= 4 && st.max_y + 1 >= h - 4 &&
                     st.pixels < cw * ch * 0.35);
                if (is_rim) {
                    rim_candidates.push_back(i);
                    continue;
                }
            }
            valid[i] = 1;
            any_valid = true;
        }

        if (!any_valid && !rim_candidates.empty()) {
            for (int idx : rim_candidates) {
                valid[idx] = 1;
                any_valid = true;
            }
        }

        PolarityChoice cur;
        cur.light = light;
        cur.ink.assign(npix, 0);
        for (size_t i = 0; i < npix; ++i) {
            if (valid[labels[i]]) {
                cur.ink[i] = 1;
                cur.has_ink = true;
            }
        }

        for (int i = 1; i < comp_count; ++i) {
            if (!valid[i]) continue;
            const auto& st = stats[i];
            const int cw = st.max_x - st.min_x + 1;
            const int ch = st.max_y - st.min_y + 1;
            const float aspect = static_cast<float>(cw) / static_cast<float>(std::max(1, ch));
            if (ch >= 4 && aspect >= 0.08f && aspect <= 5.0f) {
                cur.score += st.pixels;
                cur.heights.push_back(ch);
                cur.glyph_boxes.push_back(GlyphBox{st.min_x, st.min_y, cw, ch});
            }
        }

        if (p == 0 || cur.score > best.score || (cur.score == best.score && cur.has_ink && !best.has_ink)) {
            best = std::move(cur);
        }
    }

    if (!best.has_ink) {
        return out;
    }

    std::sort(best.heights.begin(), best.heights.end());
    const float typical = best.heights.empty() ? 8.0f : percentile_sorted(best.heights, 0.50f);
    const float glyph_height = best.heights.empty() ? typical : percentile_sorted(best.heights, 0.75f);

    std::vector<int> row_ink(static_cast<size_t>(h), 0);
    int max_row_ink = 0;
    for (int y = 0; y < h; ++y) {
        int s = 0;
        const size_t row = static_cast<size_t>(y) * w;
        for (int x = 0; x < w; ++x) {
            s += best.ink[row + x];
        }
        row_ink[y] = s;
        if (s > max_row_ink) max_row_ink = s;
    }

    const int density_floor = std::max(
        2,
        static_cast<int>(std::lround(std::min(static_cast<double>(w) * 0.015, static_cast<double>(max_row_ink) * 0.10))));

    std::vector<uint8_t> occupied(static_cast<size_t>(h), 0);
    for (int y = 0; y < h; ++y) {
        if (row_ink[y] >= density_floor) {
            occupied[y] = 1;
        }
    }
    for (const auto& gb : best.glyph_boxes) {
        if (static_cast<float>(gb.h) <= glyph_height * 1.5f) {
            const int gy0 = std::max(0, gb.y);
            const int gy1 = std::min(h, gb.y + gb.h);
            for (int y = gy0; y < gy1; ++y) {
                occupied[y] = 1;
            }
        }
    }

    // Close 1-pixel vertical gaps.
    auto raw_runs = find_runs(occupied);
    for (size_t i = 0; i + 1 < raw_runs.size(); ++i) {
        if (raw_runs[i + 1].first - raw_runs[i].second <= 1) {
            for (int y = raw_runs[i].second; y < raw_runs[i + 1].first; ++y) {
                occupied[y] = 1;
            }
        }
    }

    std::vector<std::pair<int, int>> joined;
    for (const auto& [top, bottom] : find_runs(occupied)) {
        if (!joined.empty()) {
            const int prev_top = joined.back().first;
            const int prev_bot = joined.back().second;
            const int gap = top - prev_bot;
            const bool close = static_cast<float>(gap) < typical * 0.4f;
            const int prev_h = prev_bot - prev_top;
            const int cur_h = bottom - top;
            const bool short_part = close && static_cast<float>(std::min(cur_h, prev_h)) < typical * 0.35f;

            bool glyph_parts = false;
            if (tight && close) {
                int min_px = w, max_px = -1;
                int min_cx = w, max_cx = -1;
                for (int y = prev_top; y < prev_bot; ++y) {
                    const size_t row = static_cast<size_t>(y) * w;
                    for (int x = 0; x < w; ++x) {
                        if (best.ink[row + x]) {
                            if (x < min_px) min_px = x;
                            if (x > max_px) max_px = x;
                        }
                    }
                }
                for (int y = top; y < bottom; ++y) {
                    const size_t row = static_cast<size_t>(y) * w;
                    for (int x = 0; x < w; ++x) {
                        if (best.ink[row + x]) {
                            if (x < min_cx) min_cx = x;
                            if (x > max_cx) max_cx = x;
                        }
                    }
                }
                if (max_px >= min_px && max_cx >= min_cx) {
                    int span = std::max(max_px, max_cx) - std::min(min_px, min_cx) + 1;
                    int overlap = std::min(max_px, max_cx) - std::max(min_px, min_cx) + 1;
                    int narrower = std::min(max_px - min_px + 1, max_cx - min_cx + 1);
                    float gw = typical * 2.0f;
                    glyph_parts = (static_cast<float>(bottom - prev_top) <= gw * 1.25f) &&
                                  (static_cast<float>(span) <= gw * 1.25f) &&
                                  (static_cast<float>(overlap) >= static_cast<float>(narrower) * 0.5f);
                }
            }

            if (short_part || glyph_parts) {
                joined.back().second = bottom;
                continue;
            }
        }
        joined.emplace_back(top, bottom);
    }

    for (const auto& [top, bottom] : joined) {
        int min_x = w, max_x = -1;
        int min_y = bottom, max_y = -1;
        for (int y = top; y < bottom; ++y) {
            const size_t row = static_cast<size_t>(y) * w;
            for (int x = 0; x < w; ++x) {
                if (best.ink[row + x]) {
                    if (x < min_x) min_x = x;
                    if (x > max_x) max_x = x;
                    if (y < min_y) min_y = y;
                    if (y > max_y) max_y = y;
                }
            }
        }
        if (max_x < min_x || max_y < min_y) continue;
        BoxI ink_box{x0 + min_x, y0 + min_y, x0 + max_x + 1, y0 + max_y + 1};
        BoxI padded{
            std::max(x0, ink_box.x0 - 2),
            std::max(y0, ink_box.y0 - 2),
            std::min(x1, ink_box.x1 + 2),
            std::min(y1, ink_box.y1 + 2),
        };
        out.push_back(TextLineProposal{ink_box, padded, best.light});
    }
    return out;
}

}  // namespace mmt
