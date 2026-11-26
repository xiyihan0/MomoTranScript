#include "cv_lines.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>

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

struct PointI {
    int x = 0;
    int y = 0;
};

inline uint8_t rgb_to_gray(uint8_t r, uint8_t g, uint8_t b) {
    return static_cast<uint8_t>((77 * r + 150 * g + 29 * b + 128) >> 8);
}

int hist_median(const std::array<int, 256>& hist, int total) {
    if (total <= 0) return 255;
    const int half = total / 2;
    int acc = 0;
    for (int v = 0; v < 256; ++v) {
        acc += hist[v];
        if (acc > half) return v;
    }
    return 255;
}

inline int64_t cross2d(const PointI& o, const PointI& a, const PointI& b) {
    return static_cast<int64_t>(a.x - o.x) * (b.y - o.y) -
           static_cast<int64_t>(a.y - o.y) * (b.x - o.x);
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

    // Collect luminance + RGB histograms for fast O(N) median background estimation.
    std::array<int, 256> hist_g{};
    std::array<int, 256> hist_r{};
    std::array<int, 256> hist_gr{};
    std::array<int, 256> hist_b{};
    uint8_t gray_min = 255;
    uint8_t gray_max = 0;
    for (int y = 0; y < h; ++y) {
        const uint8_t* src_row = rgb + (static_cast<size_t>(y0 + y) * img_w + x0) * 3;
        const size_t dst_row = static_cast<size_t>(y) * w;
        for (int x = 0; x < w; ++x) {
            const uint8_t r = src_row[x * 3 + 0];
            const uint8_t g = src_row[x * 3 + 1];
            const uint8_t b = src_row[x * 3 + 2];
            const uint8_t gy = rgb_to_gray(r, g, b);
            gray[dst_row + x] = gy;
            hist_g[gy]++;
            hist_r[r]++;
            hist_gr[g]++;
            hist_b[b]++;
            if (gy < gray_min) gray_min = gy;
            if (gy > gray_max) gray_max = gy;
            if (!tight) {
                const int mx = std::max({r, g, b});
                const int mn = std::min({r, g, b});
                neutral[dst_row + x] = (mx - mn <= 80) ? 1 : 0;
            }
        }
    }

    int bg = hist_median(hist_g, static_cast<int>(npix));
    const int inside_r = hist_median(hist_r, static_cast<int>(npix));
    const int inside_g = hist_median(hist_gr, static_cast<int>(npix));
    const int inside_b = hist_median(hist_b, static_cast<int>(npix));

    // Tight text (e.g. speaker_name / message_note) can contain more ink than background.
    // Use adjacent exterior pixels when that background continues across the ROI rim.
    if (tight) {
        struct RimPair {
            uint8_t or_, og, ob;
            uint8_t ir, ig, ib;
        };
        std::vector<RimPair> pairs;
        pairs.reserve(static_cast<size_t>(2 * (w + h)));
        auto push_px = [&](int ox_p, int oy_p, int ix_p, int iy_p) {
            const uint8_t* op = rgb + (static_cast<size_t>(oy_p) * img_w + ox_p) * 3;
            const uint8_t* ip = rgb + (static_cast<size_t>(iy_p) * img_w + ix_p) * 3;
            pairs.push_back(RimPair{op[0], op[1], op[2], ip[0], ip[1], ip[2]});
        };
        if (y0 > 0) {
            for (int x = x0; x < x1; ++x) push_px(x, y0 - 1, x, y0);
        }
        if (y1 < img_h) {
            for (int x = x0; x < x1; ++x) push_px(x, y1, x, y1 - 1);
        }
        if (x0 > 0) {
            for (int y = y0; y < y1; ++y) push_px(x0 - 1, y, x0, y);
        }
        if (x1 < img_w) {
            for (int y = y0; y < y1; ++y) push_px(x1, y, x1 - 1, y);
        }
        if (!pairs.empty()) {
            std::array<int, 256> hr{}, hg{}, hb{}, hgy{};
            for (const auto& pr : pairs) {
                hr[pr.or_]++;
                hg[pr.og]++;
                hb[pr.ob]++;
                hgy[rgb_to_gray(pr.or_, pr.og, pr.ob)]++;
            }
            const int n_rim = static_cast<int>(pairs.size());
            const int out_r = hist_median(hr, n_rim);
            const int out_g = hist_median(hg, n_rim);
            const int out_b = hist_median(hb, n_rim);
            bool continuous = (static_cast<int>(gray_max) - static_cast<int>(gray_min) < 24);
            if (!continuous) {
                for (const auto& pr : pairs) {
                    const int od = std::max({std::abs(int(pr.or_) - out_r),
                                             std::abs(int(pr.og) - out_g),
                                             std::abs(int(pr.ob) - out_b)});
                    const int id = std::max({std::abs(int(pr.ir) - out_r),
                                             std::abs(int(pr.ig) - out_g),
                                             std::abs(int(pr.ib) - out_b)});
                    if (od < 24 && id < 24) {
                        continuous = true;
                        break;
                    }
                }
            }
            if (continuous) {
                bg = hist_median(hgy, n_rim);
            }
        }
    }

    // Build the bubble/card body surface envelope (ported from desktop transcript.py v5).
    // Strips speech bubble tails and exterior page canvas pixels outside rounded corners
    // before glyph/line bounding boxes are extracted, preventing `[` / `【` / `1` edge artifacts.
    bool has_surface = false;
    std::vector<uint8_t> surface;
    std::vector<uint8_t> core;
    if (!tight) {
        const int sx0 = std::max(0, x0 - 3);
        const int sy0 = std::max(0, y0 - 3);
        const int sx1 = std::min(img_w, x1 + 3);
        const int sy1 = std::min(img_h, y1 + 3);
        const int sw = sx1 - sx0;
        const int sh = sy1 - sy0;
        const int ox = x0 - sx0;
        const int oy = y0 - sy0;
        const size_t snpix = static_cast<size_t>(sw) * static_cast<size_t>(sh);

        std::vector<uint8_t> bg_mask(snpix, 0);
        for (int y = 0; y < sh; ++y) {
            const uint8_t* src_row = rgb + (static_cast<size_t>(sy0 + y) * img_w + sx0) * 3;
            const size_t dst_row = static_cast<size_t>(y) * sw;
            for (int x = 0; x < sw; ++x) {
                const int dr = std::abs(static_cast<int>(src_row[x * 3 + 0]) - inside_r);
                const int dg = std::abs(static_cast<int>(src_row[x * 3 + 1]) - inside_g);
                const int db = std::abs(static_cast<int>(src_row[x * 3 + 2]) - inside_b);
                if (std::max({dr, dg, db}) < 24) {
                    bg_mask[dst_row + x] = 1;
                }
            }
        }

        std::vector<int> s_labels;
        std::vector<CompStat> s_stats;
        const int s_count = connected_components_8(bg_mask, sw, sh, s_labels, s_stats);

        std::vector<int> freq(static_cast<size_t>(s_count), 0);
        for (int y = 0; y < h; ++y) {
            const size_t row = static_cast<size_t>(oy + y) * sw + ox;
            for (int x = 0; x < w; ++x) {
                int lb = s_labels[row + x];
                if (lb > 0) freq[lb]++;
            }
        }
        int dominant = 0;
        int dom_freq = 0;
        for (int i = 1; i < s_count; ++i) {
            if (freq[i] > dom_freq) {
                dom_freq = freq[i];
                dominant = i;
            }
        }

        if (dominant > 0) {
            std::vector<uint8_t> at_frame(static_cast<size_t>(s_count), 0);
            for (int x = 0; x < sw; ++x) {
                at_frame[s_labels[x]] = 1;
                at_frame[s_labels[static_cast<size_t>(sh - 1) * sw + x]] = 1;
            }
            for (int y = 0; y < sh; ++y) {
                at_frame[s_labels[static_cast<size_t>(y) * sw]] = 1;
                at_frame[s_labels[static_cast<size_t>(y) * sw + (sw - 1)]] = 1;
            }

            const bool dom_at_frame = (at_frame[dominant] != 0);
            std::vector<uint8_t> seed(snpix, 0);
            for (size_t i = 0; i < snpix; ++i) {
                int lb = s_labels[i];
                if (lb > 0 && (dom_at_frame || !at_frame[lb])) {
                    seed[i] = 1;
                }
            }

            // Fill interior holes (letters inside the bubble) via 4-connected BFS from outer border.
            std::vector<uint8_t> ext_zero(snpix, 0);
            std::queue<int> q;
            auto try_seed_bfs = [&](int px, int py) {
                const int idx = py * sw + px;
                if (!seed[idx] && !ext_zero[idx]) {
                    ext_zero[idx] = 1;
                    q.push(idx);
                }
            };
            for (int x = 0; x < sw; ++x) {
                try_seed_bfs(x, 0);
                try_seed_bfs(x, sh - 1);
            }
            for (int y = 0; y < sh; ++y) {
                try_seed_bfs(0, y);
                try_seed_bfs(sw - 1, y);
            }
            while (!q.empty()) {
                int cur = q.front();
                q.pop();
                int cx = cur % sw;
                int cy = cur / sw;
                if (cx > 0) try_seed_bfs(cx - 1, cy);
                if (cx + 1 < sw) try_seed_bfs(cx + 1, cy);
                if (cy > 0) try_seed_bfs(cx, cy - 1);
                if (cy + 1 < sh) try_seed_bfs(cx, cy + 1);
            }

            std::vector<uint8_t> envelope(snpix, 0);
            for (size_t i = 0; i < snpix; ++i) {
                envelope[i] = ext_zero[i] ? 0 : 1;
            }
            // Vertically bridge horizontal dividers within the same card.
            for (int x = 0; x < sw; ++x) {
                int top_y = -1;
                int bot_y = -1;
                for (int y = 0; y < sh; ++y) {
                    if (envelope[static_cast<size_t>(y) * sw + x]) {
                        if (top_y < 0) top_y = y;
                        bot_y = y;
                    }
                }
                if (top_y >= 0) {
                    for (int y = top_y; y <= bot_y; ++y) {
                        envelope[static_cast<size_t>(y) * sw + x] = 1;
                    }
                }
            }

            // Vertical morphological opening with kernel (body_height, 1):
            // drops speech bubble tails (shorter than half the bubble height) while keeping the main body.
            int body_height = std::max(3, static_cast<int>(std::lround(std::min(h, w) * 0.5))) | 1;
            body_height = std::min(body_height, sh);
            std::vector<int> col_min_y(static_cast<size_t>(sw), sh);
            std::vector<int> col_max_y(static_cast<size_t>(sw), -1);
            int body_cols = 0;
            for (int x = 0; x < sw; ++x) {
                int y = 0;
                while (y < sh) {
                    if (!envelope[static_cast<size_t>(y) * sw + x]) {
                        ++y;
                        continue;
                    }
                    int start = y;
                    while (y < sh && envelope[static_cast<size_t>(y) * sw + x]) {
                        ++y;
                    }
                    if (y - start >= body_height) {
                        if (start < col_min_y[x]) col_min_y[x] = start;
                        if (y - 1 > col_max_y[x]) col_max_y[x] = y - 1;
                    }
                }
                if (col_max_y[x] >= col_min_y[x]) {
                    body_cols++;
                }
            }

            if (body_cols > 0) {
                // Compute upper & lower convex hull chains from per-column vertical extremes.
                std::vector<PointI> top_hull;
                std::vector<PointI> bot_hull;
                top_hull.reserve(static_cast<size_t>(body_cols));
                bot_hull.reserve(static_cast<size_t>(body_cols));
                for (int x = 0; x < sw; ++x) {
                    if (col_max_y[x] < col_min_y[x]) continue;
                    PointI pt_top{x, col_min_y[x]};
                    // Lower-y chain: cross <= 0 pops concave inward points (y points downward).
                    while (top_hull.size() >= 2 &&
                           cross2d(top_hull[top_hull.size() - 2], top_hull.back(), pt_top) <= 0) {
                        top_hull.pop_back();
                    }
                    top_hull.push_back(pt_top);

                    PointI pt_bot{x, col_max_y[x]};
                    // Higher-y chain: cross >= 0 pops concave inward points.
                    while (bot_hull.size() >= 2 &&
                           cross2d(bot_hull[bot_hull.size() - 2], bot_hull.back(), pt_bot) >= 0) {
                        bot_hull.pop_back();
                    }
                    bot_hull.push_back(pt_bot);
                }

                // Rasterize convex hull column-by-column into envelope.
                std::fill(envelope.begin(), envelope.end(), 0);
                const int hx_min = top_hull.front().x;
                const int hx_max = top_hull.back().x;
                size_t ti = 0;
                size_t bi = 0;
                for (int x = hx_min; x <= hx_max; ++x) {
                    while (ti + 1 < top_hull.size() && top_hull[ti + 1].x < x) ++ti;
                    while (bi + 1 < bot_hull.size() && bot_hull[bi + 1].x < x) ++bi;

                    int yt = top_hull[ti].y;
                    if (ti + 1 < top_hull.size() && top_hull[ti + 1].x > top_hull[ti].x) {
                        const auto& a = top_hull[ti];
                        const auto& b = top_hull[ti + 1];
                        const double frac = static_cast<double>(x - a.x) / static_cast<double>(b.x - a.x);
                        yt = static_cast<int>(std::floor(a.y + (b.y - a.y) * frac + 1e-6));
                    }
                    int yb = bot_hull[bi].y;
                    if (bi + 1 < bot_hull.size() && bot_hull[bi + 1].x > bot_hull[bi].x) {
                        const auto& a = bot_hull[bi];
                        const auto& b = bot_hull[bi + 1];
                        const double frac = static_cast<double>(x - a.x) / static_cast<double>(b.x - a.x);
                        yb = static_cast<int>(std::ceil(a.y + (b.y - a.y) * frac - 1e-6));
                    }
                    yt = std::clamp(yt, 0, sh - 1);
                    yb = std::clamp(yb, 0, sh - 1);
                    for (int y = yt; y <= yb; ++y) {
                        envelope[static_cast<size_t>(y) * sw + x] = 1;
                    }
                }

                // 3x3 erosion of envelope -> core (matches OpenCV BORDER_CONSTANT=1 at frame).
                surface.assign(npix, 0);
                core.assign(npix, 0);
                for (int y = 0; y < h; ++y) {
                    const int sy = oy + y;
                    const size_t dst_row = static_cast<size_t>(y) * w;
                    for (int x = 0; x < w; ++x) {
                        const int sx = ox + x;
                        const uint8_t sv = envelope[static_cast<size_t>(sy) * sw + sx];
                        surface[dst_row + x] = sv;
                        if (sv) {
                            uint8_t eroded = 1;
                            for (int dy = -1; dy <= 1 && eroded; ++dy) {
                                const int ny = sy + dy;
                                if (ny < 0 || ny >= sh) continue;
                                for (int dx = -1; dx <= 1; ++dx) {
                                    const int nx = sx + dx;
                                    if (nx < 0 || nx >= sw) continue;
                                    if (!envelope[static_cast<size_t>(ny) * sw + nx]) {
                                        eroded = 0;
                                        break;
                                    }
                                }
                            }
                            core[dst_row + x] = eroded;
                        }
                    }
                }
                has_surface = true;
            }
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

        int final_comp_count = comp_count;
        if (has_surface) {
            std::vector<uint8_t> exterior(static_cast<size_t>(comp_count), 0);
            for (size_t i = 0; i < npix; ++i) {
                if (surface[i] == 0 && labels[i] > 0) {
                    exterior[labels[i]] = 1;
                }
            }
            std::vector<uint8_t> from_exterior_px(npix, 0);
            std::vector<uint8_t> clipped_ink(npix, 0);
            for (size_t i = 0; i < npix; ++i) {
                const int lb = labels[i];
                if (lb > 0) {
                    if (exterior[lb]) from_exterior_px[i] = 1;
                    if (valid[lb] && surface[i] != 0) clipped_ink[i] = 1;
                }
            }
            final_comp_count = connected_components_8(clipped_ink, w, h, labels, stats);
            std::vector<uint8_t> enters_body(static_cast<size_t>(final_comp_count), 0);
            std::vector<uint8_t> comp_from_ext(static_cast<size_t>(final_comp_count), 0);
            for (size_t i = 0; i < npix; ++i) {
                const int lb = labels[i];
                if (lb > 0) {
                    if (core[i] != 0) enters_body[lb] = 1;
                    if (from_exterior_px[i] != 0) comp_from_ext[lb] = 1;
                }
            }
            valid.assign(static_cast<size_t>(final_comp_count), 0);
            for (int i = 1; i < final_comp_count; ++i) {
                if (enters_body[i] || !comp_from_ext[i]) {
                    valid[i] = 1;
                }
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

        for (int i = 1; i < final_comp_count; ++i) {
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
