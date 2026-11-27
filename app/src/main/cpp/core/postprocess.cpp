#include "postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <tuple>

namespace mmt {
namespace {

float fast_sigmoid(float x) {
    if (x < -60.0f) x = -60.0f;
    if (x > 60.0f) x = 60.0f;
    return 1.0f / (1.0f + std::exp(-x));
}

float box_iou(const BoxF& a, const BoxF& b) {
    const float ix0 = std::max(a.x0, b.x0);
    const float iy0 = std::max(a.y0, b.y0);
    const float ix1 = std::min(a.x1, b.x1);
    const float iy1 = std::min(a.y1, b.y1);
    if (ix1 <= ix0 || iy1 <= iy0) return 0.0f;
    const float inter = (ix1 - ix0) * (iy1 - iy0);
    const float denom = a.area() + b.area() - inter;
    return denom > 0.0f ? (inter / denom) : 0.0f;
}

bool contains_box(const BoxI& outer, const BoxI& inner) {
    return outer.x0 <= inner.x0 && outer.y0 <= inner.y0 &&
           outer.x1 >= inner.x1 && outer.y1 >= inner.y1;
}

}  // namespace

std::vector<MappedCandidate> decode_window_candidates(
    const float* bboxes,
    const float* logits,
    int num_queries,
    const Window& window,
    int window_id,
    int image_width,
    int image_height,
    float score_floor) {
    struct Row {
        float score;
        int flat_idx;
        int q;
        int cls;
    };

    const int total = num_queries * kNumClasses;
    std::vector<Row> rows;
    rows.reserve(64);
    for (int i = 0; i < total; ++i) {
        const float sc = fast_sigmoid(logits[i]);
        if (sc >= score_floor) {
            rows.push_back(Row{sc, i, i / kNumClasses, i % kNumClasses});
        }
    }
    if (rows.empty()) return {};

    std::stable_sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
        if (a.score != b.score) return a.score > b.score;
        return a.flat_idx < b.flat_idx;
    });
    if (rows.size() > 300) {
        rows.resize(300);
    }

    const float cw = static_cast<float>(window.width());
    const float ch = static_cast<float>(window.height());
    const float eff_h = std::max(ch, cw);
    const bool is_padded = eff_h > ch;

    std::vector<uint8_t> seen_q(static_cast<size_t>(num_queries), 0);
    std::vector<BoxF> active_tr;
    struct BodyRow {
        int cls;
        float score;
        BoxF local;
    };
    std::vector<BodyRow> body;
    body.reserve(rows.size());

    for (const Row& r : rows) {
        const float* bb = bboxes + static_cast<size_t>(r.q) * 4;
        const float cx = bb[0], cy = bb[1], bw = bb[2], bh = bb[3];
        const BoxF local{
            (cx - 0.5f * bw) * cw,
            (cy - 0.5f * bh) * eff_h,
            (cx + 0.5f * bw) * cw,
            (cy + 0.5f * bh) * eff_h,
        };
        if (local.x1 <= local.x0 || local.y1 <= local.y0) continue;
        if (is_padded && local.y0 >= ch) continue;
        if (seen_q[static_cast<size_t>(r.q)]) continue;
        seen_q[static_cast<size_t>(r.q)] = 1;

        if (r.cls == kTruncatedClassId) {
            const BoxF clipped{
                std::max(0.0f, local.x0),
                std::max(0.0f, local.y0),
                std::min(cw, local.x1),
                std::min(ch, local.y1),
            };
            if (clipped.x0 < clipped.x1 && clipped.y0 < clipped.y1) {
                active_tr.push_back(clipped);
            }
        } else {
            body.push_back(BodyRow{r.cls, r.score, local});
        }
    }

    const bool edge_l = window.x0 > 0;
    const bool edge_t = window.y0 > 0;
    const bool edge_r = window.x1 < image_width;
    const bool edge_b = window.y1 < image_height;

    std::vector<MappedCandidate> out;
    out.reserve(body.size());
    for (const BodyRow& b : body) {
        const float b_area = std::max(1.0f, b.local.area());
        bool suppressed = false;
        for (const BoxF& tr : active_tr) {
            const float ix0 = std::max(b.local.x0, tr.x0);
            const float iy0 = std::max(b.local.y0, tr.y0);
            const float ix1 = std::min(b.local.x1, tr.x1);
            const float iy1 = std::min(b.local.y1, tr.y1);
            if (ix0 < ix1 && iy0 < iy1) {
                // Require >= 10% overlap of the candidate box area so a 2-5px grazing
                // contact from an adjacent huge truncated photo never kills a complete reply/bubble.
                const float inter = (ix1 - ix0) * (iy1 - iy0);
                if (inter / b_area >= 0.10f) {
                    suppressed = true;
                    break;
                }
            }
        }
        if (suppressed) continue;

        const bool tl = b.local.x0 <= 0.0f;
        const bool tt = b.local.y0 <= 0.0f;
        const bool tr = b.local.x1 >= cw;
        const bool tb = b.local.y1 >= ch;
        const bool crosses = b.local.x0 < 0.0f || b.local.y0 < 0.0f || b.local.x1 > cw || b.local.y1 > ch;
        const bool edge_touch = (edge_l && tl) || (edge_t && tt) || (edge_r && tr) || (edge_b && tb);

        out.push_back(MappedCandidate{
            b.cls,
            b.score,
            BoxF{
                b.local.x0 + static_cast<float>(window.x0),
                b.local.y0 + static_cast<float>(window.y0),
                b.local.x1 + static_cast<float>(window.x0),
                b.local.y1 + static_cast<float>(window.y0),
            },
            window_id,
            edge_touch,
            crosses,
        });
    }
    return out;
}

std::vector<DetectedElement> fuse_and_finalize(
    std::vector<MappedCandidate> mapped,
    int image_width,
    int image_height,
    float iou_threshold) {
    std::stable_sort(mapped.begin(), mapped.end(), [](const MappedCandidate& a, const MappedCandidate& b) {
        if (a.score != b.score) return a.score > b.score;
        if (a.window_edge_touch != b.window_edge_touch) return !a.window_edge_touch && b.window_edge_touch;
        if (a.class_id != b.class_id) return std::string(kClassNames[a.class_id]) < std::string(kClassNames[b.class_id]);
        auto ta = std::tie(a.box.x0, a.box.y0, a.box.x1, a.box.y1, a.window_id);
        auto tb = std::tie(b.box.x0, b.box.y0, b.box.x1, b.box.y1, b.window_id);
        return ta < tb;
    });

    std::vector<MappedCandidate> kept;
    kept.reserve(mapped.size());
    for (const MappedCandidate& row : mapped) {
        bool absorbed = false;
        for (const MappedCandidate& prev : kept) {
            if (prev.class_id == row.class_id && box_iou(prev.box, row.box) >= iou_threshold) {
                absorbed = true;
                break;
            }
        }
        if (!absorbed) {
            kept.push_back(row);
        }
    }

    const float fw = static_cast<float>(image_width);
    const float fh = static_cast<float>(image_height);
    std::vector<DetectedElement> elements;
    elements.reserve(kept.size());
    int elem_idx = 0;

    for (const MappedCandidate& row : kept) {
        const BoxF vis{
            std::max(0.0f, row.box.x0),
            std::max(0.0f, row.box.y0),
            std::min(fw, row.box.x1),
            std::min(fh, row.box.y1),
        };
        if (vis.x0 >= vis.x1 || vis.y0 >= vis.y1) continue;
        const bool clipped = (vis.x0 != row.box.x0 || vis.y0 != row.box.y0 ||
                              vis.x1 != row.box.x1 || vis.y1 != row.box.y1);
        const BoxI ibox{
            std::max(0, static_cast<int>(std::floor(vis.x0))),
            std::max(0, static_cast<int>(std::floor(vis.y0))),
            std::min(image_width, static_cast<int>(std::ceil(vis.x1))),
            std::min(image_height, static_cast<int>(std::ceil(vis.y1))),
        };
        if (ibox.x0 >= ibox.x1 || ibox.y0 >= ibox.y1) continue;

        char id_buf[16];
        std::snprintf(id_buf, sizeof(id_buf), "e%04d", ++elem_idx);
        elements.push_back(DetectedElement{
            std::string(id_buf),
            row.class_id,
            row.score,
            ibox,
            vis,
            "",
            clipped || row.crosses_crop || row.window_edge_touch,
        });
    }

    // Link reply_item (7) -> smallest containing reply (6)
    for (DetectedElement& child : elements) {
        if (child.class_id != 7) continue;
        int best_idx = -1;
        int64_t best_area = 0;
        float best_score = -1.0f;
        for (size_t i = 0; i < elements.size(); ++i) {
            const DetectedElement& p = elements[i];
            if (p.class_id != 6 || !contains_box(p.box, child.box)) continue;
            const int64_t a = p.box.area();
            if (best_idx < 0 || a < best_area || (a == best_area && p.score > best_score)) {
                best_idx = static_cast<int>(i);
                best_area = a;
                best_score = p.score;
            }
        }
        if (best_idx >= 0) {
            child.parent_id = elements[static_cast<size_t>(best_idx)].id;
        } else {
            child.requires_review = true;
        }
    }

    return elements;
}

}  // namespace mmt
