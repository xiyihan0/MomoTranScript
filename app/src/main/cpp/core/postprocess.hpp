#pragma once

#include <vector>

#include "mmt_types.hpp"

namespace mmt {

/// Decode a single window's PP-DocLayoutV3 outputs:
/// - bboxes: [num_queries, 4] normalized cx, cy, w, h
/// - logits: [num_queries, 11] raw class logits
/// Applies exact V2 query-winner + same-window TR (>=0.25) suppression + translation.
std::vector<MappedCandidate> decode_window_candidates(
    const float* bboxes,
    const float* logits,
    int num_queries,
    const Window& window,
    int window_id,
    int image_width,
    int image_height,
    float score_floor = kDisplayScoreFloor);

/// Fuse cross-window same-class boxes (IoU >= 0.55), clip to visible image bounds,
/// round to integer xyxy, and link reply_item -> smallest containing reply parent.
std::vector<DetectedElement> fuse_and_finalize(
    std::vector<MappedCandidate> mapped,
    int image_width,
    int image_height,
    float iou_threshold = kFusionIouThreshold);

}  // namespace mmt
