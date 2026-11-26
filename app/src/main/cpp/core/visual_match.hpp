#pragma once

#include <cstdint>
#include <vector>

#include "mmt_types.hpp"

namespace mmt {

struct Stage2SkinScore {
    int best_index = 0;
    float diff_zncc = -1.0f;
    float struct_zncc = -1.0f;
};

/// Splits wide/tall multi-character avatar group boxes into up to `max_members` square sub-boxes.
std::vector<BoxI> split_avatar_box(const BoxI& box, int max_members = 5);

/// Prepares a single CCIP input tensor [3, 384, 384] (ImageNet mean/std normalized RGB):
/// - If `circle_mask == true`: center-squares crop and masks outside `r = 0.47 * side` to white (255).
/// - Also outputs crop luminance std via `out_std` (for blank-avatar / blank-image rejection).
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
    float* out_aspect);

/// Computes a 16x16 RGB thumbnail average for fast intra-page visual deduplication.
void compute_rgb_thumbnail(
    const uint8_t* rgb,
    int img_w,
    int img_h,
    const BoxI& box,
    bool circle_mask,
    int thumb_size,
    float* out_thumb_rgb);

/// Stage-2 skin expression alignment + variance-weighted ZNCC across `count` templates of `tpl_size x tpl_size`.
/// - `stack_rgb`: `count * tpl_size * tpl_size * 3` RGB uint8 templates (in original catalog order)
/// - `var_weights`: `tpl_size * tpl_size` float32 variance weights
Stage2SkinScore score_stage2_skin_templates(
    const uint8_t* rgb,
    int img_w,
    int img_h,
    const BoxI& box,
    const uint8_t* stack_rgb,
    int count,
    int tpl_size,
    const float* var_weights);

/// Stage-2 emoji structure ZNCC against a single `tpl_size x tpl_size` RGB template.
float score_emoji_structure_zncc(
    const uint8_t* rgb,
    int img_w,
    int img_h,
    const BoxI& box,
    const uint8_t* tpl_rgb,
    int tpl_size);

}  // namespace mmt
