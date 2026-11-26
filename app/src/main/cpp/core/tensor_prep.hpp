#pragma once

#include <cstdint>
#include <vector>

#include "cv_lines.hpp"
#include "mmt_types.hpp"

namespace mmt {

/// Resizes window RGB crop to `dst_size x dst_size` (672x672) and packs NCHW float32 RGB / 255.0f
/// matching PP-DocLayoutV3's exact normalization (no ImageNet mean/std).
void prepare_det_input_nchw(
    const uint8_t* rgb,
    int img_w,
    int img_h,
    const Window& win,
    int dst_size,
    float* out_nchw);

/// Prepares a single physical CV line crop for PP-OCRv6 recognition:
/// - Inverts & min-max normalizes light-on-dark lines
/// - Pads with median crop background (4px L/R, 3px T/B, plus optional short-line target height)
/// - Bilinearly resizes to height `dst_h` (48), converts RGB -> BGR, normalizes `(v/255 - 0.5)/0.5`
/// Returns `out_chw` of size `3 * dst_h * out_resized_w`.
void prepare_rec_line_chw(
    const uint8_t* rgb,
    int img_w,
    int img_h,
    const TextLineProposal& line,
    int target_pad_h,
    int dst_h,
    int max_w,
    std::vector<float>& out_chw,
    int& out_resized_w);

}  // namespace mmt
