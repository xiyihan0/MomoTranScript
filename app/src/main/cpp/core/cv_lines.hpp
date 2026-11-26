#pragma once

#include <cstdint>
#include <vector>

#include "mmt_types.hpp"

namespace mmt {

struct TextLineProposal {
    BoxI ink_box;
    BoxI box;
    bool is_light_fg = false;
};

/// Dual-polarity 8-connected-component physical line localizer in pure C++20.
/// Mirrors `transcript.py:physical_lines` without depending on OpenCV.
/// - `rgb`: contiguous row-major RGB24 buffer of size `img_w * img_h * 3`
/// - `roi`: integer original-pixel region box (`speaker_name`, `bubble_*`, `narration`, `reply_item`, `message_note`)
/// - `tight`: true for `speaker_name` and `message_note` (allows colored text & tighter detached-part joins)
std::vector<TextLineProposal> segment_physical_lines(
    const uint8_t* rgb,
    int img_w,
    int img_h,
    const BoxI& roi,
    bool tight = false);

/// Returns whether a detector class_id carries OCR text lines.
inline bool is_text_class(int class_id) {
    return class_id == 1 || class_id == 2 || class_id == 3 || class_id == 4 || class_id == 7 || class_id == 10;
}

/// Returns whether a detector class_id uses `tight=true` line localization.
inline bool is_tight_text_class(int class_id) {
    return class_id == 1 || class_id == 10;
}

}  // namespace mmt
