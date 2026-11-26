#pragma once

#include <cstdint>
#include <vector>

#include "mmt_types.hpp"

namespace mmt {

/// Fixed full-width round(2.2W) windows with round(1.1W) stride and bottom-anchored tail.
std::vector<Window> large_only_windows(int width, int height);

/// Cut tall RGB24 screenshots on uniform page-canvas blank bands to avoid slicing bubbles.
/// Falls back locally to 50%-overlap 2.2W / 1.1W windows when no safe blank band exists.
std::vector<Window> adaptive_blank_windows(
    const uint8_t* rgb,
    int width,
    int height,
    float min_ratio = 1.1f,
    float max_ratio = 2.2f);

}  // namespace mmt
