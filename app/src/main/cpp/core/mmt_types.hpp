#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace mmt {

inline constexpr int kNumClasses = 11;
inline constexpr int kTruncatedClassId = 9;
inline constexpr float kDisplayScoreFloor = 0.25f;
inline constexpr float kFusionIouThreshold = 0.55f;

inline constexpr std::array<const char*, kNumClasses> kClassNames = {
    "avatar",
    "speaker_name",
    "bubble_left",
    "bubble_right",
    "narration",
    "insert_image",
    "reply",
    "reply_item",
    "bond",
    "truncated_region",
    "message_note",
};

struct BoxI {
    int x0 = 0;
    int y0 = 0;
    int x1 = 0;
    int y1 = 0;

    [[nodiscard]] int width() const { return x1 - x0; }
    [[nodiscard]] int height() const { return y1 - y0; }
    [[nodiscard]] int64_t area() const { return static_cast<int64_t>(width()) * height(); }
};

struct BoxF {
    float x0 = 0.0f;
    float y0 = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;

    [[nodiscard]] float width() const { return x1 - x0; }
    [[nodiscard]] float height() const { return y1 - y0; }
    [[nodiscard]] float area() const { return width() * height(); }
};

struct Window {
    int x0 = 0;
    int y0 = 0;
    int x1 = 0;
    int y1 = 0;

    [[nodiscard]] int width() const { return x1 - x0; }
    [[nodiscard]] int height() const { return y1 - y0; }
};

struct MappedCandidate {
    int class_id = 0;
    float score = 0.0f;
    BoxF box;
    int window_id = 0;
    bool window_edge_touch = false;
    bool crosses_crop = false;
};

struct DetectedElement {
    std::string id;
    int class_id = 0;
    float score = 0.0f;
    BoxI box;
    BoxF visible_box;
    std::string parent_id;
    bool requires_review = false;
};

}  // namespace mmt
