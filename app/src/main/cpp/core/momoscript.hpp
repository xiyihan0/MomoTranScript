#pragma once

#include <string>
#include <vector>

#include "mmt_types.hpp"

namespace mmt {

struct RecognizedRegion {
    std::string id;
    int class_id = 0;
    float score = 0.0f;
    BoxI box;
    std::string parent_id;
    std::string text;  // Joined OCR text lines (empty if non-text or unreadable)
};

struct AssembledMessage {
    /// "bubble_left", "bubble_right", "narration", "insert_image", "reply", "bond"
    std::string kind;
    std::string speaker;          // Empty for implicit right or non-dialogue
    bool is_right = false;
    bool is_first_turn = false;
    std::string content;          // Plaintext or newline-joined reply items
    std::string note;             // Associated outward message_note footer (if any)
    BoxI box;
};

struct AssemblyResult {
    std::string momoscript_text;
    std::vector<AssembledMessage> messages;
};

/// Joins multi-line OCR text following MomoScript convention (no artificial space between CJK;
/// single space between wrapped ASCII alnum/punctuation words).
std::string join_plain_text(const std::string& multiline);

/// Resolves speaker names via avatar/name geometry, links reply items & outward message notes,
/// and renders MomoScript-style plaintext + structured messages ready for MoeTalk export.
AssemblyResult assemble_momoscript(
    const std::vector<RecognizedRegion>& regions,
    int image_width,
    int image_height);

}  // namespace mmt
