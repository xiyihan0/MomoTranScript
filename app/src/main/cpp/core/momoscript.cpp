#include "momoscript.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <unordered_map>

namespace mmt {
namespace {

constexpr const char* kUnreadable = "[unreadable]";
constexpr const char* kUnknownSpeaker = "[unknown speaker]";

bool is_ascii_byte(char c) {
    return (static_cast<unsigned char>(c) & 0x80) == 0;
}

bool is_ascii_word_end(char c) {
    if (!is_ascii_byte(c)) return false;
    return std::isalnum(static_cast<unsigned char>(c)) || c == ',' || c == ';' || c == ':' || c == '!' || c == '?';
}

bool is_ascii_word_start(char c) {
    if (!is_ascii_byte(c)) return false;
    return std::isalnum(static_cast<unsigned char>(c)) != 0;
}

std::string trim_ws(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    size_t b = s.size();
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

int64_t inter_area(const BoxI& a, const BoxI& b) {
    int iw = std::max(0, std::min(a.x1, b.x1) - std::max(a.x0, b.x0));
    int ih = std::max(0, std::min(a.y1, b.y1) - std::max(a.y0, b.y0));
    return static_cast<int64_t>(iw) * static_cast<int64_t>(ih);
}

bool box_intersects(const BoxI& a, const BoxI& b) {
    return std::min(a.x1, b.x1) > std::max(a.x0, b.x0) &&
           std::min(a.y1, b.y1) > std::max(a.y0, b.y0);
}

bool is_left_side(const BoxI& b, int width) {
    return (b.x0 + b.x1) < width;
}

struct AvatarGroup {
    std::string id;
    BoxI box;
    int anchor_y1 = 0;
    bool is_left = true;
    int speaker_name_idx = -1;
};

/// Returns anchor cost >= 0, or -1 if speaker_name cannot belong to avatar group.
float name_anchor_cost(const BoxI& n, const BoxI& a, bool left, int width) {
    const int nh = n.height();
    const bool above = (n.y1 <= a.y0) && ((a.y0 - n.y1) <= std::max(width * 0.06f, nh * 2.0f));
    const bool aligned = (n.y0 <= a.y1 + width * 0.025f) && (n.y1 >= a.y0 - width * 0.025f);
    if (!above && !aligned) return -1.0f;

    bool horiz = false;
    float gap = 0.0f;
    if (left) {
        horiz = (n.x1 >= a.x0) && (n.x0 >= a.x0 - width * 0.025f) && (n.x0 <= a.x1 + width * 0.15f);
        gap = std::max(0.0f, static_cast<float>(n.x0 - a.x1));
    } else {
        horiz = (n.x0 <= a.x1) && (n.x1 >= a.x0 - width * 0.15f) && (n.x1 <= a.x1 + width * 0.025f);
        gap = std::max(0.0f, static_cast<float>(a.x0 - n.x1));
    }
    if (above && std::min(n.x1, a.x1) > std::max(n.x0, a.x0)) {
        horiz = true;
    }
    if (!horiz) return -1.0f;

    float vgap = std::max({0.0f, static_cast<float>(a.y0 - n.y1), static_cast<float>(n.y0 - a.y1)});
    return (gap + vgap * 2.0f + std::abs(static_cast<float>(n.y0 - a.y0)) * 0.1f) / std::max(1.0f, static_cast<float>(width));
}

}  // namespace

std::string join_plain_text(const std::string& multiline) {
    std::vector<std::string> parts;
    size_t start = 0;
    while (start <= multiline.size()) {
        size_t nl = multiline.find('\n', start);
        std::string sub = (nl == std::string::npos)
                              ? multiline.substr(start)
                              : multiline.substr(start, nl - start);
        std::string t = trim_ws(sub);
        if (!t.empty()) parts.push_back(std::move(t));
        if (nl == std::string::npos) break;
        start = nl + 1;
    }
    if (parts.empty()) return kUnreadable;

    std::string res = parts[0];
    for (size_t i = 1; i < parts.size(); ++i) {
        if (!res.empty() && !parts[i].empty() &&
            is_ascii_word_end(res.back()) && is_ascii_word_start(parts[i].front())) {
            res.push_back(' ');
        }
        res += parts[i];
    }
    return res;
}

AssemblyResult assemble_momoscript(
    const std::vector<RecognizedRegion>& input_regions,
    int image_width,
    int image_height) {
    AssemblyResult out;
    if (input_regions.empty() || image_width <= 0) return out;

    // Sort regions in reading order (top -> left -> id).
    std::vector<RecognizedRegion> regions = input_regions;
    std::sort(regions.begin(), regions.end(), [](const RecognizedRegion& a, const RecognizedRegion& b) {
        if (a.box.y0 != b.box.y0) return a.box.y0 < b.box.y0;
        if (a.box.x0 != b.box.x0) return a.box.x0 < b.box.x0;
        return a.id < b.id;
    });

    std::unordered_map<std::string, int> id_to_idx;
    for (int i = 0; i < static_cast<int>(regions.size()); ++i) {
        id_to_idx[regions[i].id] = i;
    }

    // 1. Group overlapping same-side avatars into AvatarGroups.
    std::vector<int> avatar_indices;
    for (int i = 0; i < static_cast<int>(regions.size()); ++i) {
        if (regions[i].class_id == 0) avatar_indices.push_back(i);
    }
    std::vector<int> av_root(avatar_indices.size());
    for (size_t i = 0; i < av_root.size(); ++i) av_root[i] = static_cast<int>(i);
    auto find_av = [&](int x) {
        while (av_root[x] != x) {
            av_root[x] = av_root[av_root[x]];
            x = av_root[x];
        }
        return x;
    };
    for (size_t i = 0; i < avatar_indices.size(); ++i) {
        const auto& a = regions[avatar_indices[i]].box;
        const bool side_a = is_left_side(a, image_width);
        for (size_t j = 0; j < i; ++j) {
            const auto& b = regions[avatar_indices[j]].box;
            if (is_left_side(b, image_width) != side_a) continue;
            const int64_t min_a = std::min(a.area(), b.area());
            if (min_a > 0 && static_cast<double>(inter_area(a, b)) >= 0.04 * static_cast<double>(min_a)) {
                av_root[find_av(static_cast<int>(i))] = find_av(static_cast<int>(j));
            }
        }
    }

    std::unordered_map<int, AvatarGroup> group_map;
    for (size_t i = 0; i < avatar_indices.size(); ++i) {
        const int r = find_av(static_cast<int>(i));
        const auto& reg = regions[avatar_indices[i]];
        auto it = group_map.find(r);
        if (it == group_map.end()) {
            AvatarGroup g;
            g.id = "ag_" + reg.id;
            g.box = reg.box;
            g.is_left = is_left_side(reg.box, image_width);
            group_map.emplace(r, std::move(g));
        } else {
            auto& gb = it->second.box;
            gb.x0 = std::min(gb.x0, reg.box.x0);
            gb.y0 = std::min(gb.y0, reg.box.y0);
            gb.x1 = std::max(gb.x1, reg.box.x1);
            gb.y1 = std::max(gb.y1, reg.box.y1);
        }
    }
    std::vector<AvatarGroup> groups;
    groups.reserve(group_map.size());
    for (auto& kv : group_map) {
        kv.second.is_left = is_left_side(kv.second.box, image_width);
        groups.push_back(std::move(kv.second));
    }
    std::sort(groups.begin(), groups.end(), [](const AvatarGroup& a, const AvatarGroup& b) {
        if (a.box.y0 != b.box.y0) return a.box.y0 < b.box.y0;
        return a.box.x0 < b.box.x0;
    });

    // 2. Anchor speaker_name (1) -> closest AvatarGroup (if any), else keep as standalone name anchor.
    std::vector<bool> name_bound(regions.size(), false);
    for (int i = 0; i < static_cast<int>(regions.size()); ++i) {
        if (regions[i].class_id != 1) continue;
        float best_cost = 1e9f;
        int best_g = -1;
        for (int g = 0; g < static_cast<int>(groups.size()); ++g) {
            float c = name_anchor_cost(regions[i].box, groups[g].box, groups[g].is_left, image_width);
            if (c >= 0.0f && c < best_cost) {
                best_cost = c;
                best_g = g;
            }
        }
        if (best_g >= 0 && groups[best_g].speaker_name_idx < 0) {
            groups[best_g].speaker_name_idx = i;
            name_bound[i] = true;
        }
    }

    // 2b. Extend anchor_y1 across aligned insert_image(s) immediately below an avatar/name anchor
    //     (e.g., [Avatar + SpeakerName] -> [CharFace/Emoji insert_image] -> [Left Bubble]).
    struct StandaloneNameAnchor {
        int region_idx = -1;
        BoxI box;
        int anchor_y1 = 0;
        bool is_left = true;
    };
    std::vector<StandaloneNameAnchor> standalone_names;
    for (int i = 0; i < static_cast<int>(regions.size()); ++i) {
        if (regions[i].class_id == 1 && !name_bound[i]) {
            standalone_names.push_back(StandaloneNameAnchor{
                i, regions[i].box, regions[i].box.y1, is_left_side(regions[i].box, image_width),
            });
        }
    }
    auto is_reset_region = [](const RecognizedRegion& r) {
        return r.class_id == 4 || r.class_id == 6 || (r.class_id == 7 && r.parent_id.empty());
    };
    std::vector<int> image_indices;
    std::vector<int> reset_indices;
    for (int i = 0; i < static_cast<int>(regions.size()); ++i) {
        if (regions[i].class_id == 5) image_indices.push_back(i);
        if (is_reset_region(regions[i])) reset_indices.push_back(i);
    }
    auto has_intervening_reset = [&](int y_start, int y_end) {
        for (int ri : reset_indices) {
            int ry = regions[ri].box.y0;
            if (ry > y_start && ry <= y_end) return true;
        }
        return false;
    };
    auto has_intervening_same_side_anchor = [&](bool left, int y_start, int y_end, int skip_group, int skip_sn) {
        for (int gi = 0; gi < static_cast<int>(groups.size()); ++gi) {
            if (gi == skip_group || groups[gi].is_left != left) continue;
            int gy = groups[gi].box.y0;
            if (gy > y_start && gy <= y_end) return true;
        }
        for (int si = 0; si < static_cast<int>(standalone_names.size()); ++si) {
            if (si == skip_sn || standalone_names[si].is_left != left) continue;
            int sy = standalone_names[si].box.y0;
            if (sy > y_start && sy <= y_end) return true;
        }
        return false;
    };

    struct ImageAnchorInfo {
        std::string anchor_id;
        std::string speaker;
        bool is_left = true;
    };
    std::unordered_map<int, ImageAnchorInfo> image_anchors;

    for (int gi = 0; gi < static_cast<int>(groups.size()); ++gi) {
        auto& g = groups[gi];
        g.anchor_y1 = g.box.y1;
        std::string sp;
        if (g.speaker_name_idx >= 0) {
            std::string raw = trim_ws(regions[g.speaker_name_idx].text);
            if (!raw.empty()) sp = join_plain_text(raw);
        }
        for (int ii : image_indices) {
            const auto& p = regions[ii].box;
            if (p.y0 < g.box.y0 - static_cast<int>(image_width * 0.035f) ||
                p.y0 > g.anchor_y1 + static_cast<int>(image_width * 0.12f)) {
                continue;
            }
            if (has_intervening_reset(g.box.y0, p.y0)) break;
            if (has_intervening_same_side_anchor(g.is_left, g.box.y0, p.y0, gi, -1)) break;
            const bool aligned = g.is_left
                ? (p.x0 >= g.box.x0 - image_width * 0.025f && p.x0 <= g.box.x1 + image_width * 0.18f)
                : (p.x1 <= g.box.x1 + image_width * 0.025f && p.x1 >= g.box.x0 - image_width * 0.18f);
            if (aligned) {
                g.anchor_y1 = std::max(g.anchor_y1, p.y1);
                if (image_anchors.find(ii) == image_anchors.end()) {
                    image_anchors[ii] = ImageAnchorInfo{g.id, sp, g.is_left};
                }
            }
        }
    }
    for (int si = 0; si < static_cast<int>(standalone_names.size()); ++si) {
        auto& sn = standalone_names[si];
        sn.anchor_y1 = sn.box.y1;
        std::string raw = trim_ws(regions[sn.region_idx].text);
        std::string sp = raw.empty() ? "" : join_plain_text(raw);
        for (int ii : image_indices) {
            const auto& p = regions[ii].box;
            if (p.y0 < sn.box.y0 || p.y0 > sn.anchor_y1 + static_cast<int>(image_width * 0.12f)) {
                continue;
            }
            if (has_intervening_reset(sn.box.y0, p.y0)) break;
            if (has_intervening_same_side_anchor(sn.is_left, sn.box.y0, p.y0, -1, si)) break;
            if (std::min(sn.box.x1, p.x1) > std::max(sn.box.x0, p.x0)) {
                sn.anchor_y1 = std::max(sn.anchor_y1, p.y1);
                if (image_anchors.find(ii) == image_anchors.end()) {
                    image_anchors[ii] = ImageAnchorInfo{"name_" + regions[sn.region_idx].id, sp, sn.is_left};
                }
            }
        }
    }

    // 3. Associate outward message_note (10) to adjacent bubble / insert_image.
    std::unordered_map<int, std::vector<int>> owner_notes;
    std::vector<int> unanchored_notes;
    for (int ni = 0; ni < static_cast<int>(regions.size()); ++ni) {
        if (regions[ni].class_id != 10) continue;
        const auto& nb = regions[ni].box;
        int best_owner = -1;
        int cand_count = 0;
        for (int bi = 0; bi < static_cast<int>(regions.size()); ++bi) {
            const int cid = regions[bi].class_id;
            if (cid != 2 && cid != 3 && cid != 5) continue;
            const auto& bb = regions[bi].box;
            bool left_anchor = (cid == 2);
            if (cid == 5) {
                int off = (bb.x0 + bb.x1) - image_width;
                if (std::abs(off) < static_cast<int>(image_width * 0.08f)) continue;
                left_anchor = (off < 0);
            }
            const bool outward = left_anchor ? (nb.x0 >= bb.x1) : (nb.x1 <= bb.x0);
            if (!outward) continue;
            const float glyph = std::clamp(static_cast<float>(nb.height()), 5.0f, image_width * 0.05f);
            const float reach = std::max(glyph * 18.0f, image_width * 0.38f);
            BoxI strip{
                std::clamp(static_cast<int>(std::lround(left_anchor ? bb.x1 + 1 : bb.x0 - reach)), 0, image_width),
                std::clamp(static_cast<int>(std::lround(bb.y1 - glyph * 1.7f)), 0, image_height),
                std::clamp(static_cast<int>(std::lround(left_anchor ? bb.x1 + reach : bb.x0 - 1)), 0, image_width),
                std::clamp(static_cast<int>(std::lround(bb.y1 + glyph * 3.2f)), 0, image_height),
            };
            if (nb.area() > 0 && static_cast<double>(inter_area(strip, nb)) >= 0.5 * static_cast<double>(nb.area())) {
                bool blocked = false;
                for (int oi = 0; oi < static_cast<int>(regions.size()); ++oi) {
                    if (oi == ni || oi == bi) continue;
                    int oc = regions[oi].class_id;
                    if (oc == 0 || oc == 1 || oc == 10) continue;
                    if (box_intersects(regions[oi].box, nb)) {
                        blocked = true;
                        break;
                    }
                }
                if (!blocked) {
                    best_owner = bi;
                    cand_count++;
                }
            }
        }
        if (cand_count == 1 && best_owner >= 0) {
            owner_notes[best_owner].push_back(ni);
        } else {
            unanchored_notes.push_back(ni);
        }
    }

    // 4. Collect reply_item (7) children per reply (6).
    std::unordered_map<std::string, std::vector<int>> reply_children;
    for (int i = 0; i < static_cast<int>(regions.size()); ++i) {
        if (regions[i].class_id == 7 && !regions[i].parent_id.empty()) {
            reply_children[regions[i].parent_id].push_back(i);
        }
    }

    // 5. Walk regions in reading order, tracking active left/right speaker anchor.
    std::string active_left_speaker;
    std::string active_left_anchor;
    std::string active_right_speaker;
    std::string active_right_anchor;

    std::string last_emitted_anchor;
    int last_emitted_side = -1;  // 0=left, 1=right

    // Pre-index standalone speaker_name and avatar groups for top-to-bottom anchor updates.
    auto find_bubble_anchor = [&](int reg_idx, bool left) -> std::pair<std::string, std::string> {
        const auto& b = regions[reg_idx].box;
        // First check avatar groups matching this bubble's top Band (including extended insert_image span).
        for (int gi = static_cast<int>(groups.size()) - 1; gi >= 0; --gi) {
            const auto& g = groups[gi];
            if (g.is_left != left) continue;
            float y_lo = g.box.y0 - image_width * 0.035f;
            float y_hi = g.anchor_y1 + image_width * 0.10f;
            if (b.y0 >= y_lo && b.y0 <= y_hi) {
                if (has_intervening_reset(g.box.y0, b.y0)) continue;
                const bool side_ok = left ? (b.x1 >= g.box.x0) : (b.x0 <= g.box.x1);
                if (side_ok) {
                    std::string sp;
                    if (g.speaker_name_idx >= 0) {
                        std::string raw = trim_ws(regions[g.speaker_name_idx].text);
                        if (!raw.empty()) sp = join_plain_text(raw);
                    }
                    return {g.id, sp};
                }
            }
        }
        // Fallback: check unbound speaker_name immediately above this bubble (or across insert_image).
        for (int si = static_cast<int>(standalone_names.size()) - 1; si >= 0; --si) {
            const auto& sn = standalone_names[si];
            if (sn.is_left != left) continue;
            if (sn.box.y0 <= b.y0 + image_width * 0.02f &&
                b.y0 <= sn.anchor_y1 + image_width * 0.10f &&
                !has_intervening_reset(sn.box.y0, b.y0)) {
                std::string raw = trim_ws(regions[sn.region_idx].text);
                if (!raw.empty()) {
                    return {"name_" + regions[sn.region_idx].id, join_plain_text(raw)};
                }
            }
        }
        return {"", ""};
    };

    std::vector<std::string> lines_out;
    for (int i = 0; i < static_cast<int>(regions.size()); ++i) {
        const auto& rec = regions[i];
        const int cid = rec.class_id;

        auto collect_notes = [&]() -> std::string {
            auto it = owner_notes.find(i);
            if (it == owner_notes.end() || it->second.empty()) return "";
            std::string joined;
            for (int ni : it->second) {
                if (!joined.empty()) joined += " | ";
                joined += join_plain_text(regions[ni].text);
            }
            return joined;
        };

        if (cid == 2 || cid == 3) {
            const bool is_right = (cid == 3);
            const bool left = !is_right;
            auto [found_anchor, found_speaker] = find_bubble_anchor(i, left);

            if (!found_anchor.empty()) {
                if (left) {
                    active_left_anchor = found_anchor;
                    active_left_speaker = found_speaker;
                } else {
                    active_right_anchor = found_anchor;
                    active_right_speaker = found_speaker;
                }
            }

            const std::string& cur_anchor = left ? active_left_anchor : active_right_anchor;
            const std::string& cur_speaker = left ? active_left_speaker : active_right_speaker;
            const std::string effective_anchor = cur_anchor.empty() ? rec.id : cur_anchor;
            const int side_num = is_right ? 1 : 0;

            std::string speaker_label = cur_speaker;
            if (speaker_label.empty() && left) {
                speaker_label = kUnknownSpeaker;
            }

            const bool is_first_turn = (effective_anchor != last_emitted_anchor) || (side_num != last_emitted_side);
            const bool emit_name = !speaker_label.empty() && is_first_turn;
            const std::string body = join_plain_text(rec.text);
            const std::string note = collect_notes();

            std::string line = is_right ? "< " : "> ";
            if (emit_name) {
                line += speaker_label + ": ";
            }
            line += body;
            if (!note.empty()) {
                line += " | " + note;
            }
            lines_out.push_back(std::move(line));

            out.messages.push_back(AssembledMessage{
                is_right ? "bubble_right" : "bubble_left",
                cur_speaker,
                is_right,
                is_first_turn,
                body,
                note,
                rec.box,
            });
            last_emitted_anchor = effective_anchor;
            last_emitted_side = side_num;
        } else if (cid == 4) {
            const std::string body = join_plain_text(rec.text);
            lines_out.push_back("- " + body);
            out.messages.push_back(AssembledMessage{"narration", "", false, false, body, "", rec.box});
            last_emitted_anchor.clear();
            last_emitted_side = -1;
            active_left_anchor.clear();
            active_left_speaker.clear();
            active_right_anchor.clear();
            active_right_speaker.clear();
        } else if (cid == 5 || cid == 8) {
            const std::string note = collect_notes();
            auto ia_it = (cid == 5) ? image_anchors.find(i) : image_anchors.end();
            if (ia_it != image_anchors.end() && !ia_it->second.anchor_id.empty()) {
                const auto& ia = ia_it->second;
                const bool is_right = !ia.is_left;
                const int side_num = is_right ? 1 : 0;
                if (ia.is_left) {
                    active_left_anchor = ia.anchor_id;
                    active_left_speaker = ia.speaker;
                } else {
                    active_right_anchor = ia.anchor_id;
                    active_right_speaker = ia.speaker;
                }
                std::string speaker_label = ia.speaker;
                if (speaker_label.empty() && ia.is_left) {
                    speaker_label = kUnknownSpeaker;
                }
                const bool is_first_turn = (ia.anchor_id != last_emitted_anchor) || (side_num != last_emitted_side);
                const bool emit_name = !speaker_label.empty() && is_first_turn;
                std::string line = is_right ? "< " : "> ";
                if (emit_name) {
                    line += speaker_label + ": ";
                }
                line += "[image]";
                if (!note.empty()) line += " | " + note;
                lines_out.push_back(std::move(line));
                out.messages.push_back(AssembledMessage{
                    "insert_image",
                    ia.speaker,
                    is_right,
                    is_first_turn,
                    "",
                    note,
                    rec.box,
                });
                last_emitted_anchor = ia.anchor_id;
                last_emitted_side = side_num;
            } else {
                std::string line = "[image]";
                if (!note.empty()) line += " | " + note;
                lines_out.push_back(std::move(line));
                out.messages.push_back(AssembledMessage{
                    cid == 8 ? "bond" : "insert_image",
                    "",
                    false,
                    false,
                    cid == 8 ? join_plain_text(rec.text) : "",
                    note,
                    rec.box,
                });
            }
        } else if (cid == 6) {
            lines_out.push_back("@reply");
            std::string joined_items;
            auto it = reply_children.find(rec.id);
            if (it != reply_children.end() && !it->second.empty()) {
                for (int ci : it->second) {
                    std::string ct = join_plain_text(regions[ci].text);
                    lines_out.push_back("- " + ct);
                    if (!joined_items.empty()) joined_items.push_back('\n');
                    joined_items += ct;
                }
            } else {
                lines_out.push_back(std::string("- ") + kUnreadable);
                joined_items = kUnreadable;
            }
            lines_out.push_back("@end");
            out.messages.push_back(AssembledMessage{"reply", "", false, false, joined_items, "", rec.box});
            last_emitted_anchor.clear();
            last_emitted_side = -1;
            active_left_anchor.clear();
            active_left_speaker.clear();
            active_right_anchor.clear();
            active_right_speaker.clear();
        } else if (cid == 7 && rec.parent_id.empty()) {
            std::string ct = join_plain_text(rec.text);
            lines_out.push_back("@reply");
            lines_out.push_back("- " + ct);
            lines_out.push_back("@end");
            out.messages.push_back(AssembledMessage{"reply", "", false, false, ct, "", rec.box});
            last_emitted_anchor.clear();
            last_emitted_side = -1;
        }
    }

    for (int ni : unanchored_notes) {
        const auto& n = regions[ni];
        lines_out.push_back("# 待核附注 " + n.id + ": " + join_plain_text(n.text));
    }

    for (const auto& l : lines_out) {
        out.momoscript_text += l;
        out.momoscript_text.push_back('\n');
    }
    return out;
}

}  // namespace mmt
