#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "../avifs_pack.hpp"
#include "../cv_lines.hpp"
#include "../momoscript.hpp"
#include "../postprocess.hpp"
#include "../tensor_prep.hpp"
#include "../visual_match.hpp"
#include "../windowing.hpp"

namespace {

void test_adaptive_blank_windows() {
    const int w = 200;
    const int h = 1000;
    std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3, 250);
    auto fill_rect = [&](int x0, int y0, int x1, int y1, uint8_t val) {
        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) {
                uint8_t* p = rgb.data() + (static_cast<size_t>(y) * w + x) * 3;
                p[0] = p[1] = p[2] = val;
            }
        }
    };
    fill_rect(20, 40, 180, 320, 60);
    fill_rect(20, 380, 180, 680, 60);
    fill_rect(20, 740, 180, 960, 60);

    auto wins = mmt::adaptive_blank_windows(rgb.data(), w, h);
    assert(wins.size() == 3);
    assert(wins.front().y0 == 0);
    assert(wins.back().y1 == h);
    assert(wins[0].y1 >= 320 && wins[0].y1 <= 380);
    assert(wins[1].y0 == wins[0].y1);
    std::cout << "[PASS] test_adaptive_blank_windows (" << wins.size() << " windows)\n";
}

void test_postprocess_tr_fusion_and_reply() {
    const int Q = 6;
    std::vector<float> bboxes(static_cast<size_t>(Q) * 4, 0.0f);
    std::vector<float> logits(static_cast<size_t>(Q) * mmt::kNumClasses, -10.0f);

    bboxes[0 * 4 + 0] = 0.5f; bboxes[0 * 4 + 1] = 0.075f; bboxes[0 * 4 + 2] = 1.0f; bboxes[0 * 4 + 3] = 0.15f;
    logits[0 * mmt::kNumClasses + mmt::kTruncatedClassId] = 4.0f;

    bboxes[1 * 4 + 0] = 0.4f; bboxes[1 * 4 + 1] = 0.10f; bboxes[1 * 4 + 2] = 0.5f; bboxes[1 * 4 + 3] = 0.08f;
    logits[1 * mmt::kNumClasses + 2] = 3.0f;

    bboxes[2 * 4 + 0] = 0.4f; bboxes[2 * 4 + 1] = 0.40f; bboxes[2 * 4 + 2] = 0.5f; bboxes[2 * 4 + 3] = 0.10f;
    logits[2 * mmt::kNumClasses + 2] = 3.5f;
    logits[2 * mmt::kNumClasses + 3] = 1.5f;

    bboxes[3 * 4 + 0] = 0.5f; bboxes[3 * 4 + 1] = 0.70f; bboxes[3 * 4 + 2] = 0.8f; bboxes[3 * 4 + 3] = 0.20f;
    logits[3 * mmt::kNumClasses + 6] = 4.0f;
    bboxes[4 * 4 + 0] = 0.5f; bboxes[4 * 4 + 1] = 0.70f; bboxes[4 * 4 + 2] = 0.6f; bboxes[4 * 4 + 3] = 0.08f;
    logits[4 * mmt::kNumClasses + 7] = 3.5f;

    mmt::Window win{0, 0, 500, 1000};
    auto cands = mmt::decode_window_candidates(bboxes.data(), logits.data(), Q, win, 0, 500, 1000);
    assert(cands.size() == 3);

    mmt::MappedCandidate dup = cands[0];
    dup.window_id = 1;
    dup.score -= 0.05f;
    cands.push_back(dup);

    auto elems = mmt::fuse_and_finalize(std::move(cands), 500, 1000);
    assert(elems.size() == 3);
    bool linked_reply = false;
    for (const auto& e : elems) {
        if (e.class_id == 7) {
            assert(!e.parent_id.empty());
            linked_reply = true;
        }
    }
    assert(linked_reply);
    std::cout << "[PASS] test_postprocess_tr_fusion_and_reply\n";
}

void test_avifs_pack_reader() {
    const std::string tmp_path = "/tmp/test_mmt_avifs_pack.bin";
    const std::string hdr = R"({"schema_version":1,"tpl_size":64,"skin_count":2,"expression_count":5,"emoji_count":3})";
    const std::string payload = "HELLO_AVIFS_PAYLOAD";
    {
        std::ofstream f(tmp_path, std::ios::binary);
        f.write("MMTAVIF1", 8);
        uint32_t n = static_cast<uint32_t>(hdr.size());
        uint8_t le[4] = {
            static_cast<uint8_t>(n & 0xff),
            static_cast<uint8_t>((n >> 8) & 0xff),
            static_cast<uint8_t>((n >> 16) & 0xff),
            static_cast<uint8_t>((n >> 24) & 0xff),
        };
        f.write(reinterpret_cast<const char*>(le), 4);
        f.write(hdr.data(), static_cast<std::streamsize>(hdr.size()));
        f.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    }
    mmt::AvifsPackReader reader;
    std::string err;
    assert(reader.open(tmp_path, &err));
    assert(reader.index().schema_version == 1);
    assert(reader.index().tpl_size == 64);
    assert(reader.index().skin_count == 2);
    assert(reader.index().expression_count == 5);
    assert(reader.index().emoji_count == 3);
    std::vector<uint8_t> slice;
    assert(reader.read_payload_slice(6, 5, slice));
    assert(std::string(slice.begin(), slice.end()) == "AVIFS");
    std::remove(tmp_path.c_str());
    std::cout << "[PASS] test_avifs_pack_reader\n";
}

void test_cv_lines_and_tensor_prep() {
    const int w = 240;
    const int h = 100;
    // White bubble with 2 rows of dark glyph-like blocks
    std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3, 245);
    auto draw_glyph = [&](int gx, int gy, int gw, int gh, uint8_t r, uint8_t g, uint8_t b) {
        for (int y = gy; y < gy + gh; ++y) {
            for (int x = gx; x < gx + gw; ++x) {
                uint8_t* p = rgb.data() + (static_cast<size_t>(y) * w + x) * 3;
                p[0] = r; p[1] = g; p[2] = b;
            }
        }
    };
    for (int k = 0; k < 6; ++k) {
        draw_glyph(16 + k * 24, 16, 14, 16, 30, 30, 30);
        draw_glyph(16 + k * 24, 54, 14, 16, 30, 30, 30);
    }

    auto lines = mmt::segment_physical_lines(rgb.data(), w, h, mmt::BoxI{0, 0, w, h}, false);
    assert(lines.size() == 2);
    assert(!lines[0].is_light_fg);
    assert(lines[0].ink_box.y0 == 16 && lines[0].ink_box.y1 == 32);
    assert(lines[1].ink_box.y0 == 54 && lines[1].ink_box.y1 == 70);

    // Verify light-on-dark bubble (like MomoTalk dark blue-gray bubble)
    std::vector<uint8_t> dark_rgb(static_cast<size_t>(w) * 60 * 3);
    for (size_t i = 0; i < static_cast<size_t>(w) * 60; ++i) {
        dark_rgb[i * 3 + 0] = 68;
        dark_rgb[i * 3 + 1] = 82;
        dark_rgb[i * 3 + 2] = 96;
    }
    for (int k = 0; k < 5; ++k) {
        for (int y = 20; y < 38; ++y) {
            for (int x = 18 + k * 22; x < 30 + k * 22; ++x) {
                uint8_t* p = dark_rgb.data() + (static_cast<size_t>(y) * w + x) * 3;
                p[0] = 250; p[1] = 250; p[2] = 250;
            }
        }
    }
    auto light_lines = mmt::segment_physical_lines(dark_rgb.data(), w, 60, mmt::BoxI{0, 0, w, 60}, false);
    assert(light_lines.size() == 1);
    assert(light_lines[0].is_light_fg);

    // Verify tensor prep dimensions & normalization range
    std::vector<float> det_in(3 * 64 * 64, -1.0f);
    mmt::prepare_det_input_nchw(rgb.data(), w, h, mmt::Window{0, 0, w, h}, 64, det_in.data());
    for (float v : det_in) {
        assert(v >= 0.0f && v <= 1.0f);
    }

    std::vector<float> rec_chw;
    int rec_w = 0;
    mmt::prepare_rec_line_chw(dark_rgb.data(), w, 60, light_lines[0], 0, 48, 3200, rec_chw, rec_w);
    assert(rec_w > 0 && rec_chw.size() == static_cast<size_t>(3 * 48 * rec_w));
    for (float v : rec_chw) {
        assert(v >= -1.01f && v <= 1.01f);
    }
    std::cout << "[PASS] test_cv_lines_and_tensor_prep (rec_w=" << rec_w << ")\n";
}

void test_momoscript_assembly() {
    std::vector<mmt::RecognizedRegion> regs = {
        {"e1", 0, 0.99f, {16, 40, 84, 108}, "", ""},                   // avatar left
        {"e2", 1, 0.98f, {96, 38, 170, 62}, "", "阿洛娜"},              // speaker_name
        {"e3", 2, 0.99f, {96, 68, 340, 118}, "", "老师好！\n今天辛苦了"}, // bubble_left #1
        {"e4", 10, 0.95f, {348, 96, 410, 118}, "", "12:00"},            // outward message_note
        {"e5", 2, 0.99f, {96, 128, 300, 170}, "", "请查收日程"},         // bubble_left #2 (same speaker)
        {"e6", 3, 0.99f, {240, 190, 480, 232}, "", "好的收到"},          // bubble_right implicit
        {"e7", 4, 0.97f, {40, 250, 460, 290}, "", "十分钟后"},           // narration
        {"e8", 6, 0.99f, {40, 310, 460, 430}, "", ""},                  // reply panel
        {"e9", 7, 0.98f, {60, 340, 440, 376}, "e8", "出发吧"},           // reply_item 1
        {"e10", 7, 0.98f, {60, 384, 440, 420}, "e8", "再等一下"},        // reply_item 2
    };

    auto res = mmt::assemble_momoscript(regs, 500, 500);
    const std::string expected =
        "> 阿洛娜: 老师好！今天辛苦了 | 12:00\n"
        "> 请查收日程\n"
        "< 好的收到\n"
        "- 十分钟后\n"
        "@reply\n"
        "- 出发吧\n"
        "- 再等一下\n"
        "@end\n";
    if (res.momoscript_text != expected) {
        std::cerr << "ACTUAL:\n" << res.momoscript_text << "\nEXPECTED:\n" << expected << "\n";
        assert(false);
    }
    assert(res.messages.size() == 5);
    assert(res.messages[0].speaker == "阿洛娜" && res.messages[0].is_first_turn && res.messages[0].note == "12:00");
    assert(res.messages[1].speaker == "阿洛娜" && !res.messages[1].is_first_turn);
    assert(res.messages[2].is_right && res.messages[2].speaker.empty());
    assert(res.messages[4].kind == "reply" && res.messages[4].content == "出发吧\n再等一下");
    std::cout << "[PASS] test_momoscript_assembly\n";
}

void test_visual_match_ccip_and_stage2() {
    // 1. Multi-avatar box split
    auto single = mmt::split_avatar_box(mmt::BoxI{10, 20, 74, 84});
    assert(single.size() == 1);
    auto pair = mmt::split_avatar_box(mmt::BoxI{10, 20, 138, 84});
    assert(pair.size() == 2);

    // 2. CCIP preprocessing + circle mask
    const int w = 96, h = 96;
    std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3, 40);
    for (int y = 20; y < 76; ++y) {
        for (int x = 20; x < 76; ++x) {
            uint8_t* p = rgb.data() + (static_cast<size_t>(y) * w + x) * 3;
            p[0] = static_cast<uint8_t>(100 + (x % 60));
            p[1] = static_cast<uint8_t>(120 + (y % 50));
            p[2] = 200;
        }
    }
    std::vector<float> nchw(3 * 64 * 64, 0.0f);
    float out_std = 0.0f, out_aspect = 0.0f;
    mmt::prepare_ccip_input_nchw(rgb.data(), w, h, mmt::BoxI{0, 0, w, h}, true, false, 64, nchw.data(), &out_std, &out_aspect);
    assert(out_std > 10.0f);
    assert(std::abs(out_aspect - 1.0f) < 1e-3f);
    // Corner pixel (0,0) was masked to white (255 -> 1.0 -> (1.0 - 0.485)/0.229 ~= 2.2489)
    assert(nchw[0] > 2.0f);

    // 2b. Unified PE-Core-S preprocessing (norm_half=true: white 1.0 -> (1.0 - 0.5)/0.5 == 1.0)
    std::vector<float> nchw_pe(3 * 64 * 64, 0.0f);
    mmt::prepare_ccip_input_nchw(rgb.data(), w, h, mmt::BoxI{0, 0, w, h}, true, false, 64, nchw_pe.data(), nullptr, nullptr, true);
    assert(std::abs(nchw_pe[0] - 1.0f) < 1e-4f);

    // 3. Stage-2 skin template expression discrimination (3 expressions, expression #2 matches query!)
    const int tpl_sz = 32;
    const int count = 3;
    std::vector<uint8_t> stack(static_cast<size_t>(count) * tpl_sz * tpl_sz * 3, 220);
    for (int t = 0; t < count; ++t) {
        uint8_t* dst = stack.data() + static_cast<size_t>(t) * tpl_sz * tpl_sz * 3;
        for (int y = 4; y < 28; ++y) {
            for (int x = 4; x < 28; ++x) {
                size_t off = (static_cast<size_t>(y) * tpl_sz + x) * 3;
                // Shared face contour
                dst[off + 0] = static_cast<uint8_t>(140 + (x * 2));
                dst[off + 1] = static_cast<uint8_t>(100 + (y * 2));
                dst[off + 2] = 160;
            }
        }
        // Expression-specific eye/mouth patch at y=[14..20], x=[10..22]
        for (int y = 14; y < 20; ++y) {
            for (int x = 10; x < 22; ++x) {
                size_t off = (static_cast<size_t>(y) * tpl_sz + x) * 3;
                if (t == 0) { dst[off] = 20; dst[off + 1] = 20; dst[off + 2] = 20; }
                else if (t == 1) { dst[off] = 240; dst[off + 1] = 40; dst[off + 2] = 40; }
                else { dst[off] = 30; dst[off + 1] = 230; dst[off + 2] = 240; }
            }
        }
    }
    std::vector<float> var_w(static_cast<size_t>(tpl_sz) * tpl_sz, 0.1f);
    for (int y = 14; y < 20; ++y) {
        for (int x = 10; x < 22; ++x) {
            var_w[static_cast<size_t>(y) * tpl_sz + x] = 1.0f;
        }
    }
    const uint8_t* query = stack.data() + static_cast<size_t>(2) * tpl_sz * tpl_sz * 3;
    auto st2 = mmt::score_stage2_skin_templates(query, tpl_sz, tpl_sz, mmt::BoxI{0, 0, tpl_sz, tpl_sz}, stack.data(), count, tpl_sz, var_w.data());
    assert(st2.best_index == 2);
    assert(st2.diff_zncc > 0.95f);
    assert(st2.struct_zncc > 0.85f);

    // 3b. Asymmetric white padding around query (e.g. wide box with 20px extra right white margin)
    // must be trimmed back to symmetric framing and still align & match #2!
    const int bw = tpl_sz + 24, bh = tpl_sz;
    std::vector<uint8_t> bordered(static_cast<size_t>(bw) * bh * 3, 252);
    for (int y = 0; y < tpl_sz; ++y) {
        std::memcpy(bordered.data() + (static_cast<size_t>(y) * bw) * 3,
                    query + static_cast<size_t>(y) * tpl_sz * 3,
                    static_cast<size_t>(tpl_sz) * 3);
    }
    auto st2_bordered = mmt::score_stage2_skin_templates(bordered.data(), bw, bh, mmt::BoxI{0, 0, bw, bh}, stack.data(), count, tpl_sz, var_w.data());
    assert(st2_bordered.best_index == 2);
    assert(st2_bordered.diff_zncc > 0.90f);
    assert(st2_bordered.struct_zncc > 0.80f);

    float ezncc = mmt::score_emoji_structure_zncc(query, tpl_sz, tpl_sz, mmt::BoxI{0, 0, tpl_sz, tpl_sz}, query, tpl_sz);
    assert(ezncc > 0.99f);
    std::cout << "[PASS] test_visual_match_ccip_and_stage2 (best=" << st2.best_index << ", diff=" << st2.diff_zncc << ")\n";
}

void test_speaker_across_insert_image() {
    // Regression: [Narration] -> [Avatar + SpeakerName("柚子")] -> [Tall insert_image] -> [Left Bubble]
    std::vector<mmt::RecognizedRegion> regs = {
        {"r0", 4, 0.99f, {40, 10, 440, 44}, "", "我俩一脸怨念地看向了美咲同学:"},
        {"r1", 0, 0.99f, {20, 60, 100, 140}, "", ""},
        {"r2", 1, 0.99f, {115, 62, 175, 86}, "", "柚子"},
        {"r3", 5, 0.99f, {115, 92, 390, 430}, "", ""},
        {"r4", 2, 0.99f, {115, 442, 400, 488}, "", "不是，为什么？跑反了你不说？"},
        {"r5", 0, 0.99f, {20, 500, 100, 580}, "", ""},
        {"r6", 1, 0.99f, {115, 502, 175, 526}, "", "美咲"},
        {"r7", 2, 0.99f, {115, 534, 440, 590}, "", "...第一，当时是老师和日和去了一趟"},
    };
    auto res = mmt::assemble_momoscript(regs, 480, 640);
    assert(res.momoscript_text.find("> 柚子: [image]\n> 不是，为什么？跑反了你不说？\n") != std::string::npos);
    assert(res.momoscript_text.find("[unknown speaker]") == std::string::npos);
    assert(res.momoscript_text.find("> 美咲: ...第一，当时是老师和日和去了一趟") != std::string::npos);
    std::cout << "[PASS] test_speaker_across_insert_image\n";
}

void test_cv_lines_bubble_tail_and_rounded_corners() {
    // Regression: dark blue bubble on white canvas with left speech tail and rounded corners.
    // Exterior white pixels around the tail and corners must NOT leak into ink_box.
    const int W = 320;
    const int H = 120;
    std::vector<uint8_t> rgb(static_cast<size_t>(W) * H * 3, 255);  // white page canvas

    auto set_px = [&](int x, int y, uint8_t r, uint8_t g, uint8_t b) {
        if (x < 0 || x >= W || y < 0 || y >= H) return;
        uint8_t* p = rgb.data() + (static_cast<size_t>(y) * W + x) * 3;
        p[0] = r; p[1] = g; p[2] = b;
    };

    // Rounded dark bubble body: x=[40..280), y=[24..74), corner radius 8px
    const int bx0 = 40, by0 = 24, bx1 = 280, by1 = 74, rad = 8;
    for (int y = by0; y < by1; ++y) {
        for (int x = bx0; x < bx1; ++x) {
            int dx = 0, dy = 0;
            if (x < bx0 + rad) dx = (bx0 + rad) - x;
            else if (x >= bx1 - rad) dx = x - (bx1 - rad - 1);
            if (y < by0 + rad) dy = (by0 + rad) - y;
            else if (y >= by1 - rad) dy = y - (by1 - rad - 1);
            if (dx * dx + dy * dy <= rad * rad) {
                set_px(x, y, 76, 91, 112);  // MomoTalk dark slate-blue bubble
            }
        }
    }
    // Left speech bubble triangular tail: x=[30..41), y=[36..48) (height 12 < 0.5*50)
    for (int y = 36; y < 48; ++y) {
        int span = std::min(y - 36, 47 - y) + 1;
        for (int x = bx0 - span * 2; x < bx0 + 1; ++x) {
            set_px(x, y, 76, 91, 112);
        }
    }
    // White text glyphs inside bubble: x=[56..256), y=[40..58)
    for (int k = 0; k < 10; ++k) {
        for (int y = 40; y < 58; ++y) {
            for (int x = 56 + k * 20; x < 68 + k * 20; ++x) {
                set_px(x, y, 250, 250, 250);
            }
        }
    }

    // Detector box tightly wrapping bubble + tail: [30, 24, 280, 74]
    auto lines = mmt::segment_physical_lines(rgb.data(), W, H, mmt::BoxI{30, 24, 280, 74}, false);
    assert(lines.size() == 1);
    assert(lines[0].is_light_fg);
    // Must cleanly exclude both the left white area around tail (x=30..40) and right rounded corners (x=272..280)
    assert(lines[0].ink_box.x0 == 56);
    assert(lines[0].ink_box.x1 == 248);
    assert(lines[0].ink_box.y0 == 40);
    assert(lines[0].ink_box.y1 == 58);

    // Also test a follow-up bubble WITHOUT tail and tight detector box [40, 24, 280, 74]
    auto lines_notail = mmt::segment_physical_lines(rgb.data(), W, H, mmt::BoxI{40, 24, 280, 74}, false);
    assert(lines_notail.size() == 1);
    assert(lines_notail[0].ink_box.x0 == 56);
    assert(lines_notail[0].ink_box.x1 == 248);
    std::cout << "[PASS] test_cv_lines_bubble_tail_and_rounded_corners\n";
}

}  // namespace

int main() {
    test_adaptive_blank_windows();
    test_postprocess_tr_fusion_and_reply();
    test_avifs_pack_reader();
    test_cv_lines_and_tensor_prep();
    test_momoscript_assembly();
    test_visual_match_ccip_and_stage2();
    test_speaker_across_insert_image();
    test_cv_lines_bubble_tail_and_rounded_corners();
    return 0;
}
