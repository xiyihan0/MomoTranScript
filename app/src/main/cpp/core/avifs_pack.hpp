#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace mmt {

struct AvifsSkinMeta {
    int index = 0;
    int count = 0;
    std::vector<int> frame_order;
    uint64_t avifs_offset = 0;
    uint64_t avifs_len = 0;
    uint64_t mask_offset = 0;
    uint64_t mask_len = 0;
};

struct AvifsPackIndex {
    int schema_version = 0;
    int tpl_size = 64;
    int skin_count = 0;
    int expression_count = 0;
    int emoji_count = 0;
    uint64_t payload_base_offset = 0;
    std::string raw_header_json;
};

/// O(1) random-access reader for ``MMTAVIF1`` binary template packs.
class AvifsPackReader {
public:
    bool open(const std::string& path, std::string* error_out = nullptr);
    [[nodiscard]] bool is_open() const { return file_.is_open(); }
    [[nodiscard]] const AvifsPackIndex& index() const { return index_; }

    /// Read raw bytes from payload offset (relative to payload_base_offset).
    bool read_payload_slice(uint64_t rel_offset, uint64_t length, std::vector<uint8_t>& out);

private:
    std::ifstream file_;
    AvifsPackIndex index_;
};

}  // namespace mmt
