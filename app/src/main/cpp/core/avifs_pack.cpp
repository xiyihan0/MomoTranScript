#include "avifs_pack.hpp"

#include <cstring>

namespace mmt {
namespace {

bool extract_int_field(const std::string& json, const std::string& key, int& out) {
    const std::string needle = "\"" + key + "\":";
    const size_t pos = json.find(needle);
    if (pos == std::string::npos) return false;
    size_t i = pos + needle.size();
    while (i < json.size() && (json[i] == ' ' || json[i] == '\t')) ++i;
    char* end = nullptr;
    const long val = std::strtol(json.c_str() + i, &end, 10);
    if (end == json.c_str() + i) return false;
    out = static_cast<int>(val);
    return true;
}

}  // namespace

bool AvifsPackReader::open(const std::string& path, std::string* error_out) {
    file_.close();
    file_.open(path, std::ios::binary);
    if (!file_.is_open()) {
        if (error_out) *error_out = "Unable to open pack file: " + path;
        return false;
    }

    char magic[8] = {};
    if (!file_.read(magic, 8) || std::memcmp(magic, "MMTAVIF1", 8) != 0) {
        if (error_out) *error_out = "Invalid MMTAVIF1 pack magic";
        file_.close();
        return false;
    }

    uint8_t len_bytes[4] = {};
    if (!file_.read(reinterpret_cast<char*>(len_bytes), 4)) {
        if (error_out) *error_out = "Truncated pack header length";
        file_.close();
        return false;
    }
    const uint32_t hdr_len =
        static_cast<uint32_t>(len_bytes[0]) |
        (static_cast<uint32_t>(len_bytes[1]) << 8) |
        (static_cast<uint32_t>(len_bytes[2]) << 16) |
        (static_cast<uint32_t>(len_bytes[3]) << 24);

    if (hdr_len == 0 || hdr_len > 32U * 1024U * 1024U) {
        if (error_out) *error_out = "Invalid pack header length";
        file_.close();
        return false;
    }

    std::string hdr(hdr_len, '\0');
    if (!file_.read(hdr.data(), static_cast<std::streamsize>(hdr_len))) {
        if (error_out) *error_out = "Truncated pack header JSON";
        file_.close();
        return false;
    }

    index_ = AvifsPackIndex{};
    index_.payload_base_offset = 12ULL + hdr_len;
    extract_int_field(hdr, "schema_version", index_.schema_version);
    extract_int_field(hdr, "tpl_size", index_.tpl_size);
    extract_int_field(hdr, "skin_count", index_.skin_count);
    extract_int_field(hdr, "expression_count", index_.expression_count);
    extract_int_field(hdr, "emoji_count", index_.emoji_count);
    index_.raw_header_json = std::move(hdr);
    return true;
}

bool AvifsPackReader::read_payload_slice(uint64_t rel_offset, uint64_t length, std::vector<uint8_t>& out) {
    if (!file_.is_open()) return false;
    out.resize(static_cast<size_t>(length));
    file_.seekg(static_cast<std::streamoff>(index_.payload_base_offset + rel_offset), std::ios::beg);
    return static_cast<bool>(file_.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(length)));
}

}  // namespace mmt
