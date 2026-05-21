#include "desrt/shard.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

namespace desrt {

namespace {
constexpr size_t kNameBufLen = 32;

std::string shard_name(uint32_t shard) {
    char buf[kNameBufLen];
    std::snprintf(buf, sizeof(buf), "shard_%06u.bin", shard);
    return std::string(buf);
}
} // namespace

std::string raw_shard_path(const std::string& root, uint32_t shard) {
    return (fs::path(root) / "raw" / shard_name(shard)).string();
}

std::string sorted_shard_path(const std::string& root, uint32_t shard) {
    return (fs::path(root) / "sorted" / shard_name(shard)).string();
}

void ensure_shard_dirs(const std::string& root) {
    fs::create_directories(fs::path(root) / "raw");
    fs::create_directories(fs::path(root) / "sorted");
}

ShardWriter::ShardWriter(const std::string& root, uint32_t num_shards,
                         size_t per_shard_buffer_bytes)
    : num_shards_(num_shards),
      buf_capacity_(std::max<size_t>(64, per_shard_buffer_bytes / sizeof(Record))),
      root_(root),
      files_(num_shards, nullptr),
      buffers_(num_shards)
{
    ensure_shard_dirs(root);
    for (uint32_t s = 0; s < num_shards_; s++) {
        buffers_[s].reserve(buf_capacity_);
        const std::string path = raw_shard_path(root, s);
        FILE* f = std::fopen(path.c_str(), "ab");
        if (!f) {
            throw std::runtime_error("Failed to open shard for append: " + path
                                   + " (" + std::strerror(errno) + ")");
        }
        files_[s] = f;
    }
}

ShardWriter::~ShardWriter() {
    try { flush_all(); } catch (...) { /* destructor must not throw */ }
    for (FILE* f : files_) {
        if (f) std::fclose(f);
    }
}

void ShardWriter::append(const Record& r) {
    uint32_t s = shard_for_endpoint(r.endpoint, num_shards_);
    auto& buf = buffers_[s];
    buf.push_back(r);
    if (buf.size() >= buf_capacity_) flush_shard(s);
}

void ShardWriter::append_batch(const Record* records, size_t count) {
    for (size_t i = 0; i < count; i++) {
        append(records[i]);
    }
}

void ShardWriter::flush_shard(uint32_t s) {
    auto& buf = buffers_[s];
    if (buf.empty()) return;
    size_t written = std::fwrite(buf.data(), sizeof(Record), buf.size(), files_[s]);
    if (written != buf.size()) {
        throw std::runtime_error("Short write on shard " + std::to_string(s));
    }
    buf.clear();
}

void ShardWriter::flush_all() {
    for (uint32_t s = 0; s < num_shards_; s++) {
        flush_shard(s);
        if (files_[s]) std::fflush(files_[s]);
    }
}

size_t sort_shard(const std::string& root, uint32_t shard, bool delete_raw) {
    const std::string raw = raw_shard_path(root, shard);
    const std::string sorted = sorted_shard_path(root, shard);

    if (!fs::exists(raw)) {
        return 0; // empty / never written
    }
    const uintmax_t bytes = fs::file_size(raw);
    if (bytes % sizeof(Record) != 0) {
        throw std::runtime_error("Shard " + raw + " has non-record size " + std::to_string(bytes));
    }
    const size_t count = bytes / sizeof(Record);

    std::vector<Record> records(count);
    if (count > 0) {
        FILE* f = std::fopen(raw.c_str(), "rb");
        if (!f) throw std::runtime_error("Cannot read " + raw);
        size_t got = std::fread(records.data(), sizeof(Record), count, f);
        std::fclose(f);
        if (got != count) {
            throw std::runtime_error("Short read on " + raw);
        }

        std::sort(records.begin(), records.end(),
                  [](const Record& a, const Record& b) {
                      return a.endpoint < b.endpoint;
                  });
    }

    {
        FILE* f = std::fopen(sorted.c_str(), "wb");
        if (!f) throw std::runtime_error("Cannot open " + sorted + " for writing");
        if (count > 0) {
            size_t wrote = std::fwrite(records.data(), sizeof(Record), count, f);
            if (wrote != count) {
                std::fclose(f);
                throw std::runtime_error("Short write on " + sorted);
            }
        }
        std::fflush(f);
        std::fclose(f);
    }

    if (delete_raw) {
        std::error_code ec;
        fs::remove(raw, ec);
    }
    return count;
}

std::vector<Record> read_shard(const std::string& shard_path) {
    if (!fs::exists(shard_path)) return {};
    const uintmax_t bytes = fs::file_size(shard_path);
    if (bytes % sizeof(Record) != 0) {
        throw std::runtime_error("Shard " + shard_path + " has non-record size");
    }
    const size_t count = bytes / sizeof(Record);
    std::vector<Record> out(count);
    if (count == 0) return out;
    FILE* f = std::fopen(shard_path.c_str(), "rb");
    if (!f) throw std::runtime_error("Cannot read " + shard_path);
    size_t got = std::fread(out.data(), sizeof(Record), count, f);
    std::fclose(f);
    if (got != count) throw std::runtime_error("Short read on " + shard_path);
    return out;
}

bool find_endpoint(const Record* sorted, size_t n, uint64_t endpoint, Record* out) {
    // Standard lower_bound on the endpoint field.
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (sorted[mid].endpoint < endpoint) lo = mid + 1;
        else                                  hi = mid;
    }
    if (lo < n && sorted[lo].endpoint == endpoint) {
        if (out) *out = sorted[lo];
        return true;
    }
    return false;
}

} // namespace desrt
