// Shard I/O. The raw table is partitioned into `num_shards` files by
// shard_id = endpoint % num_shards. During build we stream Records into
// per-shard buffers and flush in fixed-size chunks. After build, each raw
// shard is read, sorted by endpoint, and written back as a sorted shard.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <memory>

#include "desrt/common.h"

namespace desrt {

// Pick a shard for an endpoint.
inline uint32_t shard_for_endpoint(uint64_t endpoint, uint32_t num_shards) {
    return static_cast<uint32_t>(endpoint % num_shards);
}

// Path helpers. Directory layout under <root>/:
//   raw/shard_NNNNNN.bin
//   sorted/shard_NNNNNN.bin
std::string raw_shard_path(const std::string& root, uint32_t shard);
std::string sorted_shard_path(const std::string& root, uint32_t shard);

// Make sure root/raw and root/sorted exist. Idempotent.
void ensure_shard_dirs(const std::string& root);

// Per-shard buffered writer. Holds open FILE* and a fixed-size buffer per
// shard. flush_all() drains every buffer; the destructor flushes and closes.
class ShardWriter {
public:
    ShardWriter(const std::string& root, uint32_t num_shards,
                size_t per_shard_buffer_bytes);
    ~ShardWriter();

    ShardWriter(const ShardWriter&) = delete;
    ShardWriter& operator=(const ShardWriter&) = delete;

    // Append a single record to its shard.
    void append(const Record& r);

    // Append a contiguous batch. Faster than calling append() per record
    // because it can amortise the per-record branch.
    void append_batch(const Record* records, size_t count);

    // Flush all buffers and fsync each file.
    void flush_all();

    uint32_t num_shards() const { return num_shards_; }

private:
    void flush_shard(uint32_t s);

    uint32_t num_shards_;
    size_t   buf_capacity_; // in records
    std::string root_;
    std::vector<FILE*> files_;
    std::vector<std::vector<Record>> buffers_;
};

// Sort one raw shard file into the corresponding sorted shard file.
// Returns the record count written. If `delete_raw` is true, the raw file is
// removed once the sorted file is on disk.
size_t sort_shard(const std::string& root, uint32_t shard, bool delete_raw);

// Read one whole shard into memory (used by stats and crack).
std::vector<Record> read_shard(const std::string& shard_path);

// Binary search a sorted shard for an endpoint. If found, fills `out` and
// returns true. The shard must be sorted ascending by endpoint.
bool find_endpoint(const Record* sorted, size_t n, uint64_t endpoint, Record* out);

} // namespace desrt
