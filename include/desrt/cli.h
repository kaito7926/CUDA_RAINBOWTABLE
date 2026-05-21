// Minimal flag parsing for subcommands. Supports
//   --name value   --name=value   -n value   -n=value
// All values are strings; callers convert via strtoull / atoi.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <stdexcept>
#include <vector>

namespace desrt::cli {

struct Args {
    // argv[0] is the subcommand name; flags start at argv[1].
    int    argc;
    char** argv;

    // Returns the value of --name (or -short), or `fallback` if missing.
    // Recognises both "--name value" and "--name=value".
    const char* opt(const char* name, const char* short_name = nullptr,
                    const char* fallback = nullptr) const
    {
        const size_t lname = std::strlen(name);
        for (int i = 1; i < argc; i++) {
            const char* a = argv[i];
            // --name= form
            if (std::strncmp(a, name, lname) == 0 && a[lname] == '=') {
                return a + lname + 1;
            }
            // --name <value> form
            if (std::strcmp(a, name) == 0 && i + 1 < argc) {
                return argv[i + 1];
            }
            if (short_name && a[0] == '-' && a[1] && a[1] == short_name[1]) {
                if (a[2] == '=') return a + 3;
                if (a[2] == 0 && i + 1 < argc) return argv[i + 1];
            }
        }
        return fallback;
    }

    // True if the flag is present with no value (e.g. --help).
    bool has(const char* name, const char* short_name = nullptr) const {
        for (int i = 1; i < argc; i++) {
            if (std::strcmp(argv[i], name) == 0) return true;
            if (short_name && std::strcmp(argv[i], short_name) == 0) return true;
        }
        return false;
    }

    uint64_t opt_u64(const char* name, uint64_t fallback,
                     const char* short_name = nullptr) const
    {
        const char* v = opt(name, short_name, nullptr);
        if (!v) return fallback;
        return std::strtoull(v, nullptr, 0);
    }

    uint32_t opt_u32(const char* name, uint32_t fallback,
                     const char* short_name = nullptr) const
    {
        return static_cast<uint32_t>(opt_u64(name, fallback, short_name));
    }

    int opt_int(const char* name, int fallback,
                const char* short_name = nullptr) const
    {
        const char* v = opt(name, short_name, nullptr);
        if (!v) return fallback;
        return std::atoi(v);
    }

    double opt_double(const char* name, double fallback,
                      const char* short_name = nullptr) const
    {
        const char* v = opt(name, short_name, nullptr);
        if (!v) return fallback;
        return std::strtod(v, nullptr);
    }

    std::string opt_str(const char* name, const std::string& fallback,
                        const char* short_name = nullptr) const
    {
        const char* v = opt(name, short_name, nullptr);
        return v ? std::string(v) : fallback;
    }
};

inline Args wrap(int argc, char** argv) { return Args{argc, argv}; }

} // namespace desrt::cli
