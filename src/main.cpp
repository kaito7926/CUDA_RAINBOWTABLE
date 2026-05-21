// desrt — DES rainbow table builder/cracker for an 8-char [a-z0-9] keyspace.
//
// Command dispatch:
//   desrt plan
//   desrt bench
//   desrt build
//   desrt sort
//   desrt make-target
//   desrt crack
//   desrt stats
//
// Use `desrt <command> --help` for per-command flags.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Subcommand entry points (defined in their respective .cpp/.cu files).
int cmd_plan(int argc, char** argv);
int cmd_bench(int argc, char** argv);
int cmd_build(int argc, char** argv);
int cmd_sort(int argc, char** argv);
int cmd_make_target(int argc, char** argv);
int cmd_crack(int argc, char** argv);
int cmd_stats(int argc, char** argv);

static void print_usage() {
    std::fprintf(stderr,
        "desrt — DES rainbow table for fixed plaintext 0x1122334455667788\n"
        "\n"
        "Usage:\n"
        "  desrt plan          Show keyspace, table size, coverage, build-time estimate\n"
        "  desrt bench         Validate DES test vector and time host/GPU throughput\n"
        "  desrt build         Build the rainbow table on GPU, streaming raw shards to disk\n"
        "  desrt sort          Sort each raw shard by endpoint and write sorted shards\n"
        "  desrt make-target   Generate random alnum-key ciphertext targets\n"
        "  desrt crack         Crack one or more ciphertexts against the sorted table\n"
        "  desrt stats         Report record counts and unique-endpoint stats per shard\n"
        "\n"
        "  desrt <command> --help    Show flags for one command\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }
    const std::string cmd = argv[1];

    // Shift past the program name + command name; subcommands see their own
    // argv[0] as the command word, which matches the convention used by
    // getopt-style parsers and lets each subcommand print its own --help.
    int sub_argc = argc - 1;
    char** sub_argv = argv + 1;

    if (cmd == "plan")             return cmd_plan(sub_argc, sub_argv);
    if (cmd == "bench")            return cmd_bench(sub_argc, sub_argv);
    if (cmd == "build")            return cmd_build(sub_argc, sub_argv);
    if (cmd == "sort")             return cmd_sort(sub_argc, sub_argv);
    if (cmd == "make-target")      return cmd_make_target(sub_argc, sub_argv);
    if (cmd == "crack")            return cmd_crack(sub_argc, sub_argv);
    if (cmd == "stats")            return cmd_stats(sub_argc, sub_argv);

    if (cmd == "-h" || cmd == "--help" || cmd == "help") {
        print_usage();
        return 0;
    }

    std::fprintf(stderr, "desrt: unknown command '%s'\n\n", argv[1]);
    print_usage();
    return 1;
}
