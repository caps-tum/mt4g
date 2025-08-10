#include <cxxopts.hpp>
#include <cstdint>
#include <vector>
#include <string>
#include <iostream>
#include <hip/hip_runtime.h>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <set>
#include <functional>
#include <nlohmann/json.hpp>
#include <sstream>

#include "typedef/cliOptions.hpp"

namespace util {
    CLIOptions parseCommandLine(int argc, char* argv[]) {
        CLIOptions opts{};
        // Set defaults
        opts.deviceId = 0;
        opts.graphs = false;
        opts.rawData = false;
        opts.fullReport = false;
        opts.writeJson = false;
        opts.randomize = false;
        opts.runSilently = false;

        opts.runL3 = false;
        opts.runL2 = false;
        opts.runL1 = false;
        opts.runScalar = false;
        opts.runConstant = false;
        opts.runReadOnly = false;
        opts.runTexture = false;
        opts.runSharedMemory = false;
        opts.runMainMemory = false;
        opts.runDepartureDelay = false;
        opts.runResourceSharing = false;

        // Default cache preference (used if --cache=auto)
        opts.cachePreference = hipFuncCachePreferL1;

        cxxopts::Options parser("mt4g", "Memory Topology for GPUs");

        parser.add_options()
            // ------- Core options -------
            ("d,device-id", "Int: GPU ID to use",
                cxxopts::value<int>()->default_value("0"))
            ("g,graphs", "Generate graphs for each benchmark",
                cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("o,raw", "Write raw timing data",
                cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("p,report", "Create Markdown report in output directory",
                cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("j,json", "Write final JSON to <GPU_NAME>.json in current directory",
                cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("r,random", "Randomize P-Chase arrays",
                cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("q,quiet", "Suppress intermediate console output",
                cxxopts::value<bool>()->default_value("false")->implicit_value("true"))

            // ------- Benchmark group toggles -------
            ("l2", "Run L2 benchmarks",
                cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("l1", "Run L1 benchmarks",
                cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("shared", "Run SharedMemory benchmarks",
                cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("memory", "Run MainMemory benchmarks",
                cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("departuredelay", "Run DepartureDelay benchmarks",
                cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("scalar", "Run amd/Scalar benchmarks",
                cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("l3", "Run amd/L3 benchmarks",
                cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("constant", "Run nvidia/Constant benchmarks",
                cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("readonly", "Run nvidia/ReadOnly benchmarks",
                cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("texture", "Run nvidia/Texture benchmarks",
                cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("resourceshare", "Run Resource Sharing benchmarks",
                cxxopts::value<bool>()->default_value("false")->implicit_value("true"))

            // ------- Single cache preference option -------
            // Use one string option instead of multiple booleans.
            ("cache", "Cache preference: l1 | shared | equal | auto (default: auto)",
                cxxopts::value<std::string>()->default_value("auto"))

            // ------- Help -------
            ("h,help", "Print help");

        // Parse with error handling
        cxxopts::ParseResult result;
        try {
            result = parser.parse(argc, argv);
        } catch (const cxxopts::exceptions::parsing& ex) {
            std::cerr << "Parsing error: " << ex.what() << std::endl;
            std::exit(EXIT_FAILURE);
        }

        // Show help and exit?
        if (result.count("help")) {
            std::cout << parser.help({""}) << std::endl;
            std::exit(EXIT_SUCCESS);
        }

        // ------- Core options -------
        opts.deviceId   = result["device-id"].as<int>();
        opts.graphs     = result["graphs"].as<bool>();
        opts.rawData    = result["raw"].as<bool>();
        opts.fullReport = result["report"].as<bool>();
        opts.writeJson  = result["json"].as<bool>();
        opts.randomize  = result["random"].as<bool>();
        opts.runSilently= result["quiet"].as<bool>();

        // ------- Cache preference parsing -------
        // Convert to lowercase for robust matching.
        auto to_lower = [](std::string s) {
            std::transform(s.begin(), s.end(), s.begin(),
                        [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
            return s;
        };

        const std::unordered_map<std::string, hipFuncCache_t> cache_map{
            {"l1",     hipFuncCachePreferL1},
            {"shared", hipFuncCachePreferShared},
            {"equal",  hipFuncCachePreferEqual},
            // "auto" selects the default defined in opts.cachePreference (set above)
            {"auto",   opts.cachePreference}
        };

        const auto pref_str = to_lower(result["cache"].as<std::string>());
        if (auto it = cache_map.find(pref_str); it != cache_map.end()) {
            opts.cachePreference = it->second;
        } else {
            std::cerr << "Invalid --cache value: '" << pref_str
                    << "'. Allowed: l1 | shared | equal | auto\n";
            std::exit(EXIT_FAILURE);
        }

        // ------- Benchmark group toggles -------
        int groupCount = 0;
        auto check = [&](const char* key, bool& flag) {
            // Read flag and track whether any group was requested explicitly.
            flag = result[key].as<bool>();
            if (flag) ++groupCount;
        };

        check("l3",             opts.runL3);
        check("l2",             opts.runL2);
        check("l1",             opts.runL1);
        check("scalar",         opts.runScalar);
        check("constant",       opts.runConstant);
        check("readonly",       opts.runReadOnly);
        check("texture",        opts.runTexture);
        check("shared",         opts.runSharedMemory);
        check("memory",         opts.runMainMemory);
        check("departuredelay", opts.runDepartureDelay);
        check("resourceshare",  opts.runResourceSharing);

        // ------- Vendor-specific sanity checks -------
        #ifdef __HIP_PLATFORM_AMD__
        if (opts.runConstant || opts.runReadOnly || opts.runTexture) {
            std::cerr << "NVIDIA-specific benchmarks cannot be run on AMD GPUs." << std::endl;
            std::exit(EXIT_FAILURE);
        }
        #endif

        #ifdef __HIP_PLATFORM_NVIDIA__
        if (opts.runL3 || opts.runScalar) {
            std::cerr << "AMD-specific benchmarks cannot be run on NVIDIA GPUs." << std::endl;
            std::exit(EXIT_FAILURE);
        }
        #endif

        // If no groups were selected, enable all by default
        if (groupCount == 0) {
            opts.runL3 = opts.runL2 = opts.runL1 =
            opts.runScalar = opts.runConstant = opts.runReadOnly =
            opts.runTexture = opts.runSharedMemory = opts.runMainMemory =
            opts.runDepartureDelay = opts.runResourceSharing = true;
        }

        // Ensure vendor-specific toggles are off even in "run all" mode
        #ifdef __HIP_PLATFORM_AMD__
        opts.runConstant = opts.runReadOnly = opts.runTexture = false;
        #endif
        #ifdef __HIP_PLATFORM_NVIDIA__
        opts.runL3 = opts.runScalar = false;
        #endif

        return opts;
    }

    void writeMarkdownReport(const std::filesystem::path& outDir, const std::string& deviceName, const nlohmann::json& result) {
        std::ofstream reportFile(outDir / "README.md");
        if (!reportFile) {
            std::cerr << "Could not write report file in '" << outDir.string() << "'" << std::endl;
            return;
        }

        reportFile << "# " << deviceName << " Benchmark Report\n\n";

        // helper to stringify JSON values
        std::function<std::string(const nlohmann::json&)> stringify;
        stringify = [&stringify](const nlohmann::json& j) -> std::string {
            if (j.is_object()) {
                if (j.contains("mean") && j.contains("unit")) {
                    std::ostringstream os;
                    os << j["mean"].get<double>();
                    os << ' ' << j["unit"].get<std::string>();
                    return os.str();
                }
                if (j.contains("size") && j.contains("unit")) {
                    std::ostringstream os;
                    if (j["size"].is_number_float())
                        os << j["size"].get<double>();
                    else
                        os << j["size"].get<long long>();
                    os << ' ' << j["unit"].get<std::string>();
                    return os.str();
                }
                if (j.contains("value") && j.contains("unit")) {
                    std::ostringstream os;
                    if (j["value"].is_number_float())
                        os << j["value"].get<double>();
                    else if (j["value"].is_number_integer())
                        os << j["value"].get<long long>();
                    else
                        os << j["value"].dump();
                    os << ' ' << j["unit"].get<std::string>();
                    return os.str();
                }
                if (j.contains("major") && j.contains("minor") && j.size() == 2) {
                    return std::to_string(j["major"].get<int>()) + "." +
                        std::to_string(j["minor"].get<int>());
                }
                std::string s;
                bool first = true;
                for (auto& [k, v] : j.items()) {
                    if (!first) s += ", ";
                    s += k + ": " + stringify(v);
                    first = false;
                }
                return s;
            }
            if (j.is_array()) {
                std::string s;
                bool first = true;
                for (auto& v : j) {
                    if (!first) s += ", ";
                    s += stringify(v);
                    first = false;
                }
                return s;
            }
            if (j.is_boolean()) return j.get<bool>() ? "true" : "false";
            if (j.is_string()) return j.get<std::string>();
            return j.dump();
        };

        auto writeTable = [&](const nlohmann::json& obj, const std::set<std::string>& skipKeys = {}) {
            reportFile << "| Key | Value |\n| --- | ----- |\n";
            for (auto& [key, val] : obj.items()) {
                if (skipKeys.contains(key)) continue;
                reportFile << "| " << key << " | " << stringify(val) << " |\n";
            }
            reportFile << "\n";
        };

        auto writeSharedBetween = [&](const nlohmann::json& groups) {
            if (!groups.is_array() || groups.empty()) return;
            reportFile << '|';
            for (size_t i = 0; i < groups.size(); ++i) {
                reportFile << " Group " << (i + 1) << " |";
            }
            reportFile << "\n|";
            for (size_t i = 0; i < groups.size(); ++i) {
                reportFile << " --- |";
            }
            reportFile << "\n|";
            for (auto& group : groups) {
                std::string cell;
                bool first = true;
                for (auto& cu : group) {
                    if (!first) cell += ", ";
                    cell += std::to_string(cu.get<uint32_t>());
                    first = false;
                }
                reportFile << ' ' << cell << " |";
            }
            reportFile << "\n\n";
        };

        if (result.contains("general")) {
            reportFile << "## General\n\n";
            writeTable(result["general"]);
        }
        if (result.contains("compute")) {
            reportFile << "## Compute\n\n";
            writeTable(result["compute"]);
        }
        if (result.contains("memory")) {
            reportFile << "## Memory\n\n";
            for (auto& [section, obj] : result["memory"].items()) {
                if (section == "constant") {
                    reportFile << "### " << section << "\n\n";
                    writeTable(obj, {"l1", "l1.5"});
                    if (obj.contains("l1")) {
                        reportFile << "#### constant l1\n\n";
                        writeTable(obj["l1"]);
                    }
                    if (obj.contains("l1.5")) {
                        reportFile << "#### constant l1.5\n\n";
                        writeTable(obj["l1.5"]);
                    }
                } else {
                    reportFile << "### " << section << "\n\n";
                    if (section == "scalarL1" && obj.contains("sharedBetween")) {
                        writeTable(obj, {"sharedBetween"});
                        writeSharedBetween(obj["sharedBetween"]);
                    } else {
                        writeTable(obj);
                    }
                }
            }
        }

        // helper to encode spaces in URLs
        auto urlEncode = [](const std::string& str) -> std::string {
            std::string encoded;
            for (char c : str) {
                if (c == ' ') encoded += "%20";
                else encoded += c;
            }
            return encoded;
        };

        std::vector<std::filesystem::path> images;
        for (auto& entry : std::filesystem::directory_iterator(outDir)) {
            if (entry.path().extension() == ".png") {
                images.push_back(entry.path().filename());
            }
        }

        if (!images.empty()) {
            reportFile << "## Graphs\n\n";
            std::sort(images.begin(), images.end());
            for (auto& img : images) {
                std::string imgFilename = urlEncode(img.string());
                reportFile << "![" << img.stem().string() << "](./" << imgFilename << ")\n";

                auto raw = img;
                raw.replace_extension(".txt");
                if (std::filesystem::exists(outDir / raw)) {
                    std::string rawFilename = urlEncode(raw.string());
                    reportFile << "[Raw data](./" << rawFilename << ")\n";
                }
                reportFile << "\n";
            }
        }

        reportFile << "## Raw JSON\n\n";
        reportFile << "```json\n" << result.dump(4) << "\n```\n";
    }
}