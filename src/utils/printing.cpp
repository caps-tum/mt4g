#include <cxxopts.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <hip/hip_runtime.h>

#include "typedef/cliOptions.hpp"

namespace util {
    CLIOptions parseCommandLine(int argc, char* argv[]) {
        CLIOptions opts{};
        // Set defaults
        opts.deviceId = 0;
        opts.graphs = false;
        opts.rawData = false;
        opts.randomize = false;
        opts.runSilently = false;
        opts.runL3 = false;
        opts.runL2 = false;
        opts.runL1 = false;
        opts.runScalar = false;
        opts.runConstant = false;
        opts.runReadOnly = false;
        opts.runTexture = false;
        opts.runSharedMemory= false;
        opts.runMainMemory = false;
        opts.runDepartureDelay = false;
        opts.runResourceSharing = false;

        cxxopts::Options parser("mt4amd", "Memory Topology for GPUs"); 

        parser.add_options()
            // Core options
            ("d,device-id", "Int: GPU ID to use", cxxopts::value<int>()->default_value("0"))
            ("g,graphs",    "Generate graphs for each benchmark", cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("o,raw",       "Write raw timing data",        cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("r,random",    "Randomize P-Chase arrays",        cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("q,quiet",     "Suppress intermediate console output", cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            //("o,output",    "Output base directory",        cxxopts::value<bool>()->default_value("false")->implicit_value("true"))

            // Benchmark group toggles
            ("l2",            "Run L2 benchmarks",             cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("l1",            "Run L1 benchmarks",             cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("shared",        "Run SharedMemory benchmarks",   cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("memory",        "Run MainMemory benchmarks",     cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("departuredelay","Run DepartureDelay benchmarks", cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("scalar",        "Run amd/Scalar benchmarks",     cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("l3",            "Run amd/L3 benchmarks",         cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("constant",      "Run nvidia/Constant benchmarks",cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("readonly",      "Run nvidia/ReadOnly benchmarks",cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("texture",       "Run nvidia/Texture benchmarks", cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("resourceshare", "Run Resource Sharing benchmarks",cxxopts::value<bool>()->default_value("false")->implicit_value("true"))

            // Standard help
            ("h,help",        "Print help");

        // Allow exceptions on parse errors 
        cxxopts::ParseResult result;
        try {
            result = parser.parse(argc, argv);
        } catch (const cxxopts::exceptions::parsing &ex) {
            std::cerr << "Parsing error: " << ex.what() << std::endl;
            std::exit(EXIT_FAILURE);
        }

        // Help requested?
        if (result.count("help")) {
            std::cout << parser.help({""}) << std::endl;
            std::exit(EXIT_SUCCESS);
        }

        // Core options
        opts.deviceId  = result["device-id"].as<int>();
        opts.graphs    = result["graphs"].as<bool>();
        opts.rawData   = result["raw"].as<bool>();
        opts.randomize = result["random"].as<bool>();
        opts.runSilently = result["quiet"].as<bool>();

        // Benchmark groups, using lowercase keys
        int groupCount = 0;
        auto check = [&](const char* key, bool &flag){
            flag = result[key].as<bool>();
            if (flag) ++groupCount;
        };

        check("l3",            opts.runL3);
        check("l2",            opts.runL2);
        check("l1",            opts.runL1);
        check("scalar",        opts.runScalar);
        check("constant",      opts.runConstant);
        check("readonly",      opts.runReadOnly);
        check("texture",       opts.runTexture);
        check("shared",        opts.runSharedMemory);
        check("memory",        opts.runMainMemory);
        check("departuredelay",opts.runDepartureDelay);
        check("resourceshare", opts.runResourceSharing);
        
        
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

        // If no groups are selected, run all by default
        if (groupCount == 0) {
            opts.runL3 = opts.runL2 = opts.runL1 =
            opts.runScalar = opts.runConstant = opts.runReadOnly =
            opts.runTexture = opts.runSharedMemory = opts.runMainMemory =
            opts.runDepartureDelay = opts.runResourceSharing = true;
        }

        #ifdef __HIP_PLATFORM_AMD__
        opts.runConstant = opts.runReadOnly = opts.runTexture = false;
        #endif
        #ifdef __HIP_PLATFORM_NVIDIA__
        opts.runL3 = opts.runScalar = false;
        #endif

        return opts;
    }
}