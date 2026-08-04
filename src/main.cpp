#include <cxxopts.hpp>
#include <nlohmann/json.hpp>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <memory>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <optional>
#include <type_traits>
#include <chrono>
#include <utility>

#include "version.hpp"
#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"
#include "utils/silent.hpp"

static constexpr auto MIN_EXPECTED_LINE_SIZE = 4;// Bytes
static constexpr auto VALIDITY_THRESHOLD = 0.5;// Factor

namespace {
    // Wall-clock timer for a single benchmark. Wrap an invocation in operator()
    // to time its full execution; the benchmark result is passed through unchanged.
    // When disabled, it only invokes the benchmark.
    struct BenchTimer {
        std::vector<std::pair<std::string, double>> timings;
        bool enabled = false;

        template <typename F>
        auto operator()(const std::string& name, F&& benchmarkCall) {
            const auto start = std::chrono::steady_clock::now();
            if constexpr (std::is_void_v<decltype(benchmarkCall())>) {
                benchmarkCall();
                record(name, start);
            } else {
                auto value = benchmarkCall();
                record(name, start);
                return value;
            }
        }

        void record(const std::string& name, std::chrono::steady_clock::time_point start) {
            if (!enabled) return;
            double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            timings.emplace_back(name, seconds);
            std::cout << "[Timing] " << name << " took " << seconds << " s" << std::endl;
        }
    };
}

int main(int argc, char* argv[]) {
    CLIOptions opts = util::parseCommandLine(argc, argv);

    std::unique_ptr<util::SilentMode> silencer;
    if (opts.runSilently) {
        silencer = std::make_unique<util::SilentMode>();
    }

    util::hipCheck(hipSetDevice(opts.deviceId));
    auto deviceProperties = util::getDeviceProperties();

    std::string fancyName = deviceProperties.name;

    std::cout << "[mt4g] Starting Benchmarks on " << fancyName << std::endl;

    std::string fancyFileName;
    if (opts.fileName.empty()) {
        fancyFileName.resize(fancyName.size());
        std::replace_copy(fancyName.begin(), fancyName.end(), fancyFileName.begin(), ' ', '_');
    } else {
        fancyFileName = opts.fileName;
    }

    std::filesystem::path graphDir = opts.location / ("results/" + fancyFileName);
    if (opts.graphs || opts.rawData || opts.fullReport) {
        std::error_code ec;
        std::filesystem::create_directories(graphDir, ec);
        if (ec) {
            std::cerr << "Could not create graph directory '" << graphDir.string() << "': " << ec.message() << std::endl;
        }
    }

    nlohmann::json metaInfo = {
        {"mt4gVersion", MT4G_VERSION},
        {"timestamp", util::getCurrentTimestamp()},
        {"hostCompiler", util::getHostCompilerVersion()}
    };
    if (auto gpuCompiler = util::getGpuCompilerVersion()) metaInfo["gpuCompiler"] = *gpuCompiler;
    if (auto cpu = util::getHostCpuModel()) metaInfo["hostCpu"] = *cpu;
    if (auto os = util::getOsDescription()) metaInfo["os"] = *os;
    if (auto driver = util::getDriverVersion()) metaInfo["driver"] = *driver;
    if (auto runtimeVersion = util::getRuntimeVersion()) metaInfo["runtime"] = *runtimeVersion;
    if (auto hostname = util::getHostname()) metaInfo["hostname"] = *hostname;

    nlohmann::json result = {
        {"meta", metaInfo},
        {
            "general", {
                {"name", deviceProperties.name},
                {"vendor", util::getVendor()},
                {"computeCapability", {
                    {"major", deviceProperties.major},
                    {"minor", deviceProperties.minor}
                }},
                {
                    "clockRate", {
                        {"value", deviceProperties.clockRate},
                        {"unit", "kHz"}
                    },
                },
                {"asicRevision", deviceProperties.asicRevision}
            }
        }, {
            "compute", {
                {"multiProcessorCount", deviceProperties.multiProcessorCount},
                {"numberOfCoresPerMultiProcessor", util::getNumberOfCoresPerSM()},
                {"maxThreadsPerBlock", deviceProperties.maxThreadsPerBlock},
                {"regsPerBlock", deviceProperties.regsPerBlock},
                {"regsPerMultiProcessor", deviceProperties.regsPerMultiprocessor},
                {"warpSize", deviceProperties.warpSize},
                {"supportsCooperativeLaunch", util::supportsCooperativeLaunch()},
                {"concurrentKernels", deviceProperties.concurrentKernels != 0},
                {"maxThreadsPerMultiProcessor", deviceProperties.maxThreadsPerMultiProcessor},
                {"maxBlocksPerMultiProcessor", deviceProperties.maxBlocksPerMultiProcessor},
                #ifdef __HIP_PLATFORM_AMD__
                {"numXCDs", util::getNumXCDs()},
                {"computeUnitsPerDie", util::getComputeUnitsPerDie()},
                {"numSIMDsPerCU", util::getSIMDsPerCU()},
                //{"logicalCUIdToPhysical", util::getLogicalToPhysicalCUsLUT()} // Not reliable on CDNA 3
                #endif
            }
        }, {
            "memory", {
                {
                    "main", {
                        {
                            "memoryClockRate", {
                                {"value", deviceProperties.memoryClockRate},
                                {"unit", "kHz"}
                            },
                        },
                        {
                            "totalGlobalMem", {
                                {"value", deviceProperties.totalGlobalMem},
                                {"unit", "bytes"}
                            },
                        },
                        {
                            "memoryBusWidth", {
                                {"value", deviceProperties.memoryBusWidth},
                                {"unit", "bit"}
                            },
                        },
                        /* // Not reliable
                        {
                            "theoreticalMaxBandwidth", {
                                {"value", util::getTheoreticalMaxGlobalMemoryBandwidthGiBs()},
                                {"unit", "GiB/s"},
                            } 
                        }
                        */
                    }
                }, {
                    "l2", {
                        {
                            "size", {
                                {"value", deviceProperties.l2CacheSize},
                                {"unit", "bytes"}
                            },
                        },
                        {
                            "persistingL2CacheMaxSize", {
                                {"value", deviceProperties.persistingL2CacheMaxSize},
                                {"unit", "bytes"}
                            }
                        }
                    }
                }, {
                    "constant", {
                        {
                            "totalConstMem", {
                                {"value", deviceProperties.totalConstMem},
                                {"unit", "bytes"}
                            },
                        }
                    }
                }, {
                    "shared", {
                        {
                            "sharedMemPerBlock", {
                                {"value", deviceProperties.sharedMemPerBlock},
                                {"unit", "bytes"}
                            }
                        }, {
                            "sharedMemPerMultiProcessor", {
                                {"value", deviceProperties.sharedMemPerMultiprocessor},
                                {"unit", "bytes"}
                            }
                        }, {
                            "reservedSharedMemPerBlock", {
                                {"value", deviceProperties.reservedSharedMemPerBlock},
                                {"unit", "bytes"}
                            }
                        }
                    }
                }, {
                    "l1", {
                        {"globalL1CacheSupported", deviceProperties.globalL1CacheSupported != 0},
                        {"localL1CacheSupported", deviceProperties.localL1CacheSupported != 0},

                    }
                }
            },
        }
    };

    #ifdef __HIP_PLATFORM_AMD__
    auto l2Size = util::getL2SizeBytes();
    auto l2Amount = util::getL2Amount();
    if (l2Size.has_value() && l2Amount.has_value()) {
        result["memory"]["l2"]["size"] = {
            {"value", l2Size.value()},
            {"unit", "bytes"}
        };
        result["memory"]["l2"]["amount"] = l2Amount.value();
    } else {
        result["memory"]["l2"]["size"] = {
            {"value", deviceProperties.l2CacheSize},
            {"unit", "bytes"}
        };
    }
    auto l2LineSize = util::getL2LineSizeBytes();
    if (l2LineSize.has_value()) {
        result["memory"]["l2"]["lineSize"] = {
            {"value", l2LineSize.value()},
            {"unit", "bytes"}
        };
    }
    auto l3Size = util::getL3SizeBytes();
    if (l3Size.has_value()) {
        result["memory"]["l3"]["size"] = {
            {"value", l3Size.value()},
            {"unit", "bytes"}
        };
        auto l3Amount = util::getL3Amount();
        if (l3Amount.has_value()) {
            result["memory"]["l3"]["amount"] = l3Amount.value();
        }
        auto l3LineSize = util::getL3LineSizeBytes();
        if (l3LineSize.has_value()) {
            result["memory"]["l3"]["lineSize"] = {
                {"value", l3LineSize.value()},
                {"unit", "bytes"}
            };
        }
    }
    #endif

    // Timing instrumentation (-t/--timing). Times the full benchmark run here;
    // every individual benchmark is timed by wrapping its call in `timed`.
    BenchTimer timed{{}, opts.timing};
    auto totalStart = std::chrono::steady_clock::now();

    if (opts.runL1) {
        std::cout << "[L1] Starting Benchmarks" << std::endl;
        std::cout << "[L1] Latency" << std::endl;
        CacheLatencyResult l1Latency = timed("l1Latency", [&] { return benchmark::measureL1Latency(); });
        result["memory"]["l1"]["latency"] = l1Latency;

        std::cout << "[L1] Fetch Granularity" << std::endl;
        CacheSizeResult l1FetchGranularity = timed("l1FetchGranularity", [&] { return benchmark::measureL1FetchGranularity(); });
        result["memory"]["l1"]["fetchGranularity"] = l1FetchGranularity;

        std::cout << "[L1] Size" << std::endl;
        CacheSizeResult l1Size = timed("l1CacheSize", [&] { return benchmark::measureL1Size(l1FetchGranularity.confidence > VALIDITY_THRESHOLD ? l1FetchGranularity.size : MIN_EXPECTED_LINE_SIZE); });
        result["memory"]["l1"]["size"] = l1Size;

        if (l1FetchGranularity.confidence > VALIDITY_THRESHOLD && l1Size.confidence > VALIDITY_THRESHOLD) {
            std::cout << "[L1] Line Size" << std::endl;
            CacheLineSizeResult l1LineSize = timed("l1LineSize", [&] { return benchmark::measureL1LineSize(l1Size.size, l1FetchGranularity.size); });
            result["memory"]["l1"]["lineSize"] = l1LineSize;
            if (opts.graphs) {
                util::exportChartsReduced(l1LineSize.timings, util::average<uint32_t>, fancyName + " - L1 Line Size", {}, "Bytes", "Cycles", graphDir.string());
            }
            if (opts.rawData) {
                util::writeNestedMapToFile(l1LineSize.timings, (graphDir / (fancyFileName + "__L1_Line_Size.txt")).string());
            }

            if (l1LineSize.confidence > VALIDITY_THRESHOLD) {
                std::cout << "[L1] Miss Penalty" << std::endl;
                double l1MissPenalty = timed("l1MissPenalty", [&] { return benchmark::measureL1MissPenalty(l1Size.size, l1LineSize.size, l1Latency.mean); });
                result["memory"]["l1"]["missPenalty"] = {
                    {"value", l1MissPenalty},
                    {"unit", "cycles"}
                };

                std::cout << "[L1] Amount" << std::endl;
                auto l1Amount = timed("l1Amount", [&] { return benchmark::measureL1Amount(l1Size.size, l1FetchGranularity.size, l1MissPenalty); });
                if (l1Amount.has_value()) {
                    result["memory"]["l1"]["amountPerMultiprocessor"] = *l1Amount;
                } else {
                    std::cout << "Could not measure valid L1 Amount, skipping L1 Amount benchmark." << std::endl;
                }
            } else {
                std::cout << "Could not measure valid L1 Line Size, skipping L1 Miss Penalty and Amount benchmarks." << std::endl;
            }
        } else {
            std::cout << "Could not measure valid L1 Size or Fetch Granularity, skipping L1 Line Size, Amount and Miss Penalty benchmarks." << std::endl;
        }

        if (l1Size.confidence > VALIDITY_THRESHOLD) {
            
            if (opts.runOptimalSearch)
            {
                std::cout << "[L1] Read Bandwidth per CU / MultiProcessor with optimal search" << std::endl;
                CacheBandwidthResult l1ReadBandwidth;
                if (util::isAMD()) 
                {
                    l1ReadBandwidth = timed("amd_l1ReadBandwidthBlocksweep", [&] { return benchmark::amd::measureL1ReadBandwidthBlockSweep(l1Size.size / 2); });
                } else {
                    l1ReadBandwidth = timed("l1ReadBandwidth", [&] { return benchmark::measureL1ReadBandwidthSweep(l1Size.size / 2); });
                }
                result["memory"]["l1"]["readBandwidthPerCU"] = l1ReadBandwidth;

                std::cout << "[L1] Write Bandwidth per CU / MultiProcessor with optimal search" << std::endl;
                CacheBandwidthResult l1WriteBandwidth;
                if (util::isAMD())
                {
                    l1WriteBandwidth = timed("amd_l1WriteBandwidthBlocksweep", [&] { return benchmark::amd::measureL1WriteBandwidthBlockSweep(l1Size.size / 2); });
                } else {
                    l1WriteBandwidth = timed("l1WriteBandwidth", [&] { return benchmark::measureL1WriteBandwidthSweep(l1Size.size / 2); });
                }
                result["memory"]["l1"]["writeBandwidthPerCU"] = l1WriteBandwidth;

                if (opts.rawData || opts.graphs)
                {
                    // vL1d on AMD (vector L1d); plain L1 on NVIDIA. The grid CSV
                    // backs the block-sweep figure: blocks->subplot, threads->line,
                    // reps->x, bandwidth->y.
                    const std::string l1Label = util::isAMD() ? "vL1d" : "L1";
                    util::writeBandwidthGridToCSV(l1ReadBandwidth, (graphDir / util::bandwidthGridFileName(fancyFileName, l1Label, "Read")).string());
                    util::writeBandwidthGridToCSV(l1WriteBandwidth, (graphDir / util::bandwidthGridFileName(fancyFileName, l1Label, "Write")).string());
                }
            }
            else
            {
                std::cout << "[L1] Read Bandwidth per CU / MultiProcessor" << std::endl;
                result["memory"]["l1"]["readBandwidthPerCU"] = {
                    {"value", timed("l1ReadBandwidth", [&] { return benchmark::measureL1ReadBandwidth(l1Size.size / 2); })},
                    {"unit", "GiB/s"}
                };

                std::cout << "[L1] Write Bandwidth per CU / MultiProcessor" << std::endl;
                result["memory"]["l1"]["writeBandwidthPerCU"] = {
                    {"value", timed("l1WriteBandwidth", [&] { return benchmark::measureL1WriteBandwidth(l1Size.size / 2); })},
                    {"unit", "GiB/s"}
                };
            }
        } else {
            std::cout << "Could not measure valid L1 Size, skipping L1 Bandwidth benchmarks." << std::endl;
        }

        if (opts.graphs) {
            util::exportChartMinMaxAvgRed(l1Size.timings, fancyName + " - L1 Size", {l1Size.size}, "Bytes", "Cycles", graphDir.string());
            util::exportChartsMinMaxAvg(l1FetchGranularity.timings, fancyName + " - L1 Fetch Granularity", {l1FetchGranularity.size}, "Bytes", "Cycles", graphDir.string());
        }
        if (opts.rawData) {
            util::writeVectorToFile(l1Latency.timings, (graphDir / (fancyFileName + "__L1_Latency.txt")).string());
            util::writeMapToFile(l1FetchGranularity.timings, (graphDir / (fancyFileName + "__L1_Fetch_Granularity.txt")).string());
            util::writeMapToFile(l1Size.timings, (graphDir / (fancyFileName + "__L1_Size.txt")).string());
        }

        std::cout << "[L1] Benchmarks finished" << std::endl;
    }

    if (opts.runL2) {
        std::cout << "[L2] Starting Benchmarks" << std::endl;
        std::cout << "[L2] Latency" << std::endl;
        CacheLatencyResult l2Latency = timed("l2Latency", [&] { return benchmark::measureL2Latency(); });
        result["memory"]["l2"]["latency"] = l2Latency;

        std::cout << "[L2] Fetch Granularity" << std::endl;
        CacheSizeResult l2FetchGranularity = timed("l2FetchGranularity", [&] { return benchmark::measureL2FetchGranularity(); });
        result["memory"]["l2"]["fetchGranularity"] = l2FetchGranularity;

        if (util::isAMD()) {
            std::cout << "L2 Segment Size is currently broken on AMD. Skipping." << std::endl;
        } else {
            std::cout << "[L2] Segment Size" << std::endl;
            CacheSizeResult l2SegmentSize = timed("l2SegmentSize", [&] { return benchmark::measureL2SegmentSize(deviceProperties.l2CacheSize, l2FetchGranularity.confidence > VALIDITY_THRESHOLD ? l2FetchGranularity.size : MIN_EXPECTED_LINE_SIZE); });
            result["memory"]["l2"]["segmentSize"] = l2SegmentSize;
            if (opts.graphs) {
                util::exportChartMinMaxAvgRed(l2SegmentSize.timings, fancyName + " - L2 Segment Size", {l2SegmentSize.size}, "Bytes", "Cycles", graphDir.string());
            }
            if (opts.rawData) {
                util::writeMapToFile(l2SegmentSize.timings, (graphDir / (fancyFileName + "__L2_Segment_Size.txt")).string());
            }
        }

        
        auto l2LineSizeValue = util::getNumeric<size_t>(result, "memory", "l2", "lineSize", "value");
        if (!l2LineSizeValue.has_value()) {
            if (l2FetchGranularity.confidence > VALIDITY_THRESHOLD) {
                std::cout << "[L2] Line Size" << std::endl;
                CacheLineSizeResult l2LineSize = timed("l2LineSize", [&] { return benchmark::measureL2LineSize(deviceProperties.l2CacheSize, l2FetchGranularity.size); }); // Unreliable on AMD because L2 Size Benchmarks are complicated
                result["memory"]["l2"]["lineSize"] = l2LineSize;
                if (opts.graphs) {
                    util::exportChartsReduced(l2LineSize.timings, util::average<uint32_t>, fancyName + " - L2 Line Size", {}, "Bytes", "Cycles", graphDir.string());
                }
                if (opts.rawData) {
                    util::writeNestedMapToFile(l2LineSize.timings, (graphDir / (fancyFileName + "__L2_Line_Size.txt")).string());
                }
                if (l2LineSize.confidence > VALIDITY_THRESHOLD) {
                    l2LineSizeValue = l2LineSize.size;
                }
            } else {
                std::cout << "Could not measure valid L2 Fetch Granularitys, skipping L2 Line Size benchmarks." << std::endl;
            }
        }
        if (l2LineSizeValue.has_value()) {
            std::cout << "[L2] Miss Penalty" << std::endl;
            double l2MissPenalty = timed("l2MissPenalty", [&] { return benchmark::measureL2MissPenalty(deviceProperties.l2CacheSize, l2LineSizeValue.value(), l2Latency.mean); });
            result["memory"]["l2"]["missPenalty"] = {
                {"value", l2MissPenalty},
                {"unit", "cycles"}
            };
        } else {
            std::cout << "Could not gather valid L2 Line Size, skipping L2 Miss Penalty benchmarks." << std::endl;
        }

        if (opts.runOptimalSearch)
        {
            std::cout << "[L2] Read Bandwidth with optimal search" << std::endl;
            CacheBandwidthResult l2ReadBandwidth = timed("l2ReadBandwidth", [&] { return benchmark::measureL2ReadBandwidthSweep(deviceProperties.l2CacheSize); });
            result["memory"]["l2"]["readBandwidth"] = l2ReadBandwidth;

            std::cout << "[L2] Write Bandwidth with optimal search" << std::endl;
            CacheBandwidthResult l2WriteBandwidth = timed("l2WriteBandwidth", [&] { return benchmark::measureL2WriteBandwidthSweep(deviceProperties.l2CacheSize); });
            result["memory"]["l2"]["writeBandwidth"] = l2WriteBandwidth;

            if (opts.rawData || opts.graphs)
            {
                util::writeBandwidthGridToCSV(l2ReadBandwidth, (graphDir / util::bandwidthGridFileName(fancyFileName, "L2", "Read")).string());
                util::writeBandwidthGridToCSV(l2WriteBandwidth, (graphDir / util::bandwidthGridFileName(fancyFileName, "L2", "Write")).string());
            }
        }
        else
        {
            std::cout << "[L2] Read Bandwidth" << std::endl;
            result["memory"]["l2"]["readBandwidth"] = {
                {"value", timed("l2ReadBandwidth", [&] { return benchmark::measureL2ReadBandwidth(deviceProperties.l2CacheSize); })},
                {"unit", "GiB/s"}
            };
            std::cout << "[L2] Write Bandwidth" << std::endl;
            result["memory"]["l2"]["writeBandwidth"] = {
                {"value", timed("l2WriteBandwidth", [&] { return benchmark::measureL2WriteBandwidth(deviceProperties.l2CacheSize); })},
                {"unit", "GiB/s"}
            };
        }

        if (opts.graphs) {
            util::exportChartsMinMaxAvg(l2FetchGranularity.timings, fancyName + " - L2 Fetch Granularity", {l2FetchGranularity.size}, "Bytes", "Cycles", graphDir.string());
        }
        if (opts.rawData) {
            util::writeVectorToFile(l2Latency.timings, (graphDir / (fancyFileName + "__L2_Latency.txt")).string());
            util::writeMapToFile(l2FetchGranularity.timings, (graphDir / (fancyFileName + "__L2_Fetch_Granularity.txt")).string());
        }

        std::cout << "[L2] Benchmarks finished" << std::endl;
    }

    if (opts.runL3) {
        std::cout << "[L3] Starting Benchmarks" << std::endl;
        auto l3Size = util::getL3SizeBytes();

        if (l3Size.has_value()) { // If flase we assume this GPU does not have an L3, therefore skipping
            result["memory"]["l3"]["size"] = {
                {"value", *l3Size},
                {"unit", "bytes"}
            };

            std::cout << "[L3] Latency" << std::endl;
            CacheLatencyResult l3Latency = timed("amd_l3Latency", [&] { return benchmark::amd::measureL3Latency(deviceProperties.l2CacheSize, 128); });
            result["memory"]["l3"]["latency"] = l3Latency;

            /* Not working yet
            std::cout << "[L3] Fetch Granularity" << std::endl;
            CacheSizeResult l3FetchGranularity = benchmark::amd::measureL3FetchGranularity();
            result["memory"]["l3"]["fetchGranularity"] = l3FetchGranularity;
            */
            auto l3LineSize = util::getNumeric<size_t>(result, "memory", "l3", "lineSize", "value");
            if (l3LineSize.has_value()) {
                std::cout << "[L3] Miss Penalty" << std::endl;
                result["memory"]["l3"]["missPenalty"] = timed("amd_l3MissPenalty", [&] { return benchmark::amd::measureL3MissPenalty(l3Size.value(), l3LineSize.value(), l3Latency.mean); });
            } else {
                std::cout << "Could not determine L3 Line Size, L3 Line Size will not be part of the output + skipping Miss Penalty benchmarks." << std::endl;
            }

            if (opts.runOptimalSearch)
            {
                std::cout << "[L3] Read Bandwidth with optimal search" << std::endl;
                CacheBandwidthResult l3ReadBandwidth = timed("amd_l3ReadBandwidth", [&] { return benchmark::amd::measureL3ReadBandwidthSweep(deviceProperties.l2CacheSize, l3Size.value()); });
                result["memory"]["l3"]["readBandwidth"] = l3ReadBandwidth;

                std::cout << "[L3] Write Bandwidth with optimal search" << std::endl;
                CacheBandwidthResult l3WriteBandwidth = timed("amd_l3WriteBandwidth", [&] { return benchmark::amd::measureL3WriteBandwidthSweep(deviceProperties.l2CacheSize, l3Size.value()); });
                result["memory"]["l3"]["writeBandwidth"] = l3WriteBandwidth;

                if (opts.rawData || opts.graphs)
                {
                    util::writeBandwidthGridToCSV(l3ReadBandwidth, (graphDir / util::bandwidthGridFileName(fancyFileName, "L3", "Read")).string());
                    util::writeBandwidthGridToCSV(l3WriteBandwidth, (graphDir / util::bandwidthGridFileName(fancyFileName, "L3", "Write")).string());
                }
            }
            else
            {
                std::cout << "[L3] Read Bandwidth" << std::endl;
                result["memory"]["l3"]["readBandwidth"] = {
                    {"value", timed("amd_l3ReadBandwidth", [&] { return benchmark::amd::measureL3ReadBandwidth(deviceProperties.l2CacheSize, l3Size.value()); })},
                    {"unit", "GiB/s"}
                };

                std::cout << "[L3] Write Bandwidth" << std::endl;
                result["memory"]["l3"]["writeBandwidth"] = {
                    {"value", timed("amd_l3WriteBandwidth", [&] { return benchmark::amd::measureL3WriteBandwidth(deviceProperties.l2CacheSize, l3Size.value()); })},
                    {"unit", "GiB/s"}
                };
            }

            /* Not working yet
            if (opts.graphs) {
            util::exportChartsMinMaxAvg(l3FetchGranularity.timings, fancyName + " - L3 Fetch Granularity", {l3FetchGranularity.size}, "Bytes", "Cycles", graphDir.string());
            }
            */

            if (opts.rawData) {
                util::writeVectorToFile(l3Latency.timings, (graphDir / (fancyName + "__L3_Latency.txt")).string());
                /* Not working yet
                util::writeMapToFile(l3FetchGranularity.timings, (graphDir / (fancyName + " - L3 Fetch Granularity.txt")).string());
                */
            }
        } else {
            std::cout << "[L3] Could not determine L3 Cache Size, probably because this GPU does not have an L3, skipping benchmarks." << std::endl;
        }
        std::cout << "[L3] Benchmarks finished" << std::endl;
    }

    if (opts.runConstant) {
        std::cout << "[Constant] Starting Benchmarks" << std::endl;
        std::cout << "[Constant] L1 Latency" << std::endl;
        CacheLatencyResult constantL1Latency = timed("nvidia_constantL1Latency", [&] { return benchmark::nvidia::measureConstantL1Latency(); });
        result["memory"]["constant"]["l1"]["latency"] = constantL1Latency;

        std::cout << "[Constant] L1 Fetch Granularity" << std::endl;
        CacheSizeResult constantL1FetchGranularity = timed("nvidia_constantL1FetchGranularity", [&] { return benchmark::nvidia::measureConstantL1FetchGranularity(); });
        result["memory"]["constant"]["l1"]["fetchGranularity"] = constantL1FetchGranularity;

        std::cout << "[Constant] L1.5 Fetch Granularity" << std::endl;
        CacheSizeResult constantL15FetchGranularity = timed("nvidia_constantL15FetchGranularity", [&] { return benchmark::nvidia::measureConstantL15FetchGranularity(constantL1FetchGranularity.size); });
        result["memory"]["constant"]["l1.5"]["fetchGranularity"] = constantL15FetchGranularity;

        CacheSizeResult constantL1Size = timed("nvidia_constantL1CacheSize", [&] { return benchmark::nvidia::measureConstantL1Size(constantL1FetchGranularity.size); });
        CacheSizeResult constantL15Size = timed("nvidia_constantL15CacheSize", [&] { return benchmark::nvidia::measureConstantL15Size(constantL15FetchGranularity.size); });
        result["memory"]["constant"]["l1"]["size"] = constantL1Size;
        result["memory"]["constant"]["l1.5"]["size"] = constantL15Size;

        if (constantL1Size.confidence > VALIDITY_THRESHOLD && constantL1FetchGranularity.confidence > VALIDITY_THRESHOLD) {
            std::cout << "[Constant] L1 Line Size" << std::endl;
            CacheLineSizeResult constantL1LineSize = timed("nvidia_constantL1LineSize", [&] { return benchmark::nvidia::measureConstantL1LineSize(constantL1Size.size, constantL1FetchGranularity.size); });
            result["memory"]["constant"]["l1"]["lineSize"] = constantL1LineSize;
            if (opts.graphs) {
                util::exportChartsReduced(constantL1LineSize.timings, util::average<uint32_t>, fancyName + " - Constant L1 Line Size", {}, "Bytes", "Cycles", graphDir.string());
            }
            if (opts.rawData) {
                util::writeNestedMapToFile(constantL1LineSize.timings, (graphDir / (fancyFileName + "__Constant_L1_Line_Size.txt")).string());
            }

            if (constantL1LineSize.confidence > VALIDITY_THRESHOLD) {
                std::cout << "[Constant] L1 Miss Penalty" << std::endl;
                double constantL1MissPenalty = timed("nvidia_constantL1MissPenalty", [&] { return benchmark::nvidia::measureConstantL1MissPenalty(constantL1Size.size, constantL1LineSize.size, constantL1Latency.mean); });
                result["memory"]["constant"]["l1"]["missPenalty"] = {
                    {"value", constantL1MissPenalty},
                    {"unit", "cycles"}
                };
                
                
                std::cout << "[Constant] L1 Amount" << std::endl;
                auto constantL1Amount = timed("nvidia_constantL1Amount", [&] { return benchmark::nvidia::measureConstantL1Amount(constantL1Size.size, constantL1FetchGranularity.size, constantL1MissPenalty); });
                if (constantL1Amount.has_value()) {
                    result["memory"]["constant"]["l1"]["amountPerMultiprocessor"] = *constantL1Amount;
                } else {
                    std::cout << "Could not measure valid Constant L1 Amount, skipping Constant L1 Amount benchmark." << std::endl;
                }
            } else {
                std::cout << "Could not measure valid Constant L1 Line Size, skipping Constant L1 Miss Penalty and Amount benchmarks." << std::endl;
            }

            std::cout << "[Constant] L1.5 Latency" << std::endl;
            CacheLatencyResult constantL15Latency = timed("nvidia_constantL15Latency", [&] { return benchmark::nvidia::measureConstantL15Latency(8 * KiB, constantL1FetchGranularity.size); });
            result["memory"]["constant"]["l1.5"]["latency"] = constantL15Latency;
            if (opts.rawData) {
                util::writeVectorToFile(constantL15Latency.timings, (graphDir / (fancyFileName + "__Constant_L1.5_Latency.txt")).string());
            }
        } else {
            std::cout << "Could not measure valid Constant L1 Size or Fetch Granularity, skipping Constant L1 Amount, Line Size, Miss Penalty and Constant L1.5 Latency benchmarks." << std::endl;
        }

        if (constantL15Size.confidence > VALIDITY_THRESHOLD && constantL1FetchGranularity.confidence > VALIDITY_THRESHOLD) {
            std::cout << "[Constant] L1.5 Line Size" << std::endl;
            CacheLineSizeResult constantL15LineSize = timed("nvidia_constantL15LineSize", [&] { return benchmark::nvidia::measureConstantL15LineSize(constantL15Size.size, constantL15FetchGranularity.size); });
            result["memory"]["constant"]["l1.5"]["lineSize"] = constantL15LineSize;
            if (opts.graphs) {
                util::exportChartsReduced(constantL15LineSize.timings, util::average<uint32_t>, fancyName + " - Constant L1.5 Line Size", {}, "Bytes", "Cycles", graphDir.string());
            }
            if (opts.rawData) {
                util::writeNestedMapToFile(constantL15LineSize.timings, (graphDir / (fancyFileName + "__Constant_L1.5_Line_Size.txt")).string());
            }
        } else {
            std::cerr << "Could not measure valid Constant L1.5 Size or Fetch Granularity, skipping Constant L1.5 Line Size benchmarks." << std::endl;
        }

        if (constantL1Size.confidence > VALIDITY_THRESHOLD) {
            if (opts.runOptimalSearch)
            {
                std::cout << "[Constant] L1 Read Bandwidth per CU / MultiProcessor with optimal search" << std::endl;
                CacheBandwidthResult constantL1ReadBandwidth = timed("nvidia_constantL1ReadBandwidth", [&] { return benchmark::nvidia::measureConstantL1ReadBandwidthSweep(constantL1Size.size / 2); });
                result["memory"]["constant"]["l1"]["readBandwidthPerCU"] = constantL1ReadBandwidth;

                if (opts.rawData || opts.graphs)
                {
                    util::writeBandwidthGridToCSV(constantL1ReadBandwidth, (graphDir / util::bandwidthGridFileName(fancyFileName, "ConstantL1", "Read")).string());
                }
            }
            else
            {
                std::cout << "[Constant] L1 Read Bandwidth per CU / MultiProcessor" << std::endl;
                result["memory"]["constant"]["l1"]["readBandwidthPerCU"] = {
                    {"value", timed("nvidia_constantL1ReadBandwidth", [&] { return benchmark::nvidia::measureConstantL1ReadBandwidth(constantL1Size.size / 2); })},
                    {"unit", "GiB/s"}
                };
            }
        } else {
            std::cout << "Could not measure valid Constant L1 Size, skipping Constant L1 Bandwidth benchmarks." << std::endl;
        }

        // No change point means the L1.5 spans the constant array, not that measurement failed.
        // Bandwidth only needs a working set above L1 and within the array, so use half the
        // constant array instead of skipping when confidence is zero.
        size_t constantL15BandwidthBytes = constantL15Size.confidence > VALIDITY_THRESHOLD
            ? constantL15Size.size / 2
            : 32 * KiB;
        // Stride by one fetch granularity per load so each access hits a fresh cache line.
        // This avoids L1 reuse and isolates L1.5 bandwidth, matching the Constant L1.5
        // Size and Latency benchmarks.
        size_t constantL15BandwidthStride = constantL15FetchGranularity.confidence > VALIDITY_THRESHOLD
            ? constantL15FetchGranularity.size
            : MIN_EXPECTED_LINE_SIZE;

        if (opts.runOptimalSearch)
        {
            std::cout << "[Constant] L1.5 Read Bandwidth per CU / MultiProcessor with optimal search" << std::endl;
            CacheBandwidthResult constantL15ReadBandwidth = timed("nvidia_constantL15ReadBandwidth", [&] { return benchmark::nvidia::measureConstantL15ReadBandwidthSweep(constantL15BandwidthBytes, constantL15BandwidthStride); });
            result["memory"]["constant"]["l1.5"]["readBandwidthPerCU"] = constantL15ReadBandwidth;

            if (opts.rawData || opts.graphs)
            {
                util::writeBandwidthGridToCSV(constantL15ReadBandwidth, (graphDir / util::bandwidthGridFileName(fancyFileName, "ConstantL1.5", "Read")).string());
            }
        }
        else
        {
            std::cout << "[Constant] L1.5 Read Bandwidth per CU / MultiProcessor" << std::endl;
            result["memory"]["constant"]["l1.5"]["readBandwidthPerCU"] = {
                {"value", timed("nvidia_constantL15ReadBandwidth", [&] { return benchmark::nvidia::measureConstantL15ReadBandwidth(constantL15BandwidthBytes, constantL15BandwidthStride); })},
                {"unit", "GiB/s"}
            };
        }

        if (opts.graphs) {
            util::exportChartMinMaxAvgRed(constantL1Size.timings, fancyName + " - Constant L1 Size", {constantL1Size.size}, "Bytes", "Cycles", graphDir.string());
            util::exportChartsMinMaxAvg(constantL1FetchGranularity.timings, fancyName + " - Constant L1 Fetch Granularity", {constantL1FetchGranularity.size}, "Bytes", "Cycles", graphDir.string());
            util::exportChartMinMaxAvgRed(constantL15Size.timings, fancyName + " - Constant L1.5 Size", {constantL15Size.size}, "Bytes", "Cycles", graphDir.string());
            util::exportChartsMinMaxAvg(constantL15FetchGranularity.timings, fancyName + " - Constant L1.5 Fetch Granularity", {constantL15FetchGranularity.size}, "Bytes", "Cycles", graphDir.string());
        }
        if (opts.rawData) {
            util::writeVectorToFile(constantL1Latency.timings, (graphDir / (fancyFileName + "__Constant_L1_Latency.txt")).string());
            util::writeMapToFile(constantL1FetchGranularity.timings, (graphDir / (fancyFileName + "__Constant_L1_Fetch_Granularity.txt")).string());
            util::writeMapToFile(constantL1Size.timings, (graphDir / (fancyFileName + "__Constant_L1_Size.txt")).string());
            util::writeMapToFile(constantL15FetchGranularity.timings, (graphDir / (fancyFileName + "__Constant_L1.5_Fetch_Granularity.txt")).string());
            util::writeMapToFile(constantL15Size.timings, (graphDir / (fancyFileName + "__Constant_L1.5_Size.txt")).string());
        }
        std::cout << "[Constant] Benchmarks finished" << std::endl;
    }

    if (opts.runReadOnly) {
        std::cout << "[Read Only] Starting Benchmarks" << std::endl;
        std::cout << "[Read Only] Latency" << std::endl;
        CacheLatencyResult readOnlyLatency = timed("nvidia_readOnlyLatency", [&] { return benchmark::nvidia::measureReadOnlyLatency(); });
        result["memory"]["readOnly"]["latency"] = readOnlyLatency;

        std::cout << "[Read Only] Fetch Granularity" << std::endl;
        CacheSizeResult readOnlyFetchGranularity = timed("nvidia_readOnlyFetchGranularity", [&] { return benchmark::nvidia::measureReadOnlyFetchGranularity(); });
        result["memory"]["readOnly"]["fetchGranularity"] = readOnlyFetchGranularity;

        std::cout << "[Read Only] Size" << std::endl;
        CacheSizeResult readOnlySize = timed("nvidia_readOnlyCacheSize", [&] { return benchmark::nvidia::measureReadOnlySize(readOnlyFetchGranularity.confidence > VALIDITY_THRESHOLD ? readOnlyFetchGranularity.size : MIN_EXPECTED_LINE_SIZE); });
        result["memory"]["readOnly"]["size"] = readOnlySize;

        if (readOnlySize.confidence > VALIDITY_THRESHOLD && readOnlyFetchGranularity.confidence > VALIDITY_THRESHOLD) {
            std::cout << "[Read Only] Line Size" << std::endl;
            CacheLineSizeResult readOnlyLineSize = timed("nvidia_readOnlyLineSize", [&] { return benchmark::nvidia::measureReadOnlyLineSize(readOnlySize.size, readOnlyFetchGranularity.size); });
            result["memory"]["readOnly"]["lineSize"] = readOnlyLineSize;
            if (opts.graphs) {
                util::exportChartsReduced(readOnlyLineSize.timings, util::average<uint32_t>, fancyName + " - Read Only Line Size", {}, "Bytes", "Cycles", graphDir.string());
            }
            if (opts.rawData) {
                util::writeNestedMapToFile(readOnlyLineSize.timings, (graphDir / (fancyFileName + "__Read_Only_Line_Size.txt")).string());
            }

            if (readOnlyLineSize.confidence > VALIDITY_THRESHOLD) {
                std::cout << "[Read Only] Miss Penalty" << std::endl;
                double readOnlyMissPenalty = timed("nvidia_readOnlyMissPenalty", [&] { return benchmark::nvidia::measureReadOnlyMissPenalty(readOnlySize.size, readOnlyLineSize.size, readOnlyLatency.mean); });
                result["memory"]["readOnly"]["missPenalty"] = {
                    {"value", readOnlyMissPenalty},
                    {"unit", "cycles"}
                };

                std::cout << "[Read Only] Amount" << std::endl;
                auto readOnlyAmount = timed("nvidia_readOnlyAmount", [&] { return benchmark::nvidia::measureReadOnlyAmount(readOnlySize.size, readOnlyFetchGranularity.size, readOnlyMissPenalty); });
                if (readOnlyAmount.has_value()) {
                    result["memory"]["readOnly"]["amountPerMultiprocessor"] = *readOnlyAmount;
                } else {
                    std::cout << "Could not measure valid Read Only Amount, skipping Read Only Amount benchmark." << std::endl;
                }
            } else {
                std::cout << "Could not measure valid Read Only Line Size, skipping Read Only Miss Penalty and Amount benchmarks." << std::endl;
            }
        } else {
            std::cout << "Could not measure valid Read Only Size or Fetch Granularity, skipping Read Only Amount, Line Size and Miss Penalty benchmarks." << std::endl;
        }

        if (readOnlySize.confidence > VALIDITY_THRESHOLD) {
            if (opts.runOptimalSearch) {
                std::cout << "[Read Only] Read Bandwidth per CU / MultiProcessor with optimal search" << std::endl;
                CacheBandwidthResult readOnlyReadBandwidth = timed("nvidia_readOnlyReadBandwidth", [&] { return benchmark::nvidia::measureReadOnlyReadBandwidthSweep(readOnlySize.size / 2); });
                result["memory"]["readOnly"]["readBandwidthPerCU"] = readOnlyReadBandwidth;

                if (opts.rawData || opts.graphs) {
                    util::writeBandwidthGridToCSV(readOnlyReadBandwidth, (graphDir / util::bandwidthGridFileName(fancyFileName, "ReadOnly", "Read")).string());
                }
            } else {
                std::cout << "[Read Only] Read Bandwidth per CU / MultiProcessor" << std::endl;
                result["memory"]["readOnly"]["readBandwidthPerCU"] = {
                    {"value", timed("nvidia_readOnlyReadBandwidth", [&] { return benchmark::nvidia::measureReadOnlyReadBandwidth(readOnlySize.size / 2); })},
                    {"unit", "GiB/s"}
                };
            }
        } else {
            std::cout << "Could not measure valid Read Only Size, skipping Read Only Bandwidth benchmarks." << std::endl;
        }

        if (opts.graphs) {
            util::exportChartMinMaxAvgRed(readOnlySize.timings, fancyName + " - Read Only Size", {readOnlySize.size}, "Bytes", "Cycles", graphDir.string());
            util::exportChartsMinMaxAvg(readOnlyFetchGranularity.timings, fancyName + " - Read Only Fetch Granularity", {readOnlyFetchGranularity.size}, "Bytes", "Cycles", graphDir.string());
        }
        if (opts.rawData) {
            util::writeVectorToFile(readOnlyLatency.timings, (graphDir / (fancyFileName + "__Read_Only_Latency.txt")).string());
            util::writeMapToFile(readOnlyFetchGranularity.timings, (graphDir / (fancyFileName + "__Read_Only_Fetch_Granularity.txt")).string());
            util::writeMapToFile(readOnlySize.timings, (graphDir / (fancyFileName + "__Read_Only_Size.txt")).string());
        }
        std::cout << "[Read Only] Benchmarks finished" << std::endl;
    }

    if (opts.runTexture) {
        std::cout << "[Texture] Starting Benchmarks" << std::endl;
        std::cout << "[Texture] Latency" << std::endl;
        CacheLatencyResult textureLatency = timed("nvidia_textureLatency", [&] { return benchmark::nvidia::measureTextureLatency(); });
        result["memory"]["texture"]["latency"] = textureLatency;

        std::cout << "[Texture] Fetch Granularity" << std::endl;
        CacheSizeResult textureFetchGranularity = timed("nvidia_textureFetchGranularity", [&] { return benchmark::nvidia::measureTextureFetchGranularity(); });
        result["memory"]["texture"]["fetchGranularity"] = textureFetchGranularity;

        std::cout << "[Texture] Size" << std::endl;
        CacheSizeResult textureSize = timed("nvidia_textureCacheSize", [&] { return benchmark::nvidia::measureTextureSize(textureFetchGranularity.confidence > VALIDITY_THRESHOLD ? textureFetchGranularity.size : MIN_EXPECTED_LINE_SIZE); });
        result["memory"]["texture"]["size"] = textureSize;

        if (textureSize.confidence > VALIDITY_THRESHOLD && textureFetchGranularity.confidence > VALIDITY_THRESHOLD) {
            std::cout << "[Texture] Line Size" << std::endl;
            CacheLineSizeResult textureLineSize = timed("nvidia_textureLineSize", [&] { return benchmark::nvidia::measureTextureLineSize(textureSize.size, textureFetchGranularity.size); });
            result["memory"]["texture"]["lineSize"] = textureLineSize;
            if (opts.graphs) {
                util::exportChartsReduced(textureLineSize.timings, util::average<uint32_t>, fancyName + " - Texture Line Size", {}, "Bytes", "Cycles", graphDir.string());
            }
            if (opts.rawData) {
                util::writeNestedMapToFile(textureLineSize.timings, (graphDir / (fancyFileName + "__Texture_Line_Size.txt")).string());
            }

            if (textureLineSize.confidence > VALIDITY_THRESHOLD) {
                std::cout << "[Texture] Miss Penalty" << std::endl;
                double textureMissPenalty = timed("nvidia_textureMissPenalty", [&] { return benchmark::nvidia::measureTextureMissPenalty(textureSize.size, textureLineSize.size, textureLatency.mean); });
                result["memory"]["texture"]["missPenalty"] = {
                    {"value", textureMissPenalty},
                    {"unit", "cycles"}
                };
                
                std::cout << "[Texture] Amount" << std::endl;
                auto textureAmount = timed("nvidia_textureAmount", [&] { return benchmark::nvidia::measureTextureAmount(textureSize.size, textureFetchGranularity.size, textureMissPenalty); });
                if (textureAmount.has_value()) {
                    result["memory"]["texture"]["amountPerMultiprocessor"] = *textureAmount;
                } else {
                    std::cout << "Could not measure valid Texture Amount, skipping Texture Amount benchmark." << std::endl;
                }
            } else {
                std::cout << "Could not measure valid Texture Line Size, skipping Texture Miss Penalty and Amount benchmarks." << std::endl;
            }
        } else {
            std::cout << "Could not measure valid Texture Size or Fetch Granularity, skipping Texture Amount, Line Size and Miss Penalty benchmarks." << std::endl;
        }

        if (textureSize.confidence > VALIDITY_THRESHOLD) {
            if (opts.runOptimalSearch) {
                std::cout << "[Texture] Read Bandwidth per CU / MultiProcessor with optimal search" << std::endl;
                CacheBandwidthResult textureReadBandwidth = timed("nvidia_textureReadBandwidth", [&] { return benchmark::nvidia::measureTextureReadBandwidthSweep(textureSize.size / 2); });
                result["memory"]["texture"]["readBandwidthPerCU"] = textureReadBandwidth;

                if (opts.rawData || opts.graphs) {
                    util::writeBandwidthGridToCSV(textureReadBandwidth, (graphDir / util::bandwidthGridFileName(fancyFileName, "Texture", "Read")).string());
                }
            } else {
                std::cout << "[Texture] Read Bandwidth per CU / MultiProcessor" << std::endl;
                result["memory"]["texture"]["readBandwidthPerCU"] = {
                    {"value", timed("nvidia_textureReadBandwidth", [&] { return benchmark::nvidia::measureTextureReadBandwidth(textureSize.size / 2); })},
                    {"unit", "GiB/s"}
                };
            }
        } else {
            std::cout << "Could not measure valid Texture Size, skipping Texture Bandwidth benchmarks." << std::endl;
        }

        if (opts.graphs) {
            util::exportChartMinMaxAvgRed(textureSize.timings, fancyName + " - Texture Size", {textureSize.size}, "Bytes", "Cycles", graphDir.string());
            util::exportChartsMinMaxAvg(textureFetchGranularity.timings, fancyName + " - Texture Fetch Granularity", {textureFetchGranularity.size}, "Bytes", "Cycles", graphDir.string());
        }
        if (opts.rawData) {
            util::writeVectorToFile(textureLatency.timings, (graphDir / (fancyFileName + "__Texture_Latency.txt")).string());
            util::writeMapToFile(textureFetchGranularity.timings, (graphDir / (fancyFileName + "__Texture_Fetch_Granularity.txt")).string());
            util::writeMapToFile(textureSize.timings, (graphDir / (fancyFileName + "__Texture_Size.txt")).string());
        }
        std::cout << "[Texture] Benchmarks finished" << std::endl;
    }

    if (opts.runScalar) {
        std::cout << "[Scalar L1] Starting Benchmarks" << std::endl;
        std::cout << "[Scalar L1] Latency" << std::endl;
        CacheLatencyResult scalarL1Latency = timed("amd_scalarL1Latency", [&] { return benchmark::amd::measureScalarL1Latency(); });
        result["memory"]["scalarL1"]["latency"] = scalarL1Latency;

        std::cout << "[Scalar L1] Fetch Granularity" << std::endl;
        CacheSizeResult scalarL1FetchGranularity = timed("amd_scalarL1FetchGranularity", [&] { return benchmark::amd::measureScalarL1FetchGranularity(); });
        result["memory"]["scalarL1"]["fetchGranularity"] = scalarL1FetchGranularity;

        std::cout << "[Scalar L1] Size" << std::endl;
        CacheSizeResult scalarL1Size = timed("amd_scalarL1CacheSize", [&] { return benchmark::amd::measureScalarL1Size(scalarL1FetchGranularity.confidence > VALIDITY_THRESHOLD ? scalarL1FetchGranularity.size : MIN_EXPECTED_LINE_SIZE); });
        result["memory"]["scalarL1"]["size"] = scalarL1Size;


        if (scalarL1Size.confidence > VALIDITY_THRESHOLD && scalarL1FetchGranularity.confidence > VALIDITY_THRESHOLD) {
            std::cout << "[Scalar L1] Line Size" << std::endl;
            CacheLineSizeResult scalarL1LineSize = timed("amd_scalarL1LineSize", [&] { return benchmark::amd::measureScalarL1LineSize(scalarL1Size.size, scalarL1FetchGranularity.size); });
            result["memory"]["scalarL1"]["lineSize"] = scalarL1LineSize;
            if (opts.graphs) {
                util::exportChartsReduced(scalarL1LineSize.timings, util::average<uint32_t>, fancyName + " - Scalar L1 Line Size", {}, "Bytes", "Cycles", graphDir.string());
            }
            if (opts.rawData) {
                util::writeNestedMapToFile(scalarL1LineSize.timings, (graphDir / (fancyFileName + "__Scalar_L1_Line_Size.txt")).string());
            }

            if (scalarL1LineSize.confidence > VALIDITY_THRESHOLD) {
                std::cout << "[Scalar L1] Miss Penalty" << std::endl;
                double scalarL1MissPenalty = timed("amd_scalarL1MissPenalty", [&] { return benchmark::amd::measureScalarL1MissPenalty(scalarL1Size.size, scalarL1LineSize.size, scalarL1Latency.mean); });
                result["memory"]["scalarL1"]["missPenalty"] = {
                    {"value", scalarL1MissPenalty},
                    {"unit", "cycles"}
                };
            } else {
                std::cout << "Could not measure valid Scalar L1 Line Size, skipping Scalar L1 Miss Penalty benchmarks." << std::endl;
            }
            
            if (opts.runOptimalSearch)
            {
                std::cout << "[Scalar L1] Read Bandwidth per CU / MultiProcessor with optimal search" << std::endl;
                CacheBandwidthResult sL1ReadBandwidth = timed("amd_sL1ReadBandwidth", [&] { return benchmark::amd::measureScalarL1ReadBandwidthSweep(scalarL1Size.size / 2); });
                result["memory"]["scalarL1"]["readBandwidthPerCU"] = sL1ReadBandwidth;

                std::cout << "[Scalar L1] Write Bandwidth per CU / MultiProcessor with optimal search" << std::endl;
                CacheBandwidthResult sL1WriteBandwidth = timed("amd_sL1WriteBandwidth", [&] { return benchmark::amd::measureScalarL1WriteBandwidthSweep(scalarL1Size.size / 2); });
                result["memory"]["scalarL1"]["writeBandwidthPerCU"] = sL1WriteBandwidth;

                if (opts.rawData || opts.graphs)
                {
                    util::writeBandwidthGridToCSV(sL1ReadBandwidth, (graphDir / util::bandwidthGridFileName(fancyFileName, "sL1d", "Read")).string());
                    util::writeBandwidthGridToCSV(sL1WriteBandwidth, (graphDir / util::bandwidthGridFileName(fancyFileName, "sL1d", "Write")).string());
                }
            }
            else
            {
                std::cout << "[Scalar L1] Read Bandwidth per CU / MultiProcessor" << std::endl;
                result["memory"]["scalarL1"]["readBandwidthPerCU"] = {
                    {"value", timed("amd_sL1ReadBandwidth", [&] { return benchmark::amd::measureScalarL1ReadBandwidth(scalarL1Size.size / 2); })},
                    {"unit", "GiB/s"}
                };

                std::cout << "[Scalar L1] Write Bandwidth per CU / MultiProcessor" << std::endl;
                result["memory"]["scalarL1"]["writeBandwidthPerCU"] = {
                    {"value", timed("amd_sL1WriteBandwidth", [&] { return benchmark::amd::measureScalarL1WriteBandwidth(scalarL1Size.size / 2); })},
                    {"unit", "GiB/s"}
                };
            }

            std::cout << "[Scalar L1] CU Sharing" << std::endl;
            if (util::isCDNA3())
            {
                std::cout << "CU Sharing is currently not available on CDNA 3." << std::endl;
            }
            else
            {
                auto sharedBetweenCUs = timed("amd_cuShareScalarL1", [&] { return benchmark::amd::measureCuShareScalarL1(scalarL1Size.size, scalarL1FetchGranularity.size); });
                result["memory"]["scalarL1"]["sharedBetween"] = sharedBetweenCUs;
                result["memory"]["scalarL1"]["uniqueAmount"] = sharedBetweenCUs.size();
            }
        } else {
            std::cout << "Could not measure valid Scalar L1 Size or Fetch Granularity, skipping Scalar L1 Line Size, Miss Penalty, Bandwidth and CU Sharing benchmarks." << std::endl;
        }

        if (opts.graphs) {
            util::exportChartMinMaxAvgRed(scalarL1Size.timings, fancyName + " - Scalar L1 Size", {scalarL1Size.size}, "Bytes", "Cycles", graphDir.string());
            util::exportChartsMinMaxAvg(scalarL1FetchGranularity.timings, fancyName + " - Scalar L1 Fetch Granularity", {scalarL1FetchGranularity.size}, "Bytes", "Cycles", graphDir.string());
        }
        if (opts.rawData) {
            util::writeVectorToFile(scalarL1Latency.timings, (graphDir / (fancyFileName + "__Scalar_L1_Latency.txt")).string());
            util::writeMapToFile(scalarL1FetchGranularity.timings, (graphDir / (fancyFileName + "__Scalar_L1_Fetch_Granularity.txt")).string());
            util::writeMapToFile(scalarL1Size.timings, (graphDir / (fancyFileName + "__Scalar_L1_Size.txt")).string());
        }

        std::cout << "[Scalar L1] Benchmarks finished" << std::endl;
    }
    
    if (opts.runSharedMemory) {
        std::cout << "[Shared Memory] Starting Benchmarks" << std::endl;
        std::cout << "[Shared Memory] Latency" << std::endl;
        CacheLatencyResult sharedLatency = timed("sharedLatency", [&] { return benchmark::measureSharedMemoryLatency(); });
        result["memory"]["shared"]["latency"] = sharedLatency;
        if (opts.rawData) {
            util::writeVectorToFile(sharedLatency.timings, (graphDir / (fancyFileName + "__Shared_Memory_Latency.txt")).string());
        }

        if (opts.runOptimalSearch)
        {
            std::cout << "[Shared Memory] Read Bandwidth per CU / MultiProcessor with optimal search" << std::endl;
            CacheBandwidthResult sharedReadBandwidth;
            if (opts.sharedStatic)
            {
                sharedReadBandwidth = timed("sharedReadBandwidthStatic", [&] { return benchmark::measureSharedReadBandwidthStaticSweep(); });
            } else {
                sharedReadBandwidth = timed("sharedReadBandwidth", [&] { return benchmark::measureSharedReadBandwidthSweep(deviceProperties.sharedMemPerBlock / 2); });
            }
            result["memory"]["shared"]["readBandwidthPerCU"] = sharedReadBandwidth;

            std::cout << "[Shared Memory] Write Bandwidth per CU / MultiProcessor with optimal search" << std::endl;
            CacheBandwidthResult sharedWriteBandwidth;
            if (opts.sharedStatic)
            {
                sharedWriteBandwidth = timed("sharedWriteBandwidthStatic", [&] { return benchmark::measureSharedWriteBandwidthStaticSweep(); });
            } else {
                sharedWriteBandwidth = timed("sharedWriteBandwidth", [&] { return benchmark::measureSharedWriteBandwidthSweep(deviceProperties.sharedMemPerBlock / 2); });
            }
            result["memory"]["shared"]["writeBandwidthPerCU"] = sharedWriteBandwidth;

            if (opts.rawData || opts.graphs)
            {
                // Encode array size (KiB) and allocation type so the LDS
                // best-per-configuration figure can label each line, e.g.
                // "32_stat (T=512)". Combine several runs (sizes / dyn|stat) into
                // one figure via: plot_bandwidth.py auto --indir <results dir>.
                const std::string alloc = opts.sharedStatic ? "stat" : "dyn";
                const std::string readSuffix = std::to_string(sharedReadBandwidth.dataBytes / 1024) + "KiB_" + alloc;
                const std::string writeSuffix = std::to_string(sharedWriteBandwidth.dataBytes / 1024) + "KiB_" + alloc;
                util::writeBandwidthGridToCSV(sharedReadBandwidth, (graphDir / util::bandwidthGridFileName(fancyFileName, "LDS", "Read", readSuffix)).string());
                util::writeBandwidthGridToCSV(sharedWriteBandwidth, (graphDir / util::bandwidthGridFileName(fancyFileName, "LDS", "Write", writeSuffix)).string());
            }
        }
        else
        {
            std::cout << "[Shared Memory] Read Bandwidth per CU / MultiProcessor" << std::endl;
            if (opts.sharedStatic)
            {
                result["memory"]["shared"]["readBandwidthPerCU"] = {
                    {"value", timed("sharedReadBandwidthStatic", [&] { return benchmark::measureSharedReadBandwidthStatic(); })},
                    {"unit", "GiB/s"}
                };
            } else {
                result["memory"]["shared"]["readBandwidthPerCU"] = {
                    {"value", timed("sharedReadBandwidth", [&] { return benchmark::measureSharedReadBandwidth(deviceProperties.sharedMemPerBlock / 2); })},
                    {"unit", "GiB/s"}
                };
            }


            std::cout << "[Shared Memory] Write Bandwidth per CU / MultiProcessor" << std::endl;
            if (opts.sharedStatic)
            {
                result["memory"]["shared"]["writeBandwidthPerCU"] = {
                    {"value", timed("sharedWriteBandwidthStatic", [&] { return benchmark::measureSharedWriteBandwidthStatic(); })},
                    {"unit", "GiB/s"}
                };
            } else {
                result["memory"]["shared"]["writeBandwidthPerCU"] = {
                    {"value", timed("sharedWriteBandwidth", [&] { return benchmark::measureSharedWriteBandwidth(deviceProperties.sharedMemPerBlock / 2); })},
                    {"unit", "GiB/s"}
                };
            }
        }

        std::cout << "[Shared Memory] Benchmarks finished" << std::endl;
    }

    if (opts.runMainMemory) {
        std::cout << "[Main Memory] Starting Benchmarks" << std::endl;

        std::cout << "[Main Memory] Latency" << std::endl;
        CacheLatencyResult mainMemLatency = timed("mainMemoryLatency", [&] { return benchmark::measureMainMemoryLatency(); });
        result["memory"]["main"]["latency"] = mainMemLatency;
        if (opts.rawData) {
            util::writeVectorToFile(mainMemLatency.timings, (graphDir / (fancyFileName + "__Main_Memory_Latency.txt")).string());
        }

        if (opts.runOptimalSearch)
        {
            std::cout << "[Main Memory] Read Bandwidth with optimal search" << std::endl;
            CacheBandwidthResult mainMemReadBandwidth = timed("mainMemoryReadBandwidth", [&] { return benchmark::measureMainMemoryReadBandwidthSweep(deviceProperties.totalGlobalMem); });
            result["memory"]["main"]["readBandwidth"] = mainMemReadBandwidth;

            std::cout << "[Main Memory] Write Bandwidth with optimal search" << std::endl;
            CacheBandwidthResult mainMemWriteBandwidth = timed("mainMemoryWriteBandwidth", [&] { return benchmark::measureMainMemoryWriteBandwidthSweep(deviceProperties.totalGlobalMem); });
            result["memory"]["main"]["writeBandwidth"] = mainMemWriteBandwidth;

            if (opts.rawData || opts.graphs)
            {
                util::writeBandwidthGridToCSV(mainMemReadBandwidth, (graphDir / util::bandwidthGridFileName(fancyFileName, "MainMemory", "Read")).string());
                util::writeBandwidthGridToCSV(mainMemWriteBandwidth, (graphDir / util::bandwidthGridFileName(fancyFileName, "MainMemory", "Write")).string());
            }
        }
        else
        {
            std::cout << "[Main Memory] Read Bandwidth" << std::endl;
            result["memory"]["main"]["readBandwidth"] = {
                {"value", timed("mainMemoryReadBandwidth", [&] { return benchmark::measureMainMemoryReadBandwidth(deviceProperties.totalGlobalMem); })},
                {"unit", "GiB/s"}
            };

            std::cout << "[Main Memory] Write Bandwidth" << std::endl;
            result["memory"]["main"]["writeBandwidth"] = {
                {"value", timed("mainMemoryWriteBandwidth", [&] { return benchmark::measureMainMemoryWriteBandwidth(deviceProperties.totalGlobalMem); })},
                {"unit", "GiB/s"}
            };
        }

        std::cout << "[Main Memory] Benchmarks finished" << std::endl;
    }
    if (opts.runResourceSharing) {
        std::cout << "[Resource Sharing] Starting Benchmarks" << std::endl;

        if (opts.runConstant && opts.runL1) {
            timed("nvidia_constantL1SharedWithL1", [&] {
                util::sharedHelper(result["memory"]["constant"]["l1"], result["memory"]["l1"],
                             "Constant L1", "L1",
                             benchmark::nvidia::measureConstantL1AndL1Shared);
            });
        }
        if (opts.runReadOnly && opts.runL1) {
            timed("nvidia_readOnlySharedWithL1", [&] {
                util::sharedHelper(result["memory"]["readOnly"], result["memory"]["l1"],
                             "Read Only", "L1",
                             benchmark::nvidia::measureReadOnlyAndL1Shared);
            });
        }
        if (opts.runTexture && opts.runL1) {
            timed("nvidia_textureSharedWithL1", [&] {
                util::sharedHelper(result["memory"]["texture"], result["memory"]["l1"],
                             "Texture", "L1",
                             benchmark::nvidia::measureTextureAndL1Shared);
            });
        }
        if (opts.runTexture && opts.runReadOnly) {
            timed("nvidia_textureSharedWithReadOnly", [&] {
                util::sharedHelper(result["memory"]["texture"], result["memory"]["readOnly"],
                             "Texture", "Read Only",
                             benchmark::nvidia::measureTextureAndReadOnlyShared);
            });
        }

        std::cout << "[Resource Sharing] Benchmarks finished" << std::endl;
    }

    // Stop the total benchmark timer before post-processing.
    double totalSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - totalStart).count();

    if (silencer) silencer.reset();

    // Print the timing summary after restoring stdout so it is visible even with --quiet.
    if (opts.timing) {
        double sumSeconds = 0.0;
        std::cout << "\n===== Timing Summary =====" << std::endl;
        for (const auto& [name, seconds] : timed.timings) {
            std::cout << "  " << name << ": " << seconds << " s" << std::endl;
            sumSeconds += seconds;
        }
        std::cout << "  Sum of individual benchmarks: " << sumSeconds << " s" << std::endl;
        std::cout << "  Total execution time: " << totalSeconds << " s"
                  << " (" << totalSeconds / 60.0 << " min)" << std::endl;
    }

    // Generate bandwidth figures (block-sweep grids + LDS
    // best-per-configuration) from the grid CSVs written above. Done once here so
    // a single generic call covers every bandwidth benchmark, and so the figures
    // exist before the Markdown report embeds them. Only meaningful when an
    // optimal search produced sweep grids.
    if (opts.graphs && opts.runOptimalSearch) {
        std::cout << "[Graphs] Generating bandwidth plots" << std::endl;
        util::generateBandwidthCharts(graphDir.string());
    }

    if (opts.fullReport) {
        util::writeMarkdownReport(graphDir, fancyName, result);
    }

    if (opts.useStdout) {
        std::cout << result.dump(4) << std::endl;
    } else {
        std::ofstream jsonFile(opts.location / (fancyFileName + ".json"));
        if (!jsonFile) {
            std::cerr << "Could not write JSON file '" << fancyFileName << ".json'" << std::endl;
            return EXIT_FAILURE;
        }

        jsonFile << result.dump(4) << std::endl;
    }

    return EXIT_SUCCESS;
}
