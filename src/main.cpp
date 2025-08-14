#include <cxxopts.hpp>
#include <nlohmann/json.hpp>
#include <hip/hip_runtime.h>
#include <stdio.h>
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

#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"
#include "utils/silent.hpp"

static constexpr auto MIN_EXPECTED_LINE_SIZE = 4;// Bytes
static constexpr auto VALIDITY_THRESHOLD = 0.5;// Factor

int main(int argc, char* argv[]) {
    CLIOptions opts = util::parseCommandLine(argc, argv);

    std::unique_ptr<util::SilentMode> silencer;
    if (opts.runSilently) {
        silencer = std::make_unique<util::SilentMode>();
    }

    util::hipCheck(hipSetDevice(opts.deviceId));
    auto deviceProperties = util::getDeviceProperties();

    std::string fancyName = deviceProperties.name;

    std::filesystem::path graphDir = "results/" + fancyName;
    if (opts.graphs || opts.rawData || opts.fullReport) {
        std::error_code ec;
        std::filesystem::create_directories(graphDir, ec);
        if (ec) {
            std::cerr << "Could not create graph directory '" << graphDir.string() << "': " << ec.message() << std::endl;
        }
    }

    nlohmann::json metaInfo = {
        {"timestamp", util::getCurrentTimestamp()},
        {"hostCompiler", util::getHostCompilerVersion()}
    };
    if (auto gpuCompiler = util::getGpuCompilerVersion()) metaInfo["gpuCompiler"] = *gpuCompiler;
    if (auto cpu = util::getHostCpuModel()) metaInfo["hostCpu"] = *cpu;
    if (auto os = util::getOsDescription()) metaInfo["os"] = *os;
    if (auto driver = util::getDriverVersion()) metaInfo["driver"] = *driver;
    if (auto runtimeVersion = util::getRuntimeVersion()) metaInfo["runtime"] = *runtimeVersion;

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

    if (opts.runL1) {
        std::cout << "[L1] Starting Benchmarks" << std::endl;
        std::cout << "[L1] Latency" << std::endl;
        CacheLatencyResult l1Latency = benchmark::measureL1Latency();
        result["memory"]["l1"]["latency"] = l1Latency;

        std::cout << "[L1] Fetch Granularity" << std::endl;
        CacheSizeResult l1FetchGranularity = benchmark::measureL1FetchGranularity();
        result["memory"]["l1"]["fetchGranularity"] = l1FetchGranularity;

        std::cout << "[L1] Size" << std::endl;
        CacheSizeResult l1Size = benchmark::measureL1Size(l1FetchGranularity.confidence > VALIDITY_THRESHOLD ? l1FetchGranularity.size : MIN_EXPECTED_LINE_SIZE);
        result["memory"]["l1"]["size"] = l1Size;

        if (l1FetchGranularity.confidence > VALIDITY_THRESHOLD && l1Size.confidence > VALIDITY_THRESHOLD) {
            std::cout << "[L1] Line Size" << std::endl;
            CacheLineSizeResult l1LineSize = benchmark::measureL1LineSize(l1Size.size, l1FetchGranularity.size);
            result["memory"]["l1"]["lineSize"] = l1LineSize;
            if (opts.graphs) {
                util::exportChartsReduced(l1LineSize.timings, util::average<uint32_t>, fancyName + " - L1 Line Size", {}, "Bytes", "Cycles", graphDir.string());
            }
            if (opts.rawData) {
                util::writeNestedMapToFile(l1LineSize.timings, (graphDir / (fancyName + " - L1 Line Size.txt")).string());
            }

            if (l1LineSize.confidence > VALIDITY_THRESHOLD) {
                std::cout << "[L1] Miss Penalty" << std::endl;
                double l1MissPenalty = benchmark::measureL1MissPenalty(l1Size.size, l1LineSize.size, l1Latency.mean);
                result["memory"]["l1"]["missPenalty"] = {
                    {"value", l1MissPenalty},
                    {"unit", "cycles"}
                };

                std::cout << "[L1] Amount" << std::endl;
                auto l1Amount = benchmark::measureL1Amount(l1Size.size, l1FetchGranularity.size, l1MissPenalty);
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
        
        if (opts.graphs) {
            util::exportChartMinMaxAvgRed(l1Size.timings, fancyName + " - L1 Size", {l1Size.size}, "Bytes", "Cycles", graphDir.string());
            util::exportChartsMinMaxAvg(l1FetchGranularity.timings, fancyName + " - L1 Fetch Granularity", {l1FetchGranularity.size}, "Bytes", "Cycles", graphDir.string());
        }
        if (opts.rawData) {
            util::writeVectorToFile(l1Latency.timings, (graphDir / (fancyName + " - L1 Latency.txt")).string());
            util::writeMapToFile(l1FetchGranularity.timings, (graphDir / (fancyName + " - L1 Fetch Granularity.txt")).string());
            util::writeMapToFile(l1Size.timings, (graphDir / (fancyName + " - L1 Size.txt")).string());
        }

        std::cout << "[L1] Benchmarks finished" << std::endl;
    }

    if (opts.runL2) {
        std::cout << "[L2] Starting Benchmarks" << std::endl;
        std::cout << "[L2] Latency" << std::endl;
        CacheLatencyResult l2Latency = benchmark::measureL2Latency();
        result["memory"]["l2"]["latency"] = l2Latency;

        std::cout << "[L2] Fetch Granularity" << std::endl;
        CacheSizeResult l2FetchGranularity = benchmark::measureL2FetchGranularity();
        result["memory"]["l2"]["fetchGranularity"] = l2FetchGranularity;

        if (util::isAMD()) {
            std::cout << "L2 Segment Size is currently broken on AMD. Skipping." << std::endl;
        } else {
            std::cout << "[L2] Segment Size" << std::endl;
            CacheSizeResult l2SegmentSize = benchmark::measureL2SegmentSize(deviceProperties.l2CacheSize, l2FetchGranularity.confidence > VALIDITY_THRESHOLD ? l2FetchGranularity.size : MIN_EXPECTED_LINE_SIZE);
            result["memory"]["l2"]["segmentSize"] = l2SegmentSize;
            if (opts.graphs) {
                util::exportChartMinMaxAvgRed(l2SegmentSize.timings, fancyName + " - L2 Segment Size", {l2SegmentSize.size}, "Bytes", "Cycles", graphDir.string());
            }
            if (opts.rawData) {
                util::writeMapToFile(l2SegmentSize.timings, (graphDir / (fancyName + " - L2 Segment Size.txt")).string());
            }
        }

        
        auto l2LineSizeValue = util::getNumeric<size_t>(result, "memory", "l2", "lineSize", "value");
        if (!l2LineSizeValue.has_value()) {
            if (l2FetchGranularity.confidence > VALIDITY_THRESHOLD) {
                std::cout << "[L2] Line Size" << std::endl;
                CacheLineSizeResult l2LineSize = benchmark::measureL2LineSize(deviceProperties.l2CacheSize, l2FetchGranularity.size); // Unreliable on AMD because L2 Size Benchmarks are complicated
                result["memory"]["l2"]["lineSize"] = l2LineSize;
                if (opts.graphs) {
                    util::exportChartsReduced(l2LineSize.timings, util::average<uint32_t>, fancyName + " - L2 Line Size", {}, "Bytes", "Cycles", graphDir.string());
                }
                if (opts.rawData) {
                    util::writeNestedMapToFile(l2LineSize.timings, (graphDir / (fancyName + " - L2 Line Size.txt")).string());
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
            double l2MissPenalty = benchmark::measureL2MissPenalty(deviceProperties.l2CacheSize, l2LineSizeValue.value(), l2Latency.mean);
            result["memory"]["l2"]["missPenalty"] = {
                {"value", l2MissPenalty},
                {"unit", "cycles"}
            };
        } else {
            std::cout << "Could not gather valid L2 Line Size, skipping L2 Miss Penalty benchmarks." << std::endl;
        }

        std::cout << "[L2] Read Bandwidth" << std::endl;
        result["memory"]["l2"]["readBandwidth"] = {
            {"value", benchmark::measureL2ReadBandwidth(deviceProperties.l2CacheSize)},
            {"unit", "GiB/s"}
        };
        std::cout << "[L2] Write Bandwidth" << std::endl;
        result["memory"]["l2"]["writeBandwidth"] = {
            {"value", benchmark::measureL2WriteBandwidth(deviceProperties.l2CacheSize)},
            {"unit", "GiB/s"}
        };

        if (opts.graphs) {
            util::exportChartsMinMaxAvg(l2FetchGranularity.timings, fancyName + " - L2 Fetch Granularity", {l2FetchGranularity.size}, "Bytes", "Cycles", graphDir.string());
        }
        if (opts.rawData) {
            util::writeVectorToFile(l2Latency.timings, (graphDir / (fancyName + " - L2 Latency.txt")).string());
            util::writeMapToFile(l2FetchGranularity.timings, (graphDir / (fancyName + " - L2 Fetch Granularity.txt")).string());
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

            /* Not working yet
            std::cout << "[L3] Latency" << std::endl;
            CacheLatencyResult l3Latency = benchmark::amd::measureL3Latency(32 * KiB, 128);
            result["memory"]["l3"]["latency"] = l3Latency;

            std::cout << "[L3] Fetch Granularity" << std::endl;
            CacheSizeResult l3FetchGranularity = benchmark::amd::measureL3FetchGranularity();
            result["memory"]["l3"]["fetchGranularity"] = l3FetchGranularity;
            */
            auto l3LineSize = util::getNumeric<size_t>(result, "memory", "l3", "lineSize", "value");
            if (l3LineSize.has_value()) {
                /* Not working yet because of Latency dep.
                std::cout << "[L3] Miss Penalty" << std::endl;
                result["memory"]["l3"]["missPenalty"] = benchmark::amd::measureL3MissPenalty(l3Size.value(), l3LineSize.value(), l3Latency.mean);
                */
            } else {
                std::cout << "Could not determine L3 Line Size, L3 Line Size will not be part of the output + skipping Miss Penalty benchmarks." << std::endl;
            }

            std::cout << "[L3] Read Bandwidth" << std::endl;
            result["memory"]["l3"]["readBandwidth"] = {
                {"value", benchmark::amd::measureL3ReadBandwidth(l3Size.value())},
                {"unit", "GiB/s"}
            };

            std::cout << "[L3] Write Bandwidth" << std::endl;
            result["memory"]["l3"]["writeBandwidth"] = {
                {"value", benchmark::amd::measureL3WriteBandwidth(l3Size.value())},
                {"unit", "GiB/s"}
            };

            /* Not working yet
            if (opts.graphs) {
            util::exportChartsMinMaxAvg(l3FetchGranularity.timings, fancyName + " - L3 Fetch Granularity", {l3FetchGranularity.size}, "Bytes", "Cycles", graphDir.string());
            }
            if (opts.rawData) {
                util::writeVectorToFile(l3Latency.timings, (graphDir / (fancyName + " - L3 Latency.txt")).string());
                util::writeMapToFile(l3FetchGranularity.timings, (graphDir / (fancyName + " - L3 Fetch Granularity.txt")).string());
            }
            */
        } else {
            std::cout << "[L3] Could not determine L3 Cache Size, probably because this GPU does not have an L3, skipping benchmarks." << std::endl;
        }
        std::cout << "[L3] Benchmarks finished" << std::endl;
    }

    if (opts.runConstant) {
        std::cout << "[Constant] Starting Benchmarks" << std::endl;
        std::cout << "[Constant] L1 Latency" << std::endl;
        CacheLatencyResult constantL1Latency = benchmark::nvidia::measureConstantL1Latency();
        result["memory"]["constant"]["l1"]["latency"] = constantL1Latency;

        std::cout << "[Constant] L1 Fetch Granularity" << std::endl;
        CacheSizeResult constantL1FetchGranularity = benchmark::nvidia::measureConstantL1FetchGranularity();
        result["memory"]["constant"]["l1"]["fetchGranularity"] = constantL1FetchGranularity;

        std::cout << "[Constant] L1.5 Fetch Granularity" << std::endl;
        CacheSizeResult constantL15FetchGranularity = benchmark::nvidia::measureConstantL15FetchGranularity(constantL1FetchGranularity.size);
        result["memory"]["constant"]["l1.5"]["fetchGranularity"] = constantL15FetchGranularity;

        CacheSizeResult constantL1Size = benchmark::nvidia::measureConstantL1Size(constantL1FetchGranularity.size);
        CacheSizeResult constantL15Size = benchmark::nvidia::measureConstantL15Size(constantL15FetchGranularity.size);
        result["memory"]["constant"]["l1"]["size"] = constantL1Size;
        result["memory"]["constant"]["l1.5"]["size"] = constantL15Size;

        if (constantL1Size.confidence > VALIDITY_THRESHOLD && constantL1FetchGranularity.confidence > VALIDITY_THRESHOLD) {
            std::cout << "[Constant] L1 Line Size" << std::endl;
            CacheLineSizeResult constantL1LineSize = benchmark::nvidia::measureConstantL1LineSize(constantL1Size.size, constantL1FetchGranularity.size);
            result["memory"]["constant"]["l1"]["lineSize"] = constantL1LineSize;
            if (opts.graphs) {
                util::exportChartsReduced(constantL1LineSize.timings, util::average<uint32_t>, fancyName + " - Constant L1 Line Size", {}, "Bytes", "Cycles", graphDir.string());
            }
            if (opts.rawData) {
                util::writeNestedMapToFile(constantL1LineSize.timings, (graphDir / (fancyName + " - Constant L1 Line Size.txt")).string());
            }

            if (constantL1LineSize.confidence > VALIDITY_THRESHOLD) {
                std::cout << "[Constant] L1 Miss Penalty" << std::endl;
                double constantL1MissPenalty = benchmark::nvidia::measureConstantL1MissPenalty(constantL1Size.size, constantL1LineSize.size, constantL1Latency.mean);
                result["memory"]["constant"]["l1"]["missPenalty"] = {
                    {"value", constantL1MissPenalty},
                    {"unit", "cycles"}
                };
                
                
                std::cout << "[Constant] L1 Amount" << std::endl;
                auto constantL1Amount = benchmark::nvidia::measureConstantL1Amount(constantL1Size.size, constantL1FetchGranularity.size, constantL1MissPenalty);
                if (constantL1Amount.has_value()) {
                    result["memory"]["constant"]["l1"]["amountPerMultiprocessor"] = *constantL1Amount;
                } else {
                    std::cout << "Could not measure valid Constant L1 Amount, skipping Constant L1 Amount benchmark." << std::endl;
                }
            } else {
                std::cout << "Could not measure valid Constant L1 Line Size, skipping Constant L1 Miss Penalty and Amount benchmarks." << std::endl;
            }

            std::cout << "[Constant] L1.5 Latency" << std::endl;
            CacheLatencyResult constantL15Latency = benchmark::nvidia::measureConstantL15Latency(8 * KiB, constantL1FetchGranularity.size);
            result["memory"]["constant"]["l1.5"]["latency"] = constantL15Latency;
            if (opts.rawData) {
                util::writeVectorToFile(constantL15Latency.timings, (graphDir / (fancyName + " - Constant L1.5 Latency.txt")).string());
            }
        } else {
            std::cout << "Could not measure valid Constant L1 Size or Fetch Granularity, skipping Constant L1 Amount, Line Size, Miss Penalty and Constant L1.5 Latency benchmarks." << std::endl;
        }

        if (constantL15Size.confidence > VALIDITY_THRESHOLD && constantL1FetchGranularity.confidence > VALIDITY_THRESHOLD) {
            std::cout << "[Constant] L1.5 Line Size" << std::endl;
            CacheLineSizeResult constantL15LineSize = benchmark::nvidia::measureConstantL15LineSize(constantL15Size.size, constantL15FetchGranularity.size);
            result["memory"]["constant"]["l1.5"]["lineSize"] = constantL15LineSize;
            if (opts.graphs) {
                util::exportChartsReduced(constantL15LineSize.timings, util::average<uint32_t>, fancyName + " - Constant L1.5 Line Size", {}, "Bytes", "Cycles", graphDir.string());
            }
            if (opts.rawData) {
                util::writeNestedMapToFile(constantL15LineSize.timings, (graphDir / (fancyName + " - Constant L1.5 Line Size.txt")).string());
            }
        } else {
            std::cerr << "Could not measure valid Constant L1.5 Size or Fetch Granularity, skipping Constant L1.5 Line Size benchmarks." << std::endl;
        }
        if (opts.graphs) {
            util::exportChartMinMaxAvgRed(constantL1Size.timings, fancyName + " - Constant L1 Size", {constantL1Size.size}, "Bytes", "Cycles", graphDir.string());
            util::exportChartsMinMaxAvg(constantL1FetchGranularity.timings, fancyName + " - Constant L1 Fetch Granularity", {constantL1FetchGranularity.size}, "Bytes", "Cycles", graphDir.string());
            util::exportChartMinMaxAvgRed(constantL15Size.timings, fancyName + " - Constant L1.5 Size", {constantL15Size.size}, "Bytes", "Cycles", graphDir.string());
            util::exportChartsMinMaxAvg(constantL15FetchGranularity.timings, fancyName + " - Constant L1.5 Fetch Granularity", {constantL15FetchGranularity.size}, "Bytes", "Cycles", graphDir.string());
        }
        if (opts.rawData) {
            util::writeVectorToFile(constantL1Latency.timings, (graphDir / (fancyName + " - Constant L1 Latency.txt")).string());
            util::writeMapToFile(constantL1FetchGranularity.timings, (graphDir / (fancyName + " - Constant L1 Fetch Granularity.txt")).string());
            util::writeMapToFile(constantL1Size.timings, (graphDir / (fancyName + " - Constant L1 Size.txt")).string());
            util::writeMapToFile(constantL15FetchGranularity.timings, (graphDir / (fancyName + " - Constant L1.5 Fetch Granularity.txt")).string());
            util::writeMapToFile(constantL15Size.timings, (graphDir / (fancyName + " - Constant L1.5 Size.txt")).string());
        }
        std::cout << "[Constant] Benchmarks finished" << std::endl;
    }

    if (opts.runReadOnly) {
        std::cout << "[Read Only] Starting Benchmarks" << std::endl;
        std::cout << "[Read Only] Latency" << std::endl;
        CacheLatencyResult readOnlyLatency = benchmark::nvidia::measureReadOnlyLatency();
        result["memory"]["readOnly"]["latency"] = readOnlyLatency;

        std::cout << "[Read Only] Fetch Granularity" << std::endl;
        CacheSizeResult readOnlyFetchGranularity = benchmark::nvidia::measureReadOnlyFetchGranularity();
        result["memory"]["readOnly"]["fetchGranularity"] = readOnlyFetchGranularity;

        std::cout << "[Read Only] Size" << std::endl;
        CacheSizeResult readOnlySize = benchmark::nvidia::measureReadOnlySize(readOnlyFetchGranularity.confidence > VALIDITY_THRESHOLD ? readOnlyFetchGranularity.size : MIN_EXPECTED_LINE_SIZE);
        result["memory"]["readOnly"]["size"] = readOnlySize;

        if (readOnlySize.confidence > VALIDITY_THRESHOLD && readOnlyFetchGranularity.confidence > VALIDITY_THRESHOLD) {
            std::cout << "[Read Only] Line Size" << std::endl;
            CacheLineSizeResult readOnlyLineSize = benchmark::nvidia::measureReadOnlyLineSize(readOnlySize.size, readOnlyFetchGranularity.size);
            result["memory"]["readOnly"]["lineSize"] = readOnlyLineSize;
            if (opts.graphs) {
                util::exportChartsReduced(readOnlyLineSize.timings, util::average<uint32_t>, fancyName + " - Read Only Line Size", {}, "Bytes", "Cycles", graphDir.string());
            }
            if (opts.rawData) {
                util::writeNestedMapToFile(readOnlyLineSize.timings, (graphDir / (fancyName + " - Read Only Line Size.txt")).string());
            }

            if (readOnlyLineSize.confidence > VALIDITY_THRESHOLD) {
                std::cout << "[Read Only] Miss Penalty" << std::endl;
                double readOnlyMissPenalty = benchmark::nvidia::measureReadOnlyMissPenalty(readOnlySize.size, readOnlyLineSize.size, readOnlyLatency.mean);
                result["memory"]["readOnly"]["missPenalty"] = {
                    {"value", readOnlyMissPenalty},
                    {"unit", "cycles"}
                };

                std::cout << "[Read Only] Amount" << std::endl;
                auto readOnlyAmount = benchmark::nvidia::measureReadOnlyAmount(readOnlySize.size, readOnlyFetchGranularity.size, readOnlyMissPenalty);
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

        if (opts.graphs) {
            util::exportChartMinMaxAvgRed(readOnlySize.timings, fancyName + " - Read Only Size", {readOnlySize.size}, "Bytes", "Cycles", graphDir.string());
            util::exportChartsMinMaxAvg(readOnlyFetchGranularity.timings, fancyName + " - Read Only Fetch Granularity", {readOnlyFetchGranularity.size}, "Bytes", "Cycles", graphDir.string());
        }
        if (opts.rawData) {
            util::writeVectorToFile(readOnlyLatency.timings, (graphDir / (fancyName + " - Read Only Latency.txt")).string());
            util::writeMapToFile(readOnlyFetchGranularity.timings, (graphDir / (fancyName + " - Read Only Fetch Granularity.txt")).string());
            util::writeMapToFile(readOnlySize.timings, (graphDir / (fancyName + " - Read Only Size.txt")).string());
        }
        std::cout << "[Read Only] Benchmarks finished" << std::endl;
    }

    if (opts.runTexture) {
        std::cout << "[Texture] Starting Benchmarks" << std::endl;
        std::cout << "[Texture] Latency" << std::endl;
        CacheLatencyResult textureLatency = benchmark::nvidia::measureTextureLatency();
        result["memory"]["texture"]["latency"] = textureLatency;

        std::cout << "[Texture] Fetch Granularity" << std::endl;
        CacheSizeResult textureFetchGranularity = benchmark::nvidia::measureTextureFetchGranularity();
        result["memory"]["texture"]["fetchGranularity"] = textureFetchGranularity;

        std::cout << "[Texture] Size" << std::endl;
        CacheSizeResult textureSize = benchmark::nvidia::measureTextureSize(textureFetchGranularity.confidence > VALIDITY_THRESHOLD ? textureFetchGranularity.size : MIN_EXPECTED_LINE_SIZE);
        result["memory"]["texture"]["size"] = textureSize;

        if (textureSize.confidence > VALIDITY_THRESHOLD && textureFetchGranularity.confidence > VALIDITY_THRESHOLD) {
            std::cout << "[Texture] Line Size" << std::endl;
            CacheLineSizeResult textureLineSize = benchmark::nvidia::measureTextureLineSize(textureSize.size, textureFetchGranularity.size);
            result["memory"]["texture"]["lineSize"] = textureLineSize;
            if (opts.graphs) {
                util::exportChartsReduced(textureLineSize.timings, util::average<uint32_t>, fancyName + " - Texture Line Size", {}, "Bytes", "Cycles", graphDir.string());
            }
            if (opts.rawData) {
                util::writeNestedMapToFile(textureLineSize.timings, (graphDir / (fancyName + " - Texture Line Size.txt")).string());
            }

            if (textureLineSize.confidence > VALIDITY_THRESHOLD) {
                std::cout << "[Texture] Miss Penalty" << std::endl;
                double textureMissPenalty = benchmark::nvidia::measureTextureMissPenalty(textureSize.size, textureLineSize.size, textureLatency.mean);
                result["memory"]["texture"]["missPenalty"] = {
                    {"value", textureMissPenalty},
                    {"unit", "cycles"}
                };
                
                std::cout << "[Texture] Amount" << std::endl;
                auto textureAmount = benchmark::nvidia::measureTextureAmount(textureSize.size, textureFetchGranularity.size, textureMissPenalty);
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

        if (opts.graphs) {
            util::exportChartMinMaxAvgRed(textureSize.timings, fancyName + " - Texture Size", {textureSize.size}, "Bytes", "Cycles", graphDir.string());
            util::exportChartsMinMaxAvg(textureFetchGranularity.timings, fancyName + " - Texture Fetch Granularity", {textureFetchGranularity.size}, "Bytes", "Cycles", graphDir.string());
        }
        if (opts.rawData) {
            util::writeVectorToFile(textureLatency.timings, (graphDir / (fancyName + " - Texture Latency.txt")).string());
            util::writeMapToFile(textureFetchGranularity.timings, (graphDir / (fancyName + " - Texture Fetch Granularity.txt")).string());
            util::writeMapToFile(textureSize.timings, (graphDir / (fancyName + " - Texture Size.txt")).string());
        }
        std::cout << "[Texture] Benchmarks finished" << std::endl;
    }

    if (opts.runScalar) {
        std::cout << "[Scalar L1] Starting Benchmarks" << std::endl;
        std::cout << "[Scalar L1] Latency" << std::endl;
        CacheLatencyResult scalarL1Latency = benchmark::amd::measureScalarL1Latency();
        result["memory"]["scalarL1"]["latency"] = scalarL1Latency;
        
        std::cout << "[Scalar L1] Fetch Granularity" << std::endl;
        CacheSizeResult scalarL1FetchGranularity = benchmark::amd::measureScalarL1FetchGranularity();
        result["memory"]["scalarL1"]["fetchGranularity"] = scalarL1FetchGranularity;

        std::cout << "[Scalar L1] Size" << std::endl;
        CacheSizeResult scalarL1Size = benchmark::amd::measureScalarL1Size(scalarL1FetchGranularity.confidence > VALIDITY_THRESHOLD ? scalarL1FetchGranularity.size : MIN_EXPECTED_LINE_SIZE);
        result["memory"]["scalarL1"]["size"] = scalarL1Size;


        if (scalarL1Size.confidence > VALIDITY_THRESHOLD && scalarL1FetchGranularity.confidence > VALIDITY_THRESHOLD) {
            std::cout << "[Scalar L1] Line Size" << std::endl;
            CacheLineSizeResult scalarL1LineSize = benchmark::amd::measureScalarL1LineSize(scalarL1Size.size, scalarL1FetchGranularity.size);
            result["memory"]["scalarL1"]["lineSize"] = scalarL1LineSize;
            if (opts.graphs) {
                util::exportChartsReduced(scalarL1LineSize.timings, util::average<uint32_t>, fancyName + " - Scalar L1 Line Size", {}, "Bytes", "Cycles", graphDir.string());
            }
            if (opts.rawData) {
                util::writeNestedMapToFile(scalarL1LineSize.timings, (graphDir / (fancyName + " - Scalar L1 Line Size.txt")).string());
            }

            if (scalarL1LineSize.confidence > VALIDITY_THRESHOLD) {
                std::cout << "[Scalar L1] Miss Penalty" << std::endl;
                double scalarL1MissPenalty = benchmark::amd::measureScalarL1MissPenalty(scalarL1Size.size, scalarL1LineSize.size, scalarL1Latency.mean);
                result["memory"]["scalarL1"]["missPenalty"] = {
                    {"value", scalarL1MissPenalty},
                    {"unit", "cycles"}
                };
            } else {
                std::cout << "Could not measure valid Scalar L1 Line Size, skipping Scalar L1 Miss Penalty benchmarks." << std::endl;
            }

            std::cout << "[Scalar L1] CU Sharing" << std::endl;
            auto sharedBetweenCUs = benchmark::amd::measureCuShareScalarL1(scalarL1Size.size, scalarL1FetchGranularity.size);
            result["memory"]["scalarL1"]["sharedBetween"] = sharedBetweenCUs;
            result["memory"]["scalarL1"]["uniqueAmount"] = sharedBetweenCUs.size();
        } else {
            std::cout << "Could not measure valid Scalar L1 Size oder Fetch Granularity, skipping Scalar L1 Line Size, Miss Penalty and CU Sharing benchmarks." << std::endl;
        }

        if (opts.graphs) {
            util::exportChartMinMaxAvgRed(scalarL1Size.timings, fancyName + " - Scalar L1 Size", {scalarL1Size.size}, "Bytes", "Cycles", graphDir.string());
            util::exportChartsMinMaxAvg(scalarL1FetchGranularity.timings, fancyName + " - Scalar L1 Fetch Granularity", {scalarL1FetchGranularity.size}, "Bytes", "Cycles", graphDir.string());
        }
        if (opts.rawData) {
            util::writeVectorToFile(scalarL1Latency.timings, (graphDir / (fancyName + " - Scalar L1 Latency.txt")).string());
            util::writeMapToFile(scalarL1FetchGranularity.timings, (graphDir / (fancyName + " - Scalar L1 Fetch Granularity.txt")).string());
            util::writeMapToFile(scalarL1Size.timings, (graphDir / (fancyName + " - Scalar L1 Size.txt")).string());
        }

        std::cout << "[Scalar L1] Benchmarks finished" << std::endl;
    }
    
    if (opts.runSharedMemory) {
        std::cout << "[Shared Memory] Starting Benchmarks" << std::endl;
        std::cout << "[Shared Memory] Latency" << std::endl;
        CacheLatencyResult sharedLatency = benchmark::measureSharedMemoryLatency();
        result["memory"]["shared"]["latency"] = sharedLatency;
        if (opts.rawData) {
            util::writeVectorToFile(sharedLatency.timings, (graphDir / (fancyName + " - Shared Memory Latency.txt")).string());
        }
        std::cout << "[Shared Memory] Benchmarks finished" << std::endl;
    }

    if (opts.runMainMemory) {
        std::cout << "[Main Memory] Starting Benchmarks" << std::endl;

        std::cout << "[Main Memory] Latency" << std::endl;
        CacheLatencyResult mainMemLatency = benchmark::measureMainMemoryLatency();
        result["memory"]["main"]["latency"] = mainMemLatency;
        if (opts.rawData) {
            util::writeVectorToFile(mainMemLatency.timings, (graphDir / (fancyName + " - Main Memory Latency.txt")).string());
        }

        std::cout << "[Main Memory] Read Bandwidth" << std::endl;
        result["memory"]["main"]["readBandwidth"] = {
            {"value", benchmark::measureMainMemoryReadBandwidth(deviceProperties.totalGlobalMem)},
            {"unit", "GiB/s"}
        };

        std::cout << "[Main Memory] Write Bandwidth" << std::endl;
        result["memory"]["main"]["writeBandwidth"] = {
            {"value", benchmark::measureMainMemoryWriteBandwidth(deviceProperties.totalGlobalMem)},
            {"unit", "GiB/s"}
        };

        std::cout << "[Main Memory] Benchmarks finished" << std::endl;
    }
    if (opts.runResourceSharing) {
        std::cout << "[Resource Sharing] Starting Benchmarks" << std::endl;

        if (opts.runConstant && opts.runL1) {
            util::sharedHelper(result["memory"]["constant"]["l1"], result["memory"]["l1"],
                         "Constant L1", "L1",
                         benchmark::nvidia::measureConstantL1AndL1Shared);
        }
        if (opts.runReadOnly && opts.runL1) {
            util::sharedHelper(result["memory"]["readOnly"], result["memory"]["l1"],
                         "Read Only", "L1",
                         benchmark::nvidia::measureReadOnlyAndL1Shared);
        }
        if (opts.runTexture && opts.runL1) {
            util::sharedHelper(result["memory"]["texture"], result["memory"]["l1"],
                         "Texture", "L1",
                         benchmark::nvidia::measureTextureAndL1Shared);
        }
        if (opts.runTexture && opts.runReadOnly) {
            util::sharedHelper(result["memory"]["texture"], result["memory"]["readOnly"],
                         "Texture", "Read Only",
                         benchmark::nvidia::measureTextureAndReadOnlyShared);
        }

        std::cout << "[Resource Sharing] Benchmarks finished" << std::endl;
    }
    if (silencer) silencer.reset();

    if (opts.fullReport) {
        util::writeMarkdownReport(graphDir, fancyName, result);
    }
    if (opts.writeJson) {
        std::ofstream jsonFile(fancyName + ".json");
        if (!jsonFile) {
            std::cerr << "Could not write JSON file '" << fancyName << ".json'" << std::endl;
        } else {
            jsonFile << result.dump(4) << std::endl;
        }
    }

    std::cout << result.dump(4) << std::endl;
    return 0;
}
