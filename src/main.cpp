#include <cxxopts.hpp>
#include <nlohmann/json.hpp>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <fstream>
#include <filesystem>
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

// Variadic helper: safely drill into nested keys and extract numeric types.
template <typename T, typename... Keys>
std::optional<T> getNumeric(const nlohmann::json& j, const std::string& key, const Keys&... rest) {
    auto it = j.find(key);
    if (it == j.end()) return std::nullopt;
    if constexpr (sizeof...(rest) == 0) {
        const nlohmann::json& leaf = *it;
        if constexpr (std::is_same_v<T, double>) {
            if (!leaf.is_number()) return std::nullopt;
            return leaf.get<double>();
        } else if constexpr (std::is_same_v<T, size_t>) {
            if (!leaf.is_number_unsigned() && !leaf.is_number_integer()) return std::nullopt;
            return static_cast<size_t>(leaf.get<unsigned long long>());
        } else {
            static_assert(!sizeof(T), "Unsupported type for getNumeric");
        }
    } else {
        if (!it->is_object()) return std::nullopt;
        return getNumeric<T>(*it, rest...); // recurse with remaining keys
    }
}

template <typename MeasureFunc>
void sharedHelper(nlohmann::json& cache1, nlohmann::json& cache2, const std::string& name1, const std::string& name2, MeasureFunc measureFunction) {
    constexpr double validityThreshold = 0.0; // anpassen wie gebraucht

    auto c1SizeConf = getNumeric<double>(cache1, "size", "confidence");
    auto c2SizeConf = getNumeric<double>(cache2, "size", "confidence");
    auto c1FetchConf = getNumeric<double>(cache1, "fetchGranularity", "confidence");
    auto c2FetchConf = getNumeric<double>(cache2, "fetchGranularity", "confidence");

    if (!(c1SizeConf && c2SizeConf && c1FetchConf && c2FetchConf &&
          *c1SizeConf > validityThreshold &&
          *c2SizeConf > validityThreshold &&
          *c1FetchConf > validityThreshold &&
          *c2FetchConf > validityThreshold)) {
        std::cout << "Could not measure valid " << name1 << " or " << name2
                  << " cache or fetch granularities, skipping " << name1
                  << " shared with " << name2 << " benchmark\n";
        return;
    }

    auto c1Size = getNumeric<size_t>(cache1, "size", "size");
    auto c1Fetch = getNumeric<size_t>(cache1, "fetchGranularity", "size");
    auto c1Latency = getNumeric<double>(cache1, "latency", "mean");
    auto c1MissPenalty = getNumeric<double>(cache1, "missPenalty"); // falls verschachtelt, z.B. ("some", "missPenalty") anpassen

    auto c2Size = getNumeric<size_t>(cache2, "size", "size");
    auto c2Fetch = getNumeric<size_t>(cache2, "fetchGranularity", "size");
    auto c2Latency = getNumeric<double>(cache2, "latency", "mean");
    auto c2MissPenalty = getNumeric<double>(cache2, "missPenalty");

    if (!(c1Size && c1Fetch && c1Latency && c1MissPenalty &&
          c2Size && c2Fetch && c2Latency && c2MissPenalty)) {
        std::cout << "Missing metric values for " << name1 << " or " << name2
                  << ", skipping measurement.\n";
        return;
    }

    std::cout << "[Resource Sharing] Running " << name1 << " shared with " << name2 << std::endl;

    bool isShared = measureFunction(*c1Size, *c1Fetch, *c1Latency, *c1MissPenalty, *c2Size, *c2Fetch, *c2Latency, *c2MissPenalty);

    if (isShared) {
        if (!cache1.contains("sharedWith")) cache1["sharedWith"] = nlohmann::json::array();
        if (!cache2.contains("sharedWith")) cache2["sharedWith"] = nlohmann::json::array();
        cache1["sharedWith"].push_back(name2);
        cache2["sharedWith"].push_back(name1);
    }
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

    std::filesystem::path graphDir = fancyName;
    if (opts.graphs || opts.rawData) {
        std::error_code ec;
        std::filesystem::create_directories(graphDir, ec);
        if (ec) {
            std::cerr << "Could not create graph directory '" << graphDir.string() << "': " << ec.message() << std::endl;
        }
    }

    nlohmann::json result = {
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
            CacheSizeResult l1LineSize = benchmark::measureL1LineSize(l1Size.size, l1FetchGranularity.size);
            result["memory"]["l1"]["lineSize"] = l1LineSize;
            if (opts.rawData) {
                util::writeMapToFile(l1LineSize.timings, (graphDir / (fancyName + " - L1 Line Size.txt")).string());
            }

            if (l1LineSize.confidence > VALIDITY_THRESHOLD) {
                std::cout << "[L1] Miss Penalty" << std::endl;
                double l1MissPenalty = result["memory"]["l1"]["missPenalty"] = benchmark::measureL1MissPenalty(l1Size.size, l1LineSize.size, l1Latency.mean);

                std::cout << "[L1] Amount" << std::endl;
                result["memory"]["l1"]["amountPerMultiprocessor"] = benchmark::measureL1Amount(l1Size.size, l1FetchGranularity.size, l1MissPenalty);
            } else {
                std::cout << "Could not measure valid L1 Line Size, skipping L1 Miss Penalty and Amount benchmarks." << std::endl;
            }
        } else {
            std::cout << "Could not measure valid L1 Size or Fetch Granularity, skipping L1 Line Size, Amount and Miss Penalty benchmarks." << std::endl;
        }
        
        if (opts.graphs) {
            util::pipeMapToPython(l1Size.timings, fancyName + " - L1 Size", {l1Size.size}, "Bytes", "Cycles", graphDir.string());
            util::pipeMapToPython(l1FetchGranularity.timings, fancyName + " - L1 Fetch Granularity", {l1FetchGranularity.size}, "Bytes", "Cycles", graphDir.string());
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
                util::pipeMapToPython(l2SegmentSize.timings, fancyName + " - L2 Segment Size", {l2SegmentSize.size}, "Bytes", "Cycles", graphDir.string());
            }
            if (opts.rawData) {
                util::writeMapToFile(l2SegmentSize.timings, (graphDir / (fancyName + " - L2 Segment Size.txt")).string());
            }
        }

        
        if (l2FetchGranularity.confidence > VALIDITY_THRESHOLD) {
            std::cout << "[L2] Line Size" << std::endl;
            CacheSizeResult l2LineSize = benchmark::measureL2LineSize(deviceProperties.l2CacheSize, l2FetchGranularity.size);
            result["memory"]["l2"]["lineSize"] = l2LineSize;
            if (opts.rawData) {
                util::writeMapToFile(l2LineSize.timings, (graphDir / (fancyName + " - L2 Line Size.txt")).string());
            }

            if (l2LineSize.confidence > VALIDITY_THRESHOLD) {
                std::cout << "[L2] Miss Penalty" << std::endl;
                result["memory"]["l2"]["missPenalty"] = benchmark::measureL2MissPenalty(deviceProperties.l2CacheSize, l2LineSize.size, l2Latency.mean);
            } else {
                std::cout << "Could not measure valid L2 Line Size, skipping L2 Miss Penalty and Amount benchmarks." << std::endl;
            }
        } else {
            std::cout << "Could not measure valid L2 Fetch Granularitys, skipping L2 Line Size benchmarks." << std::endl;
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
            util::pipeMapToPython(l2FetchGranularity.timings, fancyName + " - L2 Fetch Granularity", {l2FetchGranularity.size}, "Bytes", "Cycles", graphDir.string());
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
            auto l3LineSize = util::getL3LineSizeBytes();
            if (l3LineSize.has_value()) {
                result["memory"]["l3"]["lineSize"] = {
                    {"value", l3LineSize.value()},
                    {"unit", "bytes"}
                };
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
                util::pipeMapToPython(l3FetchGranularity.timings, fancyName + " - L3 Fetch Granularity", {l3FetchGranularity.size}, "Bytes", "Cycles", graphDir.string());
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
            CacheSizeResult constantL1LineSize = benchmark::nvidia::measureConstantL1LineSize(constantL1Size.size, constantL1FetchGranularity.size);
            result["memory"]["constant"]["l1"]["lineSize"] = constantL1LineSize;
            if (opts.rawData) {
                util::writeMapToFile(constantL1LineSize.timings, (graphDir / (fancyName + " - Constant L1 Line Size.txt")).string());
            }

            if (constantL1LineSize.confidence > VALIDITY_THRESHOLD) {
                std::cout << "[Constant] L1 Miss Penalty" << std::endl;
                double constantL1MissPenalty = result["memory"]["constant"]["l1"]["missPenalty"] = benchmark::nvidia::measureConstantL1MissPenalty(constantL1Size.size, constantL1LineSize.size, constantL1Latency.mean);
                
                
                std::cout << "[Constant] L1 Amount" << std::endl;
                result["memory"]["constant"]["l1"]["amountPerMultiprocessor"] = benchmark::nvidia::measureConstantL1Amount(constantL1Size.size, constantL1FetchGranularity.size, constantL1MissPenalty);
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
            CacheSizeResult constantL15LineSize = benchmark::nvidia::measureConstantL15LineSize(constantL15Size.size, constantL15FetchGranularity.size);
            result["memory"]["constant"]["l1.5"]["lineSize"] = constantL15LineSize;
            if (opts.rawData) {
                util::writeMapToFile(constantL15LineSize.timings, (graphDir / (fancyName + " - Constant L1.5 Line Size.txt")).string());
            }
        } else {
            std::cerr << "Could not measure valid Constant L1.5 Size or Fetch Granularity, skipping Constant L1.5 Line Size benchmarks." << std::endl;
        }
        if (opts.graphs) {
            util::pipeMapToPython(constantL1Size.timings, fancyName + " - Constant L1 Size", {constantL1Size.size}, "Bytes", "Cycles", graphDir.string());
            util::pipeMapToPython(constantL1FetchGranularity.timings, fancyName + " - Constant L1 Fetch Granularity", {constantL1FetchGranularity.size}, "Bytes", "Cycles", graphDir.string());
            util::pipeMapToPython(constantL15Size.timings, fancyName + " - Constant L1.5 Size", {constantL15Size.size}, "Bytes", "Cycles", graphDir.string());
            util::pipeMapToPython(constantL15FetchGranularity.timings, fancyName + " - Constant L1.5 Fetch Granularity", {constantL15FetchGranularity.size}, "Bytes", "Cycles", graphDir.string());
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
            CacheSizeResult readOnlyLineSize = benchmark::nvidia::measureReadOnlyLineSize(readOnlySize.size, readOnlyFetchGranularity.size);
            result["memory"]["readOnly"]["lineSize"] = readOnlyLineSize;
            if (opts.rawData) {
                util::writeMapToFile(readOnlyLineSize.timings, (graphDir / (fancyName + " - Read Only Line Size.txt")).string());
            }

            if (readOnlyLineSize.confidence > VALIDITY_THRESHOLD) {
                std::cout << "[Read Only] Miss Penalty" << std::endl;
                double readOnlyMissPenalty = result["memory"]["readOnly"]["missPenalty"] = benchmark::nvidia::measureReadOnlyMissPenalty(readOnlySize.size, readOnlyLineSize.size, readOnlyLatency.mean);

                std::cout << "[Read Only] Amount" << std::endl;
                result["memory"]["readOnly"]["amountPerMultiprocessor"] = benchmark::nvidia::measureReadOnlyAmount(readOnlySize.size, readOnlyFetchGranularity.size, readOnlyMissPenalty);
            } else {
                std::cout << "Could not measure valid Read Only Line Size, skipping Read Only Miss Penalty and Amount benchmarks." << std::endl;
            }
        } else {
            std::cout << "Could not measure valid Read Only Size or Fetch Granularity, skipping Read Only Amount, Line Size and Miss Penalty benchmarks." << std::endl;
        }

        if (opts.graphs) {
            util::pipeMapToPython(readOnlySize.timings, fancyName + " - Read Only Size", {readOnlySize.size}, "Bytes", "Cycles", graphDir.string());
            util::pipeMapToPython(readOnlyFetchGranularity.timings, fancyName + " - Read Only Fetch Granularity", {readOnlyFetchGranularity.size}, "Bytes", "Cycles", graphDir.string());
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
            CacheSizeResult textureLineSize = benchmark::nvidia::measureTextureLineSize(textureSize.size, textureFetchGranularity.size);
            result["memory"]["texture"]["lineSize"] = textureLineSize;
            if (opts.rawData) {
                util::writeMapToFile(textureLineSize.timings, (graphDir / (fancyName + " - Texture Line Size.txt")).string());
            }

            if (textureLineSize.confidence > VALIDITY_THRESHOLD) {
                std::cout << "[Texture] Miss Penalty" << std::endl;
                double textureMissPenalty = result["memory"]["texture"]["missPenalty"] = benchmark::nvidia::measureTextureMissPenalty(textureSize.size, textureLineSize.size, textureLatency.mean);
                
                std::cout << "[Texture] Amount" << std::endl;
                result["memory"]["texture"]["amountPerMultiprocessor"] = benchmark::nvidia::measureTextureAmount(textureSize.size, textureFetchGranularity.size, textureMissPenalty);
            } else {
                std::cout << "Could not measure valid Texture Line Size, skipping Texture Miss Penalty and Amount benchmarks." << std::endl;
            }
        } else {
            std::cout << "Could not measure valid Texture Size or Fetch Granularity, skipping Texture Amount, Line Size and Miss Penalty benchmarks." << std::endl;
        }

        if (opts.graphs) {
            util::pipeMapToPython(textureSize.timings, fancyName + " - Texture Size", {textureSize.size}, "Bytes", "Cycles", graphDir.string());
            util::pipeMapToPython(textureFetchGranularity.timings, fancyName + " - Texture Fetch Granularity", {textureFetchGranularity.size}, "Bytes", "Cycles", graphDir.string());
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
            CacheSizeResult scalarL1LineSize = benchmark::amd::measureScalarL1LineSize(scalarL1Size.size, scalarL1FetchGranularity.size);
            result["memory"]["scalarL1"]["lineSize"] = scalarL1LineSize;
            if (opts.rawData) {
                util::writeMapToFile(scalarL1LineSize.timings, (graphDir / (fancyName + " - Scalar L1 Line Size.txt")).string());
            }

            if (scalarL1LineSize.confidence > VALIDITY_THRESHOLD) {
                std::cout << "[Scalar L1] Miss Penalty" << std::endl;
                result["memory"]["scalarL1"]["missPenalty"] = benchmark::amd::measureScalarL1MissPenalty(scalarL1Size.size, scalarL1LineSize.size, scalarL1Latency.mean);
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
            util::pipeMapToPython(scalarL1Size.timings, fancyName + " - Scalar L1 Size", {scalarL1Size.size}, "Bytes", "Cycles", graphDir.string());
            util::pipeMapToPython(scalarL1FetchGranularity.timings, fancyName + " - Scalar L1 Fetch Granularity", {scalarL1FetchGranularity.size}, "Bytes", "Cycles", graphDir.string());
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
            sharedHelper(result["memory"]["constant"]["l1"], result["memory"]["l1"],
                         "Constant L1", "L1",
                         benchmark::nvidia::measureConstantL1AndL1Shared);
        }
        if (opts.runReadOnly && opts.runL1) {
            sharedHelper(result["memory"]["readOnly"], result["memory"]["l1"],
                         "Read Only", "L1",
                         benchmark::nvidia::measureReadOnlyAndL1Shared);
        }
        if (opts.runTexture && opts.runL1) {
            sharedHelper(result["memory"]["texture"], result["memory"]["l1"],
                         "Texture", "L1",
                         benchmark::nvidia::measureTextureAndL1Shared);
        }
        if (opts.runTexture && opts.runReadOnly) {
            sharedHelper(result["memory"]["texture"], result["memory"]["readOnly"],
                         "Texture", "Read Only",
                         benchmark::nvidia::measureTextureAndReadOnlyShared);
        }

        std::cout << "[Resource Sharing] Benchmarks finished" << std::endl;
    }
    if (silencer) silencer.reset();
    std::cout << result.dump(4) << std::endl;
    return 0;
}
