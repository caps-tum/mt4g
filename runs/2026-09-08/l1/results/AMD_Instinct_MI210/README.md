# AMD Instinct MI210 Benchmark Report

## General

| Key | Value |
| --- | ----- |
| asicRevision | 1 |
| clockRate | 1700000 kHz |
| computeCapability | 9.0 |
| name | AMD Instinct MI210 |
| vendor | AMD |

## Compute

| Key | Value |
| --- | ----- |
| computeUnitsPerDie | 104 |
| concurrentKernels | true |
| maxBlocksPerMultiProcessor | 32 |
| maxThreadsPerBlock | 1024 |
| maxThreadsPerMultiProcessor | 2048 |
| multiProcessorCount | 104 |
| numSIMDsPerCU | 4 |
| numXCDs | 1 |
| numberOfCoresPerMultiProcessor | 64 |
| regsPerBlock | 131072 |
| regsPerMultiProcessor | 131072 |
| supportsCooperativeLaunch | true |
| warpSize | 64 |

## Memory

### constant

| Key | Value |
| --- | ----- |
| totalConstMem | 2147483647 bytes |

### l1

| Key | Value |
| --- | ----- |
| fetchGranularity | 64 bytes |
| globalL1CacheSupported | true |
| latency | 120 cycles |
| lineSize | 64 bytes |
| localL1CacheSupported | true |
| missPenalty | 188.533 cycles |
| readBandwidthPerCU | cycles: 0, dataBytes: 8192, measuredBandwidth: 69.65495892747774, numBlocks: 2, numReps: 2048, numThreads: 256, time: 0.0002243199944496155 |
| size | 16384 bytes |
| writeBandwidthPerCU | cycles: 0, dataBytes: 8192, measuredBandwidth: 49.172175958447944, numBlocks: 1, numReps: 2048, numThreads: 512, time: 0.0003177610039710999 |

### l2

| Key | Value |
| --- | ----- |
| amount | 1 |
| lineSize | 128 bytes |
| persistingL2CacheMaxSize | 8388608 bytes |
| size | 8388608 bytes |

### main

| Key | Value |
| --- | ----- |
| memoryBusWidth | 4096 bit |
| memoryClockRate | 1600000 kHz |
| totalGlobalMem | 68702699520 bytes |

### shared

| Key | Value |
| --- | ----- |
| reservedSharedMemPerBlock | 0 bytes |
| sharedMemPerBlock | 65536 bytes |
| sharedMemPerMultiProcessor | 6815744 bytes |

## Graphs

![AMD_Instinct_MI210__L1_Fetch_Granularity](./AMD_Instinct_MI210__L1_Fetch_Granularity.png)
[Raw data](./AMD_Instinct_MI210__L1_Fetch_Granularity.txt)

![AMD_Instinct_MI210__L1_Line_Size](./AMD_Instinct_MI210__L1_Line_Size.png)
[Raw data](./AMD_Instinct_MI210__L1_Line_Size.txt)

![AMD_Instinct_MI210__L1_Size](./AMD_Instinct_MI210__L1_Size.png)
[Raw data](./AMD_Instinct_MI210__L1_Size.txt)

![AMD_Instinct_MI210__vL1d_read_bandwidth](./AMD_Instinct_MI210__vL1d_read_bandwidth.png)

![AMD_Instinct_MI210__vL1d_write_bandwidth](./AMD_Instinct_MI210__vL1d_write_bandwidth.png)

## Raw JSON

```json
{
    "compute": {
        "computeUnitsPerDie": 104,
        "concurrentKernels": true,
        "maxBlocksPerMultiProcessor": 32,
        "maxThreadsPerBlock": 1024,
        "maxThreadsPerMultiProcessor": 2048,
        "multiProcessorCount": 104,
        "numSIMDsPerCU": 4,
        "numXCDs": 1,
        "numberOfCoresPerMultiProcessor": 64,
        "regsPerBlock": 131072,
        "regsPerMultiProcessor": 131072,
        "supportsCooperativeLaunch": true,
        "warpSize": 64
    },
    "general": {
        "asicRevision": 1,
        "clockRate": {
            "unit": "kHz",
            "value": 1700000
        },
        "computeCapability": {
            "major": 9,
            "minor": 0
        },
        "name": "AMD Instinct MI210",
        "vendor": "AMD"
    },
    "memory": {
        "constant": {
            "totalConstMem": {
                "unit": "bytes",
                "value": 2147483647
            }
        },
        "l1": {
            "fetchGranularity": {
                "confidence": 0.9675406491870162,
                "method": "p-chase",
                "randomized": false,
                "size": 64,
                "unit": "bytes"
            },
            "globalL1CacheSupported": true,
            "latency": {
                "mean": 120.0,
                "measurements": 255,
                "method": "p-chase",
                "p50": 120.0,
                "p95": 120.0,
                "sampleSize": 256,
                "stdev": 0.0,
                "unit": "cycles"
            },
            "lineSize": {
                "confidence": 0.9980053569661067,
                "method": "p-chase",
                "randomized": false,
                "size": 64,
                "unit": "bytes"
            },
            "localL1CacheSupported": true,
            "missPenalty": {
                "unit": "cycles",
                "value": 188.53333333333336
            },
            "readBandwidthPerCU": {
                "cycles": 0,
                "dataBytes": 8192,
                "measuredBandwidth": 69.65495892747774,
                "numBlocks": 2,
                "numReps": 2048,
                "numThreads": 256,
                "time": 0.0002243199944496155
            },
            "size": {
                "confidence": 0.9738513074346282,
                "method": "p-chase",
                "randomized": false,
                "size": 16384,
                "unit": "bytes"
            },
            "writeBandwidthPerCU": {
                "cycles": 0,
                "dataBytes": 8192,
                "measuredBandwidth": 49.172175958447944,
                "numBlocks": 1,
                "numReps": 2048,
                "numThreads": 512,
                "time": 0.0003177610039710999
            }
        },
        "l2": {
            "amount": 1,
            "lineSize": {
                "unit": "bytes",
                "value": 128
            },
            "persistingL2CacheMaxSize": {
                "unit": "bytes",
                "value": 8388608
            },
            "size": {
                "unit": "bytes",
                "value": 8388608
            }
        },
        "main": {
            "memoryBusWidth": {
                "unit": "bit",
                "value": 4096
            },
            "memoryClockRate": {
                "unit": "kHz",
                "value": 1600000
            },
            "totalGlobalMem": {
                "unit": "bytes",
                "value": 68702699520
            }
        },
        "shared": {
            "reservedSharedMemPerBlock": {
                "unit": "bytes",
                "value": 0
            },
            "sharedMemPerBlock": {
                "unit": "bytes",
                "value": 65536
            },
            "sharedMemPerMultiProcessor": {
                "unit": "bytes",
                "value": 6815744
            }
        }
    },
    "meta": {
        "driver": 70125424,
        "gpuCompiler": "hipcc 7.1.25424",
        "hostCompiler": "clang 20.0.0",
        "hostCpu": "AMD EPYC 7773X 64-Core Processor",
        "hostname": "milan1",
        "mt4gVersion": "5cfff0a",
        "os": "Linux 6.4.0-150700.53.25-default",
        "runtime": 70125424,
        "timestamp": "2026-09-08T21:28:28Z"
    }
}
```
