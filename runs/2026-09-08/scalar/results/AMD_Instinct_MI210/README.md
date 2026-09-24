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
| globalL1CacheSupported | true |
| localL1CacheSupported | true |

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

### scalarL1

| Key | Value |
| --- | ----- |
| fetchGranularity | 64 bytes |
| latency | 48 cycles |
| lineSize | 64 bytes |
| missPenalty | 132.078 cycles |
| readBandwidthPerCU | cycles: 1222820, dataBytes: 8960, measuredBandwidth: 23.758798821576356, numBlocks: 1, numReps: 2048, numThreads: 1024, time: 0.0007193058823529412 |
| size | 17920 bytes |
| uniqueAmount | 56 |
| writeBandwidthPerCU | cycles: 3187992, dataBytes: 8960, measuredBandwidth: 9.113176687708124, numBlocks: 1, numReps: 2048, numThreads: 1024, time: 0.001875289411764706 |

| Group 1 | Group 2 | Group 3 | Group 4 | Group 5 | Group 6 | Group 7 | Group 8 | Group 9 | Group 10 | Group 11 | Group 12 | Group 13 | Group 14 | Group 15 | Group 16 | Group 17 | Group 18 | Group 19 | Group 20 | Group 21 | Group 22 | Group 23 | Group 24 | Group 25 | Group 26 | Group 27 | Group 28 | Group 29 | Group 30 | Group 31 | Group 32 | Group 33 | Group 34 | Group 35 | Group 36 | Group 37 | Group 38 | Group 39 | Group 40 | Group 41 | Group 42 | Group 43 | Group 44 | Group 45 | Group 46 | Group 47 | Group 48 | Group 49 | Group 50 | Group 51 | Group 52 | Group 53 | Group 54 | Group 55 | Group 56 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0, 1 | 3 | 4, 5 | 6, 7 | 8, 9 | 10, 11 | 12, 13 | 16, 17 | 18, 19 | 20, 21 | 22, 23 | 24, 25 | 26, 27 | 28 | 32, 33 | 34, 35 | 36, 37 | 38 | 40, 41 | 42, 43 | 44, 45 | 48, 49 | 50, 51 | 52, 53 | 54, 55 | 56, 57 | 58, 59 | 60 | 64, 65 | 66, 67 | 68, 69 | 70, 71 | 72, 73 | 74, 75 | 76 | 80, 81 | 82, 83 | 84, 85 | 86, 87 | 88, 89 | 90, 91 | 92 | 96, 97 | 98, 99 | 100, 101 | 102, 103 | 104, 105 | 106, 107 | 108 | 112, 113 | 114, 115 | 116, 117 | 118, 119 | 120, 121 | 122, 123 | 124 |

### shared

| Key | Value |
| --- | ----- |
| reservedSharedMemPerBlock | 0 bytes |
| sharedMemPerBlock | 65536 bytes |
| sharedMemPerMultiProcessor | 6815744 bytes |

## Graphs

![AMD_Instinct_MI210__Scalar_L1_Fetch_Granularity](./AMD_Instinct_MI210__Scalar_L1_Fetch_Granularity.png)
[Raw data](./AMD_Instinct_MI210__Scalar_L1_Fetch_Granularity.txt)

![AMD_Instinct_MI210__Scalar_L1_Line_Size](./AMD_Instinct_MI210__Scalar_L1_Line_Size.png)
[Raw data](./AMD_Instinct_MI210__Scalar_L1_Line_Size.txt)

![AMD_Instinct_MI210__Scalar_L1_Size](./AMD_Instinct_MI210__Scalar_L1_Size.png)
[Raw data](./AMD_Instinct_MI210__Scalar_L1_Size.txt)

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
            "globalL1CacheSupported": true,
            "localL1CacheSupported": true
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
        "scalarL1": {
            "fetchGranularity": {
                "confidence": 0.999840003199936,
                "method": "p-chase",
                "randomized": false,
                "size": 64,
                "unit": "bytes"
            },
            "latency": {
                "mean": 48.0,
                "measurements": 255,
                "method": "p-chase",
                "p50": 48.0,
                "p95": 48.0,
                "sampleSize": 256,
                "stdev": 0.0,
                "unit": "cycles"
            },
            "lineSize": {
                "confidence": 0.9990623273858028,
                "method": "p-chase",
                "randomized": false,
                "size": 64,
                "unit": "bytes"
            },
            "missPenalty": {
                "unit": "cycles",
                "value": 132.07843137254903
            },
            "readBandwidthPerCU": {
                "cycles": 1222820,
                "dataBytes": 8960,
                "measuredBandwidth": 23.758798821576356,
                "numBlocks": 1,
                "numReps": 2048,
                "numThreads": 1024,
                "time": 0.0007193058823529412
            },
            "sharedBetween": [
                [
                    0,
                    1
                ],
                [
                    3
                ],
                [
                    4,
                    5
                ],
                [
                    6,
                    7
                ],
                [
                    8,
                    9
                ],
                [
                    10,
                    11
                ],
                [
                    12,
                    13
                ],
                [
                    16,
                    17
                ],
                [
                    18,
                    19
                ],
                [
                    20,
                    21
                ],
                [
                    22,
                    23
                ],
                [
                    24,
                    25
                ],
                [
                    26,
                    27
                ],
                [
                    28
                ],
                [
                    32,
                    33
                ],
                [
                    34,
                    35
                ],
                [
                    36,
                    37
                ],
                [
                    38
                ],
                [
                    40,
                    41
                ],
                [
                    42,
                    43
                ],
                [
                    44,
                    45
                ],
                [
                    48,
                    49
                ],
                [
                    50,
                    51
                ],
                [
                    52,
                    53
                ],
                [
                    54,
                    55
                ],
                [
                    56,
                    57
                ],
                [
                    58,
                    59
                ],
                [
                    60
                ],
                [
                    64,
                    65
                ],
                [
                    66,
                    67
                ],
                [
                    68,
                    69
                ],
                [
                    70,
                    71
                ],
                [
                    72,
                    73
                ],
                [
                    74,
                    75
                ],
                [
                    76
                ],
                [
                    80,
                    81
                ],
                [
                    82,
                    83
                ],
                [
                    84,
                    85
                ],
                [
                    86,
                    87
                ],
                [
                    88,
                    89
                ],
                [
                    90,
                    91
                ],
                [
                    92
                ],
                [
                    96,
                    97
                ],
                [
                    98,
                    99
                ],
                [
                    100,
                    101
                ],
                [
                    102,
                    103
                ],
                [
                    104,
                    105
                ],
                [
                    106,
                    107
                ],
                [
                    108
                ],
                [
                    112,
                    113
                ],
                [
                    114,
                    115
                ],
                [
                    116,
                    117
                ],
                [
                    118,
                    119
                ],
                [
                    120,
                    121
                ],
                [
                    122,
                    123
                ],
                [
                    124
                ]
            ],
            "size": {
                "confidence": 0.5088498230035399,
                "method": "p-chase",
                "randomized": false,
                "size": 17920,
                "unit": "bytes"
            },
            "uniqueAmount": 56,
            "writeBandwidthPerCU": {
                "cycles": 3187992,
                "dataBytes": 8960,
                "measuredBandwidth": 9.113176687708124,
                "numBlocks": 1,
                "numReps": 2048,
                "numThreads": 1024,
                "time": 0.001875289411764706
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
        "timestamp": "2026-09-08T21:28:57Z"
    }
}
```
