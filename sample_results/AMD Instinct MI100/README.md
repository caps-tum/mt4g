# AMD Instinct MI100 Benchmark Report

## General

| Key | Value |
| --- | ----- |
| asicRevision | 2 |
| clockRate | 1502000 kHz |
| computeCapability | 9.0 |
| name | AMD Instinct MI100 |
| vendor | AMD |

## Compute

| Key | Value |
| --- | ----- |
| computeUnitsPerDie | 120 |
| concurrentKernels | true |
| maxBlocksPerMultiProcessor | 2 |
| maxThreadsPerBlock | 1024 |
| maxThreadsPerMultiProcessor | 2560 |
| multiProcessorCount | 120 |
| numSIMDsPerCU | 4 |
| numXCDs | 1 |
| numberOfCoresPerMultiProcessor | 64 |
| regsPerBlock | 65536 |
| regsPerMultiProcessor | 65536 |
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
| amountPerMultiprocessor | 1 |
| fetchGranularity | 64 bytes |
| globalL1CacheSupported | true |
| latency | 136 cycles |
| lineSize | 64 bytes |
| localL1CacheSupported | true |
| missPenalty | 196.439 cycles |
| size | 16384 bytes |

### l2

| Key | Value |
| --- | ----- |
| amount | 1 |
| fetchGranularity | 64 bytes |
| latency | 343.686 cycles |
| lineSize | 64 bytes |
| missPenalty | 370.369 cycles |
| persistingL2CacheMaxSize | 8388608 bytes |
| readBandwidth | 2484.74 GiB/s |
| size | 8388608 bytes |
| writeBandwidth | 2208.98 GiB/s |

### main

| Key | Value |
| --- | ----- |
| latency | 703.166 cycles |
| memoryBusWidth | 4096 bit |
| memoryClockRate | 1200000 kHz |
| readBandwidth | 683.526 GiB/s |
| totalGlobalMem | 34342961152 bytes |
| writeBandwidth | 665.029 GiB/s |

### scalarL1

| Key | Value |
| --- | ----- |
| fetchGranularity | 64 bytes |
| latency | 49.5686 cycles |
| lineSize | 64 bytes |
| missPenalty | 155.515 cycles |
| size | 16128 bytes |
| uniqueAmount | 48 |

| Group 1 | Group 2 | Group 3 | Group 4 | Group 5 | Group 6 | Group 7 | Group 8 | Group 9 | Group 10 | Group 11 | Group 12 | Group 13 | Group 14 | Group 15 | Group 16 | Group 17 | Group 18 | Group 19 | Group 20 | Group 21 | Group 22 | Group 23 | Group 24 | Group 25 | Group 26 | Group 27 | Group 28 | Group 29 | Group 30 | Group 31 | Group 32 | Group 33 | Group 34 | Group 35 | Group 36 | Group 37 | Group 38 | Group 39 | Group 40 | Group 41 | Group 42 | Group 43 | Group 44 | Group 45 | Group 46 | Group 47 | Group 48 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1, 2 | 3, 4, 5 | 6, 7 | 8, 9, 10 | 11, 12, 13 | 14, 15 | 17, 18 | 19, 20, 21 | 22, 23 | 24, 25, 26 | 27, 28, 29 | 30, 31 | 33, 34 | 35, 36, 37 | 38, 39 | 40, 41, 42 | 43, 44, 45 | 46, 47 | 49, 50 | 51, 52, 53 | 54, 55 | 56, 57, 58 | 59, 60, 61 | 62, 63 | 64, 65, 66 | 67, 68 | 69, 70, 71 | 72, 73, 74 | 75, 76 | 78, 79 | 81, 82 | 83, 84, 85 | 86, 87 | 88, 89, 90 | 91, 92, 93 | 94, 95 | 97, 98 | 99, 100, 101 | 102, 103 | 104, 105, 106 | 107, 108, 109 | 110, 111 | 113, 114 | 115, 116, 117 | 118, 119 | 120, 121, 122 | 123, 124, 125 | 126, 127 |

### shared

| Key | Value |
| --- | ----- |
| latency | 49.9765 cycles |
| reservedSharedMemPerBlock | 0 bytes |
| sharedMemPerBlock | 65536 bytes |
| sharedMemPerMultiProcessor | 7864320 bytes |

## Graphs

![AMD Instinct MI100 - L1 Fetch Granularity](./AMD%20Instinct%20MI100%20-%20L1%20Fetch%20Granularity.png)
[Raw data](./AMD%20Instinct%20MI100%20-%20L1%20Fetch%20Granularity.txt)

![AMD Instinct MI100 - L1 Line Size](./AMD%20Instinct%20MI100%20-%20L1%20Line%20Size.png)
[Raw data](./AMD%20Instinct%20MI100%20-%20L1%20Line%20Size.txt)

![AMD Instinct MI100 - L1 Size](./AMD%20Instinct%20MI100%20-%20L1%20Size.png)
[Raw data](./AMD%20Instinct%20MI100%20-%20L1%20Size.txt)

![AMD Instinct MI100 - L2 Fetch Granularity](./AMD%20Instinct%20MI100%20-%20L2%20Fetch%20Granularity.png)
[Raw data](./AMD%20Instinct%20MI100%20-%20L2%20Fetch%20Granularity.txt)

![AMD Instinct MI100 - Scalar L1 Fetch Granularity](./AMD%20Instinct%20MI100%20-%20Scalar%20L1%20Fetch%20Granularity.png)
[Raw data](./AMD%20Instinct%20MI100%20-%20Scalar%20L1%20Fetch%20Granularity.txt)

![AMD Instinct MI100 - Scalar L1 Line Size](./AMD%20Instinct%20MI100%20-%20Scalar%20L1%20Line%20Size.png)
[Raw data](./AMD%20Instinct%20MI100%20-%20Scalar%20L1%20Line%20Size.txt)

![AMD Instinct MI100 - Scalar L1 Size](./AMD%20Instinct%20MI100%20-%20Scalar%20L1%20Size.png)
[Raw data](./AMD%20Instinct%20MI100%20-%20Scalar%20L1%20Size.txt)

## Raw JSON

```json
{
    "compute": {
        "computeUnitsPerDie": 120,
        "concurrentKernels": true,
        "maxBlocksPerMultiProcessor": 2,
        "maxThreadsPerBlock": 1024,
        "maxThreadsPerMultiProcessor": 2560,
        "multiProcessorCount": 120,
        "numSIMDsPerCU": 4,
        "numXCDs": 1,
        "numberOfCoresPerMultiProcessor": 64,
        "regsPerBlock": 65536,
        "regsPerMultiProcessor": 65536,
        "supportsCooperativeLaunch": true,
        "warpSize": 64
    },
    "general": {
        "asicRevision": 2,
        "clockRate": {
            "unit": "kHz",
            "value": 1502000
        },
        "computeCapability": {
            "major": 9,
            "minor": 0
        },
        "name": "AMD Instinct MI100",
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
            "amountPerMultiprocessor": 1,
            "fetchGranularity": {
                "confidence": 0.999060018799624,
                "method": "p-chase",
                "randomized": false,
                "size": 64,
                "unit": "bytes"
            },
            "globalL1CacheSupported": true,
            "latency": {
                "mean": 136.0,
                "measurements": 255,
                "method": "p-chase",
                "p50": 136.0,
                "p95": 136.0,
                "sampleSize": 256,
                "stdev": 0.0,
                "unit": "cycles"
            },
            "lineSize": {
                "confidence": 0.9964183259581963,
                "method": "p-chase",
                "randomized": false,
                "size": 64,
                "unit": "bytes"
            },
            "localL1CacheSupported": true,
            "missPenalty": {
                "unit": "cycles",
                "value": 196.4392156862745
            },
            "size": {
                "confidence": 0.9745012749362532,
                "method": "p-chase",
                "randomized": false,
                "size": 16384,
                "unit": "bytes"
            }
        },
        "l2": {
            "amount": 1,
            "fetchGranularity": {
                "confidence": 0.9393612127757445,
                "method": "p-chase",
                "randomized": false,
                "size": 64,
                "unit": "bytes"
            },
            "latency": {
                "mean": 343.6862745098039,
                "measurements": 255,
                "method": "p-chase",
                "p50": 336.0,
                "p95": 356.0,
                "sampleSize": 256,
                "stdev": 17.159614503955115,
                "unit": "cycles"
            },
            "lineSize": {
                "unit": "bytes",
                "value": 64
            },
            "missPenalty": {
                "unit": "cycles",
                "value": 370.36862745098034
            },
            "persistingL2CacheMaxSize": {
                "unit": "bytes",
                "value": 8388608
            },
            "readBandwidth": {
                "unit": "GiB/s",
                "value": 2484.7356487907145
            },
            "size": {
                "unit": "bytes",
                "value": 8388608
            },
            "writeBandwidth": {
                "unit": "GiB/s",
                "value": 2208.980855030352
            }
        },
        "main": {
            "latency": {
                "mean": 703.1656082071324,
                "measurements": 2047,
                "method": "p-chase",
                "p50": 664.0,
                "p95": 986.7999999999993,
                "sampleSize": 2048,
                "stdev": 138.5689160879028,
                "unit": "cycles"
            },
            "memoryBusWidth": {
                "unit": "bit",
                "value": 4096
            },
            "memoryClockRate": {
                "unit": "kHz",
                "value": 1200000
            },
            "readBandwidth": {
                "unit": "GiB/s",
                "value": 683.5258673276861
            },
            "totalGlobalMem": {
                "unit": "bytes",
                "value": 34342961152
            },
            "writeBandwidth": {
                "unit": "GiB/s",
                "value": 665.0294389762723
            }
        },
        "scalarL1": {
            "fetchGranularity": {
                "confidence": 0.999920001599968,
                "method": "p-chase",
                "randomized": false,
                "size": 64,
                "unit": "bytes"
            },
            "latency": {
                "mean": 49.568627450980394,
                "measurements": 255,
                "method": "p-chase",
                "p50": 48.0,
                "p95": 48.0,
                "sampleSize": 256,
                "stdev": 17.680959666343405,
                "unit": "cycles"
            },
            "lineSize": {
                "confidence": 0.9978924086625234,
                "method": "p-chase",
                "randomized": false,
                "size": 64,
                "unit": "bytes"
            },
            "missPenalty": {
                "unit": "cycles",
                "value": 155.51503788766504
            },
            "sharedBetween": [
                [
                    1,
                    2
                ],
                [
                    3,
                    4,
                    5
                ],
                [
                    6,
                    7
                ],
                [
                    8,
                    9,
                    10
                ],
                [
                    11,
                    12,
                    13
                ],
                [
                    14,
                    15
                ],
                [
                    17,
                    18
                ],
                [
                    19,
                    20,
                    21
                ],
                [
                    22,
                    23
                ],
                [
                    24,
                    25,
                    26
                ],
                [
                    27,
                    28,
                    29
                ],
                [
                    30,
                    31
                ],
                [
                    33,
                    34
                ],
                [
                    35,
                    36,
                    37
                ],
                [
                    38,
                    39
                ],
                [
                    40,
                    41,
                    42
                ],
                [
                    43,
                    44,
                    45
                ],
                [
                    46,
                    47
                ],
                [
                    49,
                    50
                ],
                [
                    51,
                    52,
                    53
                ],
                [
                    54,
                    55
                ],
                [
                    56,
                    57,
                    58
                ],
                [
                    59,
                    60,
                    61
                ],
                [
                    62,
                    63
                ],
                [
                    64,
                    65,
                    66
                ],
                [
                    67,
                    68
                ],
                [
                    69,
                    70,
                    71
                ],
                [
                    72,
                    73,
                    74
                ],
                [
                    75,
                    76
                ],
                [
                    78,
                    79
                ],
                [
                    81,
                    82
                ],
                [
                    83,
                    84,
                    85
                ],
                [
                    86,
                    87
                ],
                [
                    88,
                    89,
                    90
                ],
                [
                    91,
                    92,
                    93
                ],
                [
                    94,
                    95
                ],
                [
                    97,
                    98
                ],
                [
                    99,
                    100,
                    101
                ],
                [
                    102,
                    103
                ],
                [
                    104,
                    105,
                    106
                ],
                [
                    107,
                    108,
                    109
                ],
                [
                    110,
                    111
                ],
                [
                    113,
                    114
                ],
                [
                    115,
                    116,
                    117
                ],
                [
                    118,
                    119
                ],
                [
                    120,
                    121,
                    122
                ],
                [
                    123,
                    124,
                    125
                ],
                [
                    126,
                    127
                ]
            ],
            "size": {
                "confidence": 0.5059898802023959,
                "method": "p-chase",
                "randomized": false,
                "size": 16128,
                "unit": "bytes"
            },
            "uniqueAmount": 48
        },
        "shared": {
            "latency": {
                "mean": 49.976470588235294,
                "measurements": 255,
                "method": "p-chase",
                "p50": 48.0,
                "p95": 48.0,
                "sampleSize": 256,
                "stdev": 33.271719495735056,
                "unit": "cycles"
            },
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
                "value": 7864320
            }
        }
    },
    "meta": {
        "driver": 60342134,
        "gpuCompiler": "hipcc 6.3.42134",
        "hostCompiler": "clang 18.0.0",
        "hostCpu": "AMD EPYC 7742 64-Core Processor",
        "os": "Linux 5.14.21-150500.55.88-default",
        "runtime": 60342134,
        "timestamp": "2025-08-08T19:23:35Z"
    }
}
```
