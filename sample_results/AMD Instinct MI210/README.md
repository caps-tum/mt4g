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
| maxBlocksPerMultiProcessor | 2 |
| maxThreadsPerBlock | 1024 |
| maxThreadsPerMultiProcessor | 2048 |
| multiProcessorCount | 104 |
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
| latency | 124 cycles |
| lineSize | 64 bytes |
| localL1CacheSupported | true |
| missPenalty | 164.408 cycles |
| size | 16384 bytes |

### l2

| Key | Value |
| --- | ----- |
| amount | 1 |
| fetchGranularity | 64 bytes |
| latency | 302.149 cycles |
| lineSize | 128 bytes |
| missPenalty | 412.831 cycles |
| persistingL2CacheMaxSize | 8388608 bytes |
| readBandwidth | 4014.12 GiB/s |
| size | 8388608 bytes |
| writeBandwidth | 2403.75 GiB/s |

### main

| Key | Value |
| --- | ----- |
| latency | 740.883 cycles |
| memoryBusWidth | 4096 bit |
| memoryClockRate | 1600000 kHz |
| readBandwidth | 1085.42 GiB/s |
| totalGlobalMem | 68702699520 bytes |
| writeBandwidth | 966.707 GiB/s |

### scalarL1

| Key | Value |
| --- | ----- |
| fetchGranularity | 64 bytes |
| latency | 52.7686 cycles |
| lineSize | 64 bytes |
| missPenalty | 126.949 cycles |
| size | 16896 bytes |
| uniqueAmount | 56 |

| Group 1 | Group 2 | Group 3 | Group 4 | Group 5 | Group 6 | Group 7 | Group 8 | Group 9 | Group 10 | Group 11 | Group 12 | Group 13 | Group 14 | Group 15 | Group 16 | Group 17 | Group 18 | Group 19 | Group 20 | Group 21 | Group 22 | Group 23 | Group 24 | Group 25 | Group 26 | Group 27 | Group 28 | Group 29 | Group 30 | Group 31 | Group 32 | Group 33 | Group 34 | Group 35 | Group 36 | Group 37 | Group 38 | Group 39 | Group 40 | Group 41 | Group 42 | Group 43 | Group 44 | Group 45 | Group 46 | Group 47 | Group 48 | Group 49 | Group 50 | Group 51 | Group 52 | Group 53 | Group 54 | Group 55 | Group 56 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0, 1 | 3 | 4, 5 | 6, 7 | 8, 9 | 10, 11 | 12, 13 | 16, 17 | 18, 19 | 20, 21 | 22, 23 | 24, 25 | 26, 27 | 28 | 32, 33 | 34, 35 | 36, 37 | 38 | 40, 41 | 42, 43 | 44, 45 | 48, 49 | 50, 51 | 52, 53 | 54, 55 | 56, 57 | 58, 59 | 60 | 64, 65 | 66, 67 | 68, 69 | 70, 71 | 72, 73 | 74, 75 | 76 | 80, 81 | 82, 83 | 84, 85 | 86, 87 | 88, 89 | 90, 91 | 92 | 96, 97 | 98, 99 | 100, 101 | 102, 103 | 104, 105 | 106, 107 | 108 | 112, 113 | 114, 115 | 116, 117 | 118, 119 | 120, 121 | 122, 123 | 124 |

### shared

| Key | Value |
| --- | ----- |
| latency | 49.9451 cycles |
| reservedSharedMemPerBlock | 0 bytes |
| sharedMemPerBlock | 65536 bytes |
| sharedMemPerMultiProcessor | 6815744 bytes |

## Graphs

![AMD Instinct MI210 - L1 Fetch Granularity](./AMD%20Instinct%20MI210%20-%20L1%20Fetch%20Granularity.png)
[Raw data](./AMD%20Instinct%20MI210%20-%20L1%20Fetch%20Granularity.txt)

![AMD Instinct MI210 - L1 Line Size](./AMD%20Instinct%20MI210%20-%20L1%20Line%20Size.png)
[Raw data](./AMD%20Instinct%20MI210%20-%20L1%20Line%20Size.txt)

![AMD Instinct MI210 - L1 Size](./AMD%20Instinct%20MI210%20-%20L1%20Size.png)
[Raw data](./AMD%20Instinct%20MI210%20-%20L1%20Size.txt)

![AMD Instinct MI210 - L2 Fetch Granularity](./AMD%20Instinct%20MI210%20-%20L2%20Fetch%20Granularity.png)
[Raw data](./AMD%20Instinct%20MI210%20-%20L2%20Fetch%20Granularity.txt)

![AMD Instinct MI210 - Scalar L1 Fetch Granularity](./AMD%20Instinct%20MI210%20-%20Scalar%20L1%20Fetch%20Granularity.png)
[Raw data](./AMD%20Instinct%20MI210%20-%20Scalar%20L1%20Fetch%20Granularity.txt)

![AMD Instinct MI210 - Scalar L1 Line Size](./AMD%20Instinct%20MI210%20-%20Scalar%20L1%20Line%20Size.png)
[Raw data](./AMD%20Instinct%20MI210%20-%20Scalar%20L1%20Line%20Size.txt)

![AMD Instinct MI210 - Scalar L1 Size](./AMD%20Instinct%20MI210%20-%20Scalar%20L1%20Size.png)
[Raw data](./AMD%20Instinct%20MI210%20-%20Scalar%20L1%20Size.txt)

## Raw JSON

```json
{
    "compute": {
        "computeUnitsPerDie": 104,
        "concurrentKernels": true,
        "maxBlocksPerMultiProcessor": 2,
        "maxThreadsPerBlock": 1024,
        "maxThreadsPerMultiProcessor": 2048,
        "multiProcessorCount": 104,
        "numSIMDsPerCU": 4,
        "numXCDs": 1,
        "numberOfCoresPerMultiProcessor": 64,
        "regsPerBlock": 65536,
        "regsPerMultiProcessor": 65536,
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
            "amountPerMultiprocessor": 1,
            "fetchGranularity": {
                "confidence": 0.9679006419871603,
                "method": "p-chase",
                "randomized": false,
                "size": 64,
                "unit": "bytes"
            },
            "globalL1CacheSupported": true,
            "latency": {
                "mean": 124.0,
                "measurements": 255,
                "method": "p-chase",
                "p50": 124.0,
                "p95": 124.0,
                "sampleSize": 256,
                "stdev": 0.0,
                "unit": "cycles"
            },
            "lineSize": {
                "confidence": 0.9746311193759021,
                "method": "p-chase",
                "randomized": false,
                "size": 64,
                "unit": "bytes"
            },
            "localL1CacheSupported": true,
            "missPenalty": {
                "unit": "cycles",
                "value": 164.40784313725493
            },
            "size": {
                "confidence": 0.9739013049347532,
                "method": "p-chase",
                "randomized": false,
                "size": 16384,
                "unit": "bytes"
            }
        },
        "l2": {
            "amount": 1,
            "fetchGranularity": {
                "confidence": 0.9678006439871203,
                "method": "p-chase",
                "randomized": false,
                "size": 64,
                "unit": "bytes"
            },
            "latency": {
                "mean": 302.1490196078431,
                "measurements": 255,
                "method": "p-chase",
                "p50": 292.0,
                "p95": 324.0,
                "sampleSize": 256,
                "stdev": 23.341890400313403,
                "unit": "cycles"
            },
            "lineSize": {
                "unit": "bytes",
                "value": 128
            },
            "missPenalty": {
                "unit": "cycles",
                "value": 412.83137254901965
            },
            "persistingL2CacheMaxSize": {
                "unit": "bytes",
                "value": 8388608
            },
            "readBandwidth": {
                "unit": "GiB/s",
                "value": 4014.1185476901746
            },
            "size": {
                "unit": "bytes",
                "value": 8388608
            },
            "writeBandwidth": {
                "unit": "GiB/s",
                "value": 2403.7476546766156
            }
        },
        "main": {
            "latency": {
                "mean": 740.8832437713727,
                "measurements": 2047,
                "method": "p-chase",
                "p50": 684.0,
                "p95": 1208.0,
                "sampleSize": 2048,
                "stdev": 186.54251268736903,
                "unit": "cycles"
            },
            "memoryBusWidth": {
                "unit": "bit",
                "value": 4096
            },
            "memoryClockRate": {
                "unit": "kHz",
                "value": 1600000
            },
            "readBandwidth": {
                "unit": "GiB/s",
                "value": 1085.4164756189655
            },
            "totalGlobalMem": {
                "unit": "bytes",
                "value": 68702699520
            },
            "writeBandwidth": {
                "unit": "GiB/s",
                "value": 966.7072954012621
            }
        },
        "scalarL1": {
            "fetchGranularity": {
                "confidence": 0.998920021599568,
                "method": "p-chase",
                "randomized": false,
                "size": 64,
                "unit": "bytes"
            },
            "latency": {
                "mean": 52.76862745098039,
                "measurements": 255,
                "method": "p-chase",
                "p50": 48.0,
                "p95": 48.0,
                "sampleSize": 256,
                "stdev": 59.81471881310992,
                "unit": "cycles"
            },
            "lineSize": {
                "confidence": 0.9934507078808761,
                "method": "p-chase",
                "randomized": false,
                "size": 64,
                "unit": "bytes"
            },
            "missPenalty": {
                "unit": "cycles",
                "value": 126.94901960784314
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
                "confidence": 0.5104897902041959,
                "method": "p-chase",
                "randomized": false,
                "size": 16896,
                "unit": "bytes"
            },
            "uniqueAmount": 56
        },
        "shared": {
            "latency": {
                "mean": 49.94509803921569,
                "measurements": 255,
                "method": "p-chase",
                "p50": 48.0,
                "p95": 48.0,
                "sampleSize": 256,
                "stdev": 29.331343006570542,
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
                "value": 6815744
            }
        }
    },
    "meta": {
        "driver": 60342134,
        "gpuCompiler": "hipcc 6.3.42134",
        "hostCompiler": "clang 18.0.0",
        "hostCpu": "AMD EPYC 7773X 64-Core Processor",
        "os": "Linux 5.14.21-150400.24.63-default",
        "runtime": 60342134,
        "timestamp": "2025-08-08T22:15:20Z"
    }
}
```
