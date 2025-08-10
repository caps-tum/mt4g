# AMD Instinct MI300X VF Benchmark Report

## General

| Key | Value |
| --- | ----- |
| asicRevision | 1 |
| clockRate | 2100000 kHz |
| computeCapability | 9.4 |
| name | AMD Instinct MI300X VF |
| vendor | AMD |

## Compute

| Key | Value |
| --- | ----- |
| computeUnitsPerDie | 38 |
| concurrentKernels | true |
| maxBlocksPerMultiProcessor | 2 |
| maxThreadsPerBlock | 1024 |
| maxThreadsPerMultiProcessor | 2048 |
| multiProcessorCount | 304 |
| numSIMDsPerCU | 4 |
| numXCDs | 8 |
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
| fetchGranularity | 128 bytes |
| globalL1CacheSupported | true |
| latency | 116 cycles |
| lineSize | 128 bytes |
| localL1CacheSupported | true |
| missPenalty | 133.961 cycles |
| size | 32768 bytes |

### l2

| Key | Value |
| --- | ----- |
| amount | 1 |
| fetchGranularity | 128 bytes |
| latency | 287.545 cycles |
| lineSize | 128 bytes |
| missPenalty | 208.878 cycles |
| persistingL2CacheMaxSize | 4194304 bytes |
| readBandwidth | 14364.7 GiB/s |
| size | 4194304 bytes |
| writeBandwidth | 6562.21 GiB/s |

### l3

| Key | Value |
| --- | ----- |
| amount | 1 |
| lineSize | 64 bytes |
| readBandwidth | 5399.65 GiB/s |
| size | 268435456 bytes |
| writeBandwidth | 5090.35 GiB/s |

### main

| Key | Value |
| --- | ----- |
| latency | 899.658 cycles |
| memoryBusWidth | 8192 bit |
| memoryClockRate | 1300000 kHz |
| readBandwidth | 3689.67 GiB/s |
| totalGlobalMem | 205822885888 bytes |
| writeBandwidth | 4116.64 GiB/s |

### scalarL1

| Key | Value |
| --- | ----- |
| fetchGranularity | 64 bytes |
| latency | 48 cycles |
| lineSize | 64 bytes |
| missPenalty | 123.937 cycles |
| size | 17920 bytes |
| uniqueAmount | 0 |

### shared

| Key | Value |
| --- | ----- |
| latency | 51.2941 cycles |
| reservedSharedMemPerBlock | 0 bytes |
| sharedMemPerBlock | 65536 bytes |
| sharedMemPerMultiProcessor | 19922944 bytes |

## Graphs

![AMD Instinct MI300X VF - L1 Fetch Granularity](./AMD%20Instinct%20MI300X%20VF%20-%20L1%20Fetch%20Granularity.png)
[Raw data](./AMD%20Instinct%20MI300X%20VF%20-%20L1%20Fetch%20Granularity.txt)

![AMD Instinct MI300X VF - L1 Line Size](./AMD%20Instinct%20MI300X%20VF%20-%20L1%20Line%20Size.png)
[Raw data](./AMD%20Instinct%20MI300X%20VF%20-%20L1%20Line%20Size.txt)

![AMD Instinct MI300X VF - L1 Size](./AMD%20Instinct%20MI300X%20VF%20-%20L1%20Size.png)
[Raw data](./AMD%20Instinct%20MI300X%20VF%20-%20L1%20Size.txt)

![AMD Instinct MI300X VF - L2 Fetch Granularity](./AMD%20Instinct%20MI300X%20VF%20-%20L2%20Fetch%20Granularity.png)
[Raw data](./AMD%20Instinct%20MI300X%20VF%20-%20L2%20Fetch%20Granularity.txt)

![AMD Instinct MI300X VF - Scalar L1 Fetch Granularity](./AMD%20Instinct%20MI300X%20VF%20-%20Scalar%20L1%20Fetch%20Granularity.png)
[Raw data](./AMD%20Instinct%20MI300X%20VF%20-%20Scalar%20L1%20Fetch%20Granularity.txt)

![AMD Instinct MI300X VF - Scalar L1 Line Size](./AMD%20Instinct%20MI300X%20VF%20-%20Scalar%20L1%20Line%20Size.png)
[Raw data](./AMD%20Instinct%20MI300X%20VF%20-%20Scalar%20L1%20Line%20Size.txt)

![AMD Instinct MI300X VF - Scalar L1 Size](./AMD%20Instinct%20MI300X%20VF%20-%20Scalar%20L1%20Size.png)
[Raw data](./AMD%20Instinct%20MI300X%20VF%20-%20Scalar%20L1%20Size.txt)

## Raw JSON

```json
{
    "compute": {
        "computeUnitsPerDie": 38,
        "concurrentKernels": true,
        "maxBlocksPerMultiProcessor": 2,
        "maxThreadsPerBlock": 1024,
        "maxThreadsPerMultiProcessor": 2048,
        "multiProcessorCount": 304,
        "numSIMDsPerCU": 4,
        "numXCDs": 8,
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
            "value": 2100000
        },
        "computeCapability": {
            "major": 9,
            "minor": 4
        },
        "name": "AMD Instinct MI300X VF",
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
                "confidence": 0.9668406631867362,
                "method": "p-chase",
                "randomized": false,
                "size": 128,
                "unit": "bytes"
            },
            "globalL1CacheSupported": true,
            "latency": {
                "mean": 116.0,
                "measurements": 255,
                "method": "p-chase",
                "p50": 116.0,
                "p95": 116.0,
                "sampleSize": 256,
                "stdev": 0.0,
                "unit": "cycles"
            },
            "lineSize": {
                "confidence": 0.9758227733222214,
                "method": "p-chase",
                "randomized": false,
                "size": 128,
                "unit": "bytes"
            },
            "localL1CacheSupported": true,
            "missPenalty": {
                "unit": "cycles",
                "value": 133.9607843137255
            },
            "size": {
                "confidence": 0.9740512974351282,
                "method": "p-chase",
                "randomized": false,
                "size": 32768,
                "unit": "bytes"
            }
        },
        "l2": {
            "amount": 1,
            "fetchGranularity": {
                "confidence": 0.9664006719865603,
                "method": "p-chase",
                "randomized": false,
                "size": 128,
                "unit": "bytes"
            },
            "latency": {
                "mean": 287.5450980392157,
                "measurements": 255,
                "method": "p-chase",
                "p50": 288.0,
                "p95": 292.0,
                "sampleSize": 256,
                "stdev": 4.286598495728145,
                "unit": "cycles"
            },
            "lineSize": {
                "unit": "bytes",
                "value": 128
            },
            "missPenalty": {
                "unit": "cycles",
                "value": 208.87843137254902
            },
            "persistingL2CacheMaxSize": {
                "unit": "bytes",
                "value": 4194304
            },
            "readBandwidth": {
                "unit": "GiB/s",
                "value": 14364.720146712272
            },
            "size": {
                "unit": "bytes",
                "value": 4194304
            },
            "writeBandwidth": {
                "unit": "GiB/s",
                "value": 6562.20514007817
            }
        },
        "l3": {
            "amount": 1,
            "lineSize": {
                "unit": "bytes",
                "value": 64
            },
            "readBandwidth": {
                "unit": "GiB/s",
                "value": 5399.646850198547
            },
            "size": {
                "unit": "bytes",
                "value": 268435456
            },
            "writeBandwidth": {
                "unit": "GiB/s",
                "value": 5090.345027469441
            }
        },
        "main": {
            "latency": {
                "mean": 899.6580361504641,
                "measurements": 2047,
                "method": "p-chase",
                "p50": 876.0,
                "p95": 1114.7999999999993,
                "sampleSize": 2048,
                "stdev": 145.14091414416808,
                "unit": "cycles"
            },
            "memoryBusWidth": {
                "unit": "bit",
                "value": 8192
            },
            "memoryClockRate": {
                "unit": "kHz",
                "value": 1300000
            },
            "readBandwidth": {
                "unit": "GiB/s",
                "value": 3689.6705494791395
            },
            "totalGlobalMem": {
                "unit": "bytes",
                "value": 205822885888
            },
            "writeBandwidth": {
                "unit": "GiB/s",
                "value": 4116.64088707054
            }
        },
        "scalarL1": {
            "fetchGranularity": {
                "confidence": 0.999980000399992,
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
                "confidence": 0.9985869882181577,
                "method": "p-chase",
                "randomized": false,
                "size": 64,
                "unit": "bytes"
            },
            "missPenalty": {
                "unit": "cycles",
                "value": 123.93725490196078
            },
            "sharedBetween": [],
            "size": {
                "confidence": 0.5076498470030599,
                "method": "p-chase",
                "randomized": false,
                "size": 17920,
                "unit": "bytes"
            },
            "uniqueAmount": 0
        },
        "shared": {
            "latency": {
                "mean": 51.294117647058826,
                "measurements": 255,
                "method": "p-chase",
                "p50": 52.0,
                "p95": 52.0,
                "sampleSize": 256,
                "stdev": 2.829082080671313,
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
                "value": 19922944
            }
        }
    },
    "meta": {
        "driver": 60443483,
        "gpuCompiler": "hipcc 6.4.43483",
        "hostCompiler": "clang 19.0.0",
        "hostCpu": "INTEL(R) XEON(R) PLATINUM 8568Y+",
        "os": "Linux 6.8.0-60-generic",
        "runtime": 60443483,
        "timestamp": "2025-08-08T23:16:36Z"
    }
}
```
