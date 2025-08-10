# NVIDIA H100 80GB HBM3 Benchmark Report

## General

| Key | Value |
| --- | ----- |
| asicRevision | 0 |
| clockRate | 1980000 kHz |
| computeCapability | 9.0 |
| name | NVIDIA H100 80GB HBM3 |
| vendor | NVIDIA |

## Compute

| Key | Value |
| --- | ----- |
| concurrentKernels | true |
| maxBlocksPerMultiProcessor | 32 |
| maxThreadsPerBlock | 1024 |
| maxThreadsPerMultiProcessor | 2048 |
| multiProcessorCount | 132 |
| numberOfCoresPerMultiProcessor | 128 |
| regsPerBlock | 65536 |
| regsPerMultiProcessor | 65536 |
| supportsCooperativeLaunch | true |
| warpSize | 32 |

## Memory

### constant

| Key | Value |
| --- | ----- |
| totalConstMem | 65536 bytes |

#### constant l1

| Key | Value |
| --- | ----- |
| amountPerMultiprocessor | 1 |
| fetchGranularity | 64 bytes |
| latency | 20.698 cycles |
| lineSize | 64 bytes |
| missPenalty | 55.302 cycles |
| size | 2112 bytes |

#### constant l1.5

| Key | Value |
| --- | ----- |
| fetchGranularity | 256 bytes |
| latency | 104.898 cycles |
| size | 65537 bytes |

### l1

| Key | Value |
| --- | ----- |
| amountPerMultiprocessor | 1 |
| fetchGranularity | 32 bytes |
| globalL1CacheSupported | true |
| latency | 38 cycles |
| lineSize | 128 bytes |
| localL1CacheSupported | true |
| missPenalty | 179.816 cycles |
| sharedWith | Read Only, Texture |
| size | 238848 bytes |

### l2

| Key | Value |
| --- | ----- |
| fetchGranularity | 32 bytes |
| latency | 219.345 cycles |
| lineSize | 128 bytes |
| missPenalty | 399.945 cycles |
| persistingL2CacheMaxSize | 39321600 bytes |
| readBandwidth | 4528.92 GiB/s |
| segmentSize | 26214400 bytes |
| size | 52428800 bytes |
| writeBandwidth | 3494.61 GiB/s |

### main

| Key | Value |
| --- | ----- |
| latency | 834.476 cycles |
| memoryBusWidth | 5120 bit |
| memoryClockRate | 2619000 kHz |
| readBandwidth | 2561.03 GiB/s |
| totalGlobalMem | 85176483840 bytes |
| writeBandwidth | 2805.9 GiB/s |

### readOnly

| Key | Value |
| --- | ----- |
| amountPerMultiprocessor | 1 |
| fetchGranularity | 32 bytes |
| latency | 34.8627 cycles |
| lineSize | 128 bytes |
| missPenalty | 183.435 cycles |
| sharedWith | L1, Texture |
| size | 238848 bytes |

### shared

| Key | Value |
| --- | ----- |
| latency | 29.8824 cycles |
| reservedSharedMemPerBlock | 1024 bytes |
| sharedMemPerBlock | 49152 bytes |
| sharedMemPerMultiProcessor | 233472 bytes |

### texture

| Key | Value |
| --- | ----- |
| amountPerMultiprocessor | 1 |
| fetchGranularity | 32 bytes |
| latency | 38.7765 cycles |
| lineSize | 128 bytes |
| missPenalty | 258.953 cycles |
| sharedWith | L1, Read Only |
| size | 243456 bytes |

## Graphs

![NVIDIA H100 80GB HBM3 - Constant L1 Fetch Granularity](./NVIDIA%20H100%2080GB%20HBM3%20-%20Constant%20L1%20Fetch%20Granularity.png)
[Raw data](./NVIDIA%20H100%2080GB%20HBM3%20-%20Constant%20L1%20Fetch%20Granularity.txt)

![NVIDIA H100 80GB HBM3 - Constant L1 Line Size](./NVIDIA%20H100%2080GB%20HBM3%20-%20Constant%20L1%20Line%20Size.png)
[Raw data](./NVIDIA%20H100%2080GB%20HBM3%20-%20Constant%20L1%20Line%20Size.txt)

![NVIDIA H100 80GB HBM3 - Constant L1 Size](./NVIDIA%20H100%2080GB%20HBM3%20-%20Constant%20L1%20Size.png)
[Raw data](./NVIDIA%20H100%2080GB%20HBM3%20-%20Constant%20L1%20Size.txt)

![NVIDIA H100 80GB HBM3 - Constant L1.5 Fetch Granularity](./NVIDIA%20H100%2080GB%20HBM3%20-%20Constant%20L1.5%20Fetch%20Granularity.png)
[Raw data](./NVIDIA%20H100%2080GB%20HBM3%20-%20Constant%20L1.5%20Fetch%20Granularity.txt)

![NVIDIA H100 80GB HBM3 - Constant L1.5 Size](./NVIDIA%20H100%2080GB%20HBM3%20-%20Constant%20L1.5%20Size.png)
[Raw data](./NVIDIA%20H100%2080GB%20HBM3%20-%20Constant%20L1.5%20Size.txt)

![NVIDIA H100 80GB HBM3 - L1 Fetch Granularity](./NVIDIA%20H100%2080GB%20HBM3%20-%20L1%20Fetch%20Granularity.png)
[Raw data](./NVIDIA%20H100%2080GB%20HBM3%20-%20L1%20Fetch%20Granularity.txt)

![NVIDIA H100 80GB HBM3 - L1 Line Size](./NVIDIA%20H100%2080GB%20HBM3%20-%20L1%20Line%20Size.png)
[Raw data](./NVIDIA%20H100%2080GB%20HBM3%20-%20L1%20Line%20Size.txt)

![NVIDIA H100 80GB HBM3 - L1 Size](./NVIDIA%20H100%2080GB%20HBM3%20-%20L1%20Size.png)
[Raw data](./NVIDIA%20H100%2080GB%20HBM3%20-%20L1%20Size.txt)

![NVIDIA H100 80GB HBM3 - L2 Fetch Granularity](./NVIDIA%20H100%2080GB%20HBM3%20-%20L2%20Fetch%20Granularity.png)
[Raw data](./NVIDIA%20H100%2080GB%20HBM3%20-%20L2%20Fetch%20Granularity.txt)

![NVIDIA H100 80GB HBM3 - L2 Line Size](./NVIDIA%20H100%2080GB%20HBM3%20-%20L2%20Line%20Size.png)
[Raw data](./NVIDIA%20H100%2080GB%20HBM3%20-%20L2%20Line%20Size.txt)

![NVIDIA H100 80GB HBM3 - L2 Segment Size](./NVIDIA%20H100%2080GB%20HBM3%20-%20L2%20Segment%20Size.png)
[Raw data](./NVIDIA%20H100%2080GB%20HBM3%20-%20L2%20Segment%20Size.txt)

![NVIDIA H100 80GB HBM3 - Read Only Fetch Granularity](./NVIDIA%20H100%2080GB%20HBM3%20-%20Read%20Only%20Fetch%20Granularity.png)
[Raw data](./NVIDIA%20H100%2080GB%20HBM3%20-%20Read%20Only%20Fetch%20Granularity.txt)

![NVIDIA H100 80GB HBM3 - Read Only Line Size](./NVIDIA%20H100%2080GB%20HBM3%20-%20Read%20Only%20Line%20Size.png)
[Raw data](./NVIDIA%20H100%2080GB%20HBM3%20-%20Read%20Only%20Line%20Size.txt)

![NVIDIA H100 80GB HBM3 - Read Only Size](./NVIDIA%20H100%2080GB%20HBM3%20-%20Read%20Only%20Size.png)
[Raw data](./NVIDIA%20H100%2080GB%20HBM3%20-%20Read%20Only%20Size.txt)

![NVIDIA H100 80GB HBM3 - Texture Fetch Granularity](./NVIDIA%20H100%2080GB%20HBM3%20-%20Texture%20Fetch%20Granularity.png)
[Raw data](./NVIDIA%20H100%2080GB%20HBM3%20-%20Texture%20Fetch%20Granularity.txt)

![NVIDIA H100 80GB HBM3 - Texture Line Size](./NVIDIA%20H100%2080GB%20HBM3%20-%20Texture%20Line%20Size.png)
[Raw data](./NVIDIA%20H100%2080GB%20HBM3%20-%20Texture%20Line%20Size.txt)

![NVIDIA H100 80GB HBM3 - Texture Size](./NVIDIA%20H100%2080GB%20HBM3%20-%20Texture%20Size.png)
[Raw data](./NVIDIA%20H100%2080GB%20HBM3%20-%20Texture%20Size.txt)

## Raw JSON

```json
{
    "compute": {
        "concurrentKernels": true,
        "maxBlocksPerMultiProcessor": 32,
        "maxThreadsPerBlock": 1024,
        "maxThreadsPerMultiProcessor": 2048,
        "multiProcessorCount": 132,
        "numberOfCoresPerMultiProcessor": 128,
        "regsPerBlock": 65536,
        "regsPerMultiProcessor": 65536,
        "supportsCooperativeLaunch": true,
        "warpSize": 32
    },
    "general": {
        "asicRevision": 0,
        "clockRate": {
            "unit": "kHz",
            "value": 1980000
        },
        "computeCapability": {
            "major": 9,
            "minor": 0
        },
        "name": "NVIDIA H100 80GB HBM3",
        "vendor": "NVIDIA"
    },
    "memory": {
        "constant": {
            "l1": {
                "amountPerMultiprocessor": 1,
                "fetchGranularity": {
                    "confidence": 0.999980000399992,
                    "method": "p-chase",
                    "randomized": false,
                    "size": 64,
                    "unit": "bytes"
                },
                "latency": {
                    "mean": 20.698039215686276,
                    "measurements": 255,
                    "method": "p-chase",
                    "p50": 17.0,
                    "p95": 76.0,
                    "sampleSize": 256,
                    "stdev": 14.32081094287081,
                    "unit": "cycles"
                },
                "lineSize": {
                    "confidence": 0.9997892055722847,
                    "method": "p-chase",
                    "randomized": false,
                    "size": 64,
                    "unit": "bytes"
                },
                "missPenalty": {
                    "unit": "cycles",
                    "value": 55.30196078431372
                },
                "size": {
                    "confidence": 0.8989220215595688,
                    "method": "p-chase",
                    "randomized": false,
                    "size": 2112,
                    "unit": "bytes"
                }
            },
            "l1.5": {
                "fetchGranularity": {
                    "confidence": 0.99990000499975,
                    "method": "p-chase",
                    "randomized": false,
                    "size": 256,
                    "unit": "bytes"
                },
                "latency": {
                    "mean": 104.89763779527559,
                    "measurements": 127,
                    "method": "p-chase",
                    "p50": 106.0,
                    "p95": 106.0,
                    "sampleSize": 256,
                    "stdev": 8.7494531762642,
                    "unit": "cycles"
                },
                "size": {
                    "confidence": 0.0,
                    "method": "p-chase",
                    "randomized": false,
                    "size": 65537,
                    "unit": "bytes"
                }
            },
            "totalConstMem": {
                "unit": "bytes",
                "value": 65536
            }
        },
        "l1": {
            "amountPerMultiprocessor": 1,
            "fetchGranularity": {
                "confidence": 0.999980000399992,
                "method": "p-chase",
                "randomized": false,
                "size": 32,
                "unit": "bytes"
            },
            "globalL1CacheSupported": true,
            "latency": {
                "mean": 38.0,
                "measurements": 255,
                "method": "p-chase",
                "p50": 38.0,
                "p95": 38.0,
                "sampleSize": 256,
                "stdev": 0.0,
                "unit": "cycles"
            },
            "lineSize": {
                "confidence": 0.9759562845174032,
                "method": "p-chase",
                "randomized": false,
                "size": 128,
                "unit": "bytes"
            },
            "localL1CacheSupported": true,
            "missPenalty": {
                "unit": "cycles",
                "value": 179.8156862745098
            },
            "sharedWith": [
                "Read Only",
                "Texture"
            ],
            "size": {
                "confidence": 0.9715514224288786,
                "method": "p-chase",
                "randomized": false,
                "size": 238848,
                "unit": "bytes"
            }
        },
        "l2": {
            "fetchGranularity": {
                "confidence": 0.999980000399992,
                "method": "p-chase",
                "randomized": false,
                "size": 32,
                "unit": "bytes"
            },
            "latency": {
                "mean": 219.34509803921569,
                "measurements": 255,
                "method": "p-chase",
                "p50": 222.0,
                "p95": 231.0,
                "sampleSize": 256,
                "stdev": 11.568289431215238,
                "unit": "cycles"
            },
            "lineSize": {
                "confidence": 0.9714499706490953,
                "method": "p-chase",
                "randomized": false,
                "size": 128,
                "unit": "bytes"
            },
            "missPenalty": {
                "unit": "cycles",
                "value": 399.9450980392156
            },
            "persistingL2CacheMaxSize": {
                "unit": "bytes",
                "value": 39321600
            },
            "readBandwidth": {
                "unit": "GiB/s",
                "value": 4528.922496649036
            },
            "segmentSize": {
                "confidence": 0.9288281825358617,
                "method": "p-chase",
                "randomized": false,
                "size": 26214400,
                "unit": "bytes"
            },
            "size": {
                "unit": "bytes",
                "value": 52428800
            },
            "writeBandwidth": {
                "unit": "GiB/s",
                "value": 3494.612631053556
            }
        },
        "main": {
            "latency": {
                "mean": 834.4758182706399,
                "measurements": 2047,
                "method": "p-chase",
                "p50": 864.0,
                "p95": 1126.3999999999996,
                "sampleSize": 2048,
                "stdev": 140.4647198184919,
                "unit": "cycles"
            },
            "memoryBusWidth": {
                "unit": "bit",
                "value": 5120
            },
            "memoryClockRate": {
                "unit": "kHz",
                "value": 2619000
            },
            "readBandwidth": {
                "unit": "GiB/s",
                "value": 2561.031719280527
            },
            "totalGlobalMem": {
                "unit": "bytes",
                "value": 85176483840
            },
            "writeBandwidth": {
                "unit": "GiB/s",
                "value": 2805.8959593455534
            }
        },
        "readOnly": {
            "amountPerMultiprocessor": 1,
            "fetchGranularity": {
                "confidence": 0.998920021599568,
                "method": "p-chase",
                "randomized": false,
                "size": 32,
                "unit": "bytes"
            },
            "latency": {
                "mean": 34.86274509803921,
                "measurements": 255,
                "method": "p-chase",
                "p50": 34.0,
                "p95": 44.0,
                "sampleSize": 256,
                "stdev": 2.5488889647166313,
                "unit": "cycles"
            },
            "lineSize": {
                "confidence": 0.9981085859730913,
                "method": "p-chase",
                "randomized": false,
                "size": 128,
                "unit": "bytes"
            },
            "missPenalty": {
                "unit": "cycles",
                "value": 183.43529411764706
            },
            "sharedWith": [
                "L1",
                "Texture"
            ],
            "size": {
                "confidence": 0.9701014949252538,
                "method": "p-chase",
                "randomized": false,
                "size": 238848,
                "unit": "bytes"
            }
        },
        "shared": {
            "latency": {
                "mean": 29.88235294117647,
                "measurements": 255,
                "method": "p-chase",
                "p50": 30.0,
                "p95": 30.0,
                "sampleSize": 256,
                "stdev": 0.4715136801118872,
                "unit": "cycles"
            },
            "reservedSharedMemPerBlock": {
                "unit": "bytes",
                "value": 1024
            },
            "sharedMemPerBlock": {
                "unit": "bytes",
                "value": 49152
            },
            "sharedMemPerMultiProcessor": {
                "unit": "bytes",
                "value": 233472
            }
        },
        "texture": {
            "amountPerMultiprocessor": 1,
            "fetchGranularity": {
                "confidence": 0.9682806343873123,
                "method": "p-chase",
                "randomized": false,
                "size": 32,
                "unit": "bytes"
            },
            "latency": {
                "mean": 38.77647058823529,
                "measurements": 255,
                "method": "p-chase",
                "p50": 42.0,
                "p95": 42.0,
                "sampleSize": 256,
                "stdev": 5.776090694811043,
                "unit": "cycles"
            },
            "lineSize": {
                "confidence": 0.9962346701253728,
                "method": "p-chase",
                "randomized": false,
                "size": 128,
                "unit": "bytes"
            },
            "missPenalty": {
                "unit": "cycles",
                "value": 258.95294117647063
            },
            "sharedWith": [
                "L1",
                "Read Only"
            ],
            "size": {
                "confidence": 0.9722513874306284,
                "method": "p-chase",
                "randomized": false,
                "size": 243456,
                "unit": "bytes"
            }
        }
    },
    "meta": {
        "driver": 12020,
        "gpuCompiler": "nvcc 12.9.41",
        "hostCompiler": "gcc 11.3.1",
        "hostCpu": "AMD EPYC 9374F 32-Core Processor",
        "os": "Linux 5.14.0-162.6.1.el9_1.x86_64",
        "runtime": 12020,
        "timestamp": "2025-08-10T00:03:09Z"
    }
}
```
