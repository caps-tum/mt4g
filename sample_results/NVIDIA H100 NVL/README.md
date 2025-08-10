# NVIDIA H100 NVL Benchmark Report

## General

| Key | Value |
| --- | ----- |
| asicRevision | 0 |
| clockRate | 1785000 kHz |
| computeCapability | 9.0 |
| name | NVIDIA H100 NVL |
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
| missPenalty | 176.667 cycles |
| sharedWith | Read Only, Texture |
| size | 255232 bytes |

### l2

| Key | Value |
| --- | ----- |
| fetchGranularity | 32 bytes |
| latency | 237.82 cycles |
| lineSize | 128 bytes |
| missPenalty | 342.118 cycles |
| persistingL2CacheMaxSize | 39321600 bytes |
| readBandwidth | 4306.94 GiB/s |
| segmentSize | 31457280 bytes |
| size | 62914560 bytes |
| writeBandwidth | 3928.96 GiB/s |

### main

| Key | Value |
| --- | ----- |
| latency | 803.03 cycles |
| memoryBusWidth | 6144 bit |
| memoryClockRate | 2619000 kHz |
| readBandwidth | 2853.07 GiB/s |
| totalGlobalMem | 99960750080 bytes |
| writeBandwidth | 3334.19 GiB/s |

### readOnly

| Key | Value |
| --- | ----- |
| amountPerMultiprocessor | 1 |
| fetchGranularity | 32 bytes |
| latency | 34.8627 cycles |
| lineSize | 128 bytes |
| missPenalty | 179.898 cycles |
| sharedWith | L1, Texture |
| size | 258560 bytes |

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
| missPenalty | 256.031 cycles |
| sharedWith | L1, Read Only |
| size | 241920 bytes |

## Graphs

![NVIDIA H100 NVL - Constant L1 Fetch Granularity](./NVIDIA%20H100%20NVL%20-%20Constant%20L1%20Fetch%20Granularity.png)
[Raw data](./NVIDIA%20H100%20NVL%20-%20Constant%20L1%20Fetch%20Granularity.txt)

![NVIDIA H100 NVL - Constant L1 Line Size](./NVIDIA%20H100%20NVL%20-%20Constant%20L1%20Line%20Size.png)
[Raw data](./NVIDIA%20H100%20NVL%20-%20Constant%20L1%20Line%20Size.txt)

![NVIDIA H100 NVL - Constant L1 Size](./NVIDIA%20H100%20NVL%20-%20Constant%20L1%20Size.png)
[Raw data](./NVIDIA%20H100%20NVL%20-%20Constant%20L1%20Size.txt)

![NVIDIA H100 NVL - Constant L1.5 Fetch Granularity](./NVIDIA%20H100%20NVL%20-%20Constant%20L1.5%20Fetch%20Granularity.png)
[Raw data](./NVIDIA%20H100%20NVL%20-%20Constant%20L1.5%20Fetch%20Granularity.txt)

![NVIDIA H100 NVL - Constant L1.5 Size](./NVIDIA%20H100%20NVL%20-%20Constant%20L1.5%20Size.png)
[Raw data](./NVIDIA%20H100%20NVL%20-%20Constant%20L1.5%20Size.txt)

![NVIDIA H100 NVL - L1 Fetch Granularity](./NVIDIA%20H100%20NVL%20-%20L1%20Fetch%20Granularity.png)
[Raw data](./NVIDIA%20H100%20NVL%20-%20L1%20Fetch%20Granularity.txt)

![NVIDIA H100 NVL - L1 Line Size](./NVIDIA%20H100%20NVL%20-%20L1%20Line%20Size.png)
[Raw data](./NVIDIA%20H100%20NVL%20-%20L1%20Line%20Size.txt)

![NVIDIA H100 NVL - L1 Size](./NVIDIA%20H100%20NVL%20-%20L1%20Size.png)
[Raw data](./NVIDIA%20H100%20NVL%20-%20L1%20Size.txt)

![NVIDIA H100 NVL - L2 Fetch Granularity](./NVIDIA%20H100%20NVL%20-%20L2%20Fetch%20Granularity.png)
[Raw data](./NVIDIA%20H100%20NVL%20-%20L2%20Fetch%20Granularity.txt)

![NVIDIA H100 NVL - L2 Line Size](./NVIDIA%20H100%20NVL%20-%20L2%20Line%20Size.png)
[Raw data](./NVIDIA%20H100%20NVL%20-%20L2%20Line%20Size.txt)

![NVIDIA H100 NVL - L2 Segment Size](./NVIDIA%20H100%20NVL%20-%20L2%20Segment%20Size.png)
[Raw data](./NVIDIA%20H100%20NVL%20-%20L2%20Segment%20Size.txt)

![NVIDIA H100 NVL - Read Only Fetch Granularity](./NVIDIA%20H100%20NVL%20-%20Read%20Only%20Fetch%20Granularity.png)
[Raw data](./NVIDIA%20H100%20NVL%20-%20Read%20Only%20Fetch%20Granularity.txt)

![NVIDIA H100 NVL - Read Only Line Size](./NVIDIA%20H100%20NVL%20-%20Read%20Only%20Line%20Size.png)
[Raw data](./NVIDIA%20H100%20NVL%20-%20Read%20Only%20Line%20Size.txt)

![NVIDIA H100 NVL - Read Only Size](./NVIDIA%20H100%20NVL%20-%20Read%20Only%20Size.png)
[Raw data](./NVIDIA%20H100%20NVL%20-%20Read%20Only%20Size.txt)

![NVIDIA H100 NVL - Texture Fetch Granularity](./NVIDIA%20H100%20NVL%20-%20Texture%20Fetch%20Granularity.png)
[Raw data](./NVIDIA%20H100%20NVL%20-%20Texture%20Fetch%20Granularity.txt)

![NVIDIA H100 NVL - Texture Line Size](./NVIDIA%20H100%20NVL%20-%20Texture%20Line%20Size.png)
[Raw data](./NVIDIA%20H100%20NVL%20-%20Texture%20Line%20Size.txt)

![NVIDIA H100 NVL - Texture Size](./NVIDIA%20H100%20NVL%20-%20Texture%20Size.png)
[Raw data](./NVIDIA%20H100%20NVL%20-%20Texture%20Size.txt)

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
            "value": 1785000
        },
        "computeCapability": {
            "major": 9,
            "minor": 0
        },
        "name": "NVIDIA H100 NVL",
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
                    "confidence": 0.8967220655586888,
                    "method": "p-chase",
                    "randomized": false,
                    "size": 2112,
                    "unit": "bytes"
                }
            },
            "l1.5": {
                "fetchGranularity": {
                    "confidence": 0.9916504174791261,
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
                "confidence": 0.9675006499870002,
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
                "confidence": 0.9960453436547531,
                "method": "p-chase",
                "randomized": false,
                "size": 128,
                "unit": "bytes"
            },
            "localL1CacheSupported": true,
            "missPenalty": {
                "unit": "cycles",
                "value": 176.66666666666666
            },
            "sharedWith": [
                "Read Only",
                "Texture"
            ],
            "size": {
                "confidence": 0.9732513374331283,
                "method": "p-chase",
                "randomized": false,
                "size": 255232,
                "unit": "bytes"
            }
        },
        "l2": {
            "fetchGranularity": {
                "confidence": 0.998920021599568,
                "method": "p-chase",
                "randomized": false,
                "size": 32,
                "unit": "bytes"
            },
            "latency": {
                "mean": 237.81960784313725,
                "measurements": 255,
                "method": "p-chase",
                "p50": 227.0,
                "p95": 266.0,
                "sampleSize": 256,
                "stdev": 21.834172988272933,
                "unit": "cycles"
            },
            "lineSize": {
                "confidence": 0.956362009568931,
                "method": "p-chase",
                "randomized": false,
                "size": 128,
                "unit": "bytes"
            },
            "missPenalty": {
                "unit": "cycles",
                "value": 342.1176470588235
            },
            "persistingL2CacheMaxSize": {
                "unit": "bytes",
                "value": 39321600
            },
            "readBandwidth": {
                "unit": "GiB/s",
                "value": 4306.938893669694
            },
            "segmentSize": {
                "confidence": 0.9182937772515456,
                "method": "p-chase",
                "randomized": false,
                "size": 31457280,
                "unit": "bytes"
            },
            "size": {
                "unit": "bytes",
                "value": 62914560
            },
            "writeBandwidth": {
                "unit": "GiB/s",
                "value": 3928.9593960080233
            }
        },
        "main": {
            "latency": {
                "mean": 803.0302882266732,
                "measurements": 2047,
                "method": "p-chase",
                "p50": 802.0,
                "p95": 1081.0999999999995,
                "sampleSize": 2048,
                "stdev": 135.06754785637662,
                "unit": "cycles"
            },
            "memoryBusWidth": {
                "unit": "bit",
                "value": 6144
            },
            "memoryClockRate": {
                "unit": "kHz",
                "value": 2619000
            },
            "readBandwidth": {
                "unit": "GiB/s",
                "value": 2853.0717133773824
            },
            "totalGlobalMem": {
                "unit": "bytes",
                "value": 99960750080
            },
            "writeBandwidth": {
                "unit": "GiB/s",
                "value": 3334.1914340057683
            }
        },
        "readOnly": {
            "amountPerMultiprocessor": 1,
            "fetchGranularity": {
                "confidence": 0.9697206055878882,
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
                "confidence": 0.9969066160918526,
                "method": "p-chase",
                "randomized": false,
                "size": 128,
                "unit": "bytes"
            },
            "missPenalty": {
                "unit": "cycles",
                "value": 179.89803921568625
            },
            "sharedWith": [
                "L1",
                "Texture"
            ],
            "size": {
                "confidence": 0.9714514274286286,
                "method": "p-chase",
                "randomized": false,
                "size": 258560,
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
                "confidence": 0.9675606487870243,
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
                "confidence": 0.995957471127085,
                "method": "p-chase",
                "randomized": false,
                "size": 128,
                "unit": "bytes"
            },
            "missPenalty": {
                "unit": "cycles",
                "value": 256.03137254901964
            },
            "sharedWith": [
                "L1",
                "Read Only"
            ],
            "size": {
                "confidence": 0.9704014799260037,
                "method": "p-chase",
                "randomized": false,
                "size": 241920,
                "unit": "bytes"
            }
        }
    },
    "meta": {
        "driver": 12080,
        "gpuCompiler": "nvcc 12.9.41",
        "hostCompiler": "gcc 13.3.0",
        "hostCpu": "AMD EPYC 9374F 32-Core Processor",
        "os": "Linux 6.8.0-64-generic",
        "runtime": 12090,
        "timestamp": "2025-08-10T00:31:29Z"
    }
}
```
