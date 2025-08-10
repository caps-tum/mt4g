# NVIDIA T1000 Benchmark Report

## General

| Key | Value |
| --- | ----- |
| asicRevision | 0 |
| clockRate | 1395000 kHz |
| computeCapability | 7.5 |
| name | NVIDIA T1000 |
| vendor | NVIDIA |

## Compute

| Key | Value |
| --- | ----- |
| concurrentKernels | true |
| maxBlocksPerMultiProcessor | 16 |
| maxThreadsPerBlock | 1024 |
| maxThreadsPerMultiProcessor | 1024 |
| multiProcessorCount | 14 |
| numberOfCoresPerMultiProcessor | 64 |
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
| latency | 71.8824 cycles |
| lineSize | 64 bytes |
| missPenalty | 29.0873 cycles |
| size | 2112 bytes |

#### constant l1.5

| Key | Value |
| --- | ----- |
| fetchGranularity | 256 bytes |
| latency | 115.079 cycles |
| size | 65537 bytes |

### l1

| Key | Value |
| --- | ----- |
| amountPerMultiprocessor | 1 |
| fetchGranularity | 32 bytes |
| globalL1CacheSupported | true |
| latency | 66 cycles |
| lineSize | 128 bytes |
| localL1CacheSupported | true |
| missPenalty | 73.8196 cycles |
| sharedWith | Read Only, Texture |
| size | 60416 bytes |

### l2

| Key | Value |
| --- | ----- |
| fetchGranularity | 32 bytes |
| latency | 157.733 cycles |
| lineSize | 128 bytes |
| missPenalty | 170.302 cycles |
| persistingL2CacheMaxSize | 0 bytes |
| readBandwidth | 203.897 GiB/s |
| segmentSize | 1048576 bytes |
| size | 1048576 bytes |
| writeBandwidth | 205.035 GiB/s |

### main

| Key | Value |
| --- | ----- |
| latency | 495.902 cycles |
| memoryBusWidth | 128 bit |
| memoryClockRate | 5001000 kHz |
| readBandwidth | 139.71 GiB/s |
| totalGlobalMem | 3897229312 bytes |
| writeBandwidth | 135.263 GiB/s |

### readOnly

| Key | Value |
| --- | ----- |
| amountPerMultiprocessor | 1 |
| fetchGranularity | 32 bytes |
| latency | 34 cycles |
| lineSize | 128 bytes |
| missPenalty | 105.89 cycles |
| sharedWith | L1, Texture |
| size | 60416 bytes |

### shared

| Key | Value |
| --- | ----- |
| latency | 58 cycles |
| reservedSharedMemPerBlock | 0 bytes |
| sharedMemPerBlock | 49152 bytes |
| sharedMemPerMultiProcessor | 65536 bytes |

### texture

| Key | Value |
| --- | ----- |
| amountPerMultiprocessor | 1 |
| fetchGranularity | 32 bytes |
| latency | 38 cycles |
| lineSize | 128 bytes |
| missPenalty | 130.016 cycles |
| sharedWith | L1, Read Only |
| size | 61440 bytes |

## Graphs

![NVIDIA T1000 - Constant L1 Fetch Granularity](./NVIDIA%20T1000%20-%20Constant%20L1%20Fetch%20Granularity.png)
[Raw data](./NVIDIA%20T1000%20-%20Constant%20L1%20Fetch%20Granularity.txt)

![NVIDIA T1000 - Constant L1 Line Size](./NVIDIA%20T1000%20-%20Constant%20L1%20Line%20Size.png)
[Raw data](./NVIDIA%20T1000%20-%20Constant%20L1%20Line%20Size.txt)

![NVIDIA T1000 - Constant L1 Size](./NVIDIA%20T1000%20-%20Constant%20L1%20Size.png)
[Raw data](./NVIDIA%20T1000%20-%20Constant%20L1%20Size.txt)

![NVIDIA T1000 - Constant L1.5 Fetch Granularity](./NVIDIA%20T1000%20-%20Constant%20L1.5%20Fetch%20Granularity.png)
[Raw data](./NVIDIA%20T1000%20-%20Constant%20L1.5%20Fetch%20Granularity.txt)

![NVIDIA T1000 - Constant L1.5 Size](./NVIDIA%20T1000%20-%20Constant%20L1.5%20Size.png)
[Raw data](./NVIDIA%20T1000%20-%20Constant%20L1.5%20Size.txt)

![NVIDIA T1000 - L1 Fetch Granularity](./NVIDIA%20T1000%20-%20L1%20Fetch%20Granularity.png)
[Raw data](./NVIDIA%20T1000%20-%20L1%20Fetch%20Granularity.txt)

![NVIDIA T1000 - L1 Line Size](./NVIDIA%20T1000%20-%20L1%20Line%20Size.png)
[Raw data](./NVIDIA%20T1000%20-%20L1%20Line%20Size.txt)

![NVIDIA T1000 - L1 Size](./NVIDIA%20T1000%20-%20L1%20Size.png)
[Raw data](./NVIDIA%20T1000%20-%20L1%20Size.txt)

![NVIDIA T1000 - L2 Fetch Granularity](./NVIDIA%20T1000%20-%20L2%20Fetch%20Granularity.png)
[Raw data](./NVIDIA%20T1000%20-%20L2%20Fetch%20Granularity.txt)

![NVIDIA T1000 - L2 Line Size](./NVIDIA%20T1000%20-%20L2%20Line%20Size.png)
[Raw data](./NVIDIA%20T1000%20-%20L2%20Line%20Size.txt)

![NVIDIA T1000 - L2 Segment Size](./NVIDIA%20T1000%20-%20L2%20Segment%20Size.png)
[Raw data](./NVIDIA%20T1000%20-%20L2%20Segment%20Size.txt)

![NVIDIA T1000 - Read Only Fetch Granularity](./NVIDIA%20T1000%20-%20Read%20Only%20Fetch%20Granularity.png)
[Raw data](./NVIDIA%20T1000%20-%20Read%20Only%20Fetch%20Granularity.txt)

![NVIDIA T1000 - Read Only Line Size](./NVIDIA%20T1000%20-%20Read%20Only%20Line%20Size.png)
[Raw data](./NVIDIA%20T1000%20-%20Read%20Only%20Line%20Size.txt)

![NVIDIA T1000 - Read Only Size](./NVIDIA%20T1000%20-%20Read%20Only%20Size.png)
[Raw data](./NVIDIA%20T1000%20-%20Read%20Only%20Size.txt)

![NVIDIA T1000 - Texture Fetch Granularity](./NVIDIA%20T1000%20-%20Texture%20Fetch%20Granularity.png)
[Raw data](./NVIDIA%20T1000%20-%20Texture%20Fetch%20Granularity.txt)

![NVIDIA T1000 - Texture Line Size](./NVIDIA%20T1000%20-%20Texture%20Line%20Size.png)
[Raw data](./NVIDIA%20T1000%20-%20Texture%20Line%20Size.txt)

![NVIDIA T1000 - Texture Size](./NVIDIA%20T1000%20-%20Texture%20Size.png)
[Raw data](./NVIDIA%20T1000%20-%20Texture%20Size.txt)

## Raw JSON

```json
{
    "compute": {
        "concurrentKernels": true,
        "maxBlocksPerMultiProcessor": 16,
        "maxThreadsPerBlock": 1024,
        "maxThreadsPerMultiProcessor": 1024,
        "multiProcessorCount": 14,
        "numberOfCoresPerMultiProcessor": 64,
        "regsPerBlock": 65536,
        "regsPerMultiProcessor": 65536,
        "supportsCooperativeLaunch": true,
        "warpSize": 32
    },
    "general": {
        "asicRevision": 0,
        "clockRate": {
            "unit": "kHz",
            "value": 1395000
        },
        "computeCapability": {
            "major": 7,
            "minor": 5
        },
        "name": "NVIDIA T1000",
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
                    "mean": 71.88235294117646,
                    "measurements": 255,
                    "method": "p-chase",
                    "p50": 72.0,
                    "p95": 72.0,
                    "sampleSize": 256,
                    "stdev": 1.8786728732554483,
                    "unit": "cycles"
                },
                "lineSize": {
                    "confidence": 0.9997382901049023,
                    "method": "p-chase",
                    "randomized": false,
                    "size": 64,
                    "unit": "bytes"
                },
                "missPenalty": {
                    "unit": "cycles",
                    "value": 29.087344028520505
                },
                "size": {
                    "confidence": 0.8976020479590409,
                    "method": "p-chase",
                    "randomized": false,
                    "size": 2112,
                    "unit": "bytes"
                }
            },
            "l1.5": {
                "fetchGranularity": {
                    "confidence": 0.991500424978751,
                    "method": "p-chase",
                    "randomized": false,
                    "size": 256,
                    "unit": "bytes"
                },
                "latency": {
                    "mean": 115.07874015748031,
                    "measurements": 127,
                    "method": "p-chase",
                    "p50": 105.0,
                    "p95": 245.7,
                    "sampleSize": 256,
                    "stdev": 39.28075709256739,
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
                "confidence": 0.9700805983880323,
                "method": "p-chase",
                "randomized": false,
                "size": 32,
                "unit": "bytes"
            },
            "globalL1CacheSupported": true,
            "latency": {
                "mean": 66.0,
                "measurements": 255,
                "method": "p-chase",
                "p50": 66.0,
                "p95": 66.0,
                "sampleSize": 256,
                "stdev": 0.0,
                "unit": "cycles"
            },
            "lineSize": {
                "confidence": 0.9996224513210873,
                "method": "p-chase",
                "randomized": false,
                "size": 128,
                "unit": "bytes"
            },
            "localL1CacheSupported": true,
            "missPenalty": {
                "unit": "cycles",
                "value": 73.81960784313725
            },
            "sharedWith": [
                "Read Only",
                "Texture"
            ],
            "size": {
                "confidence": 0.9689515524223788,
                "method": "p-chase",
                "randomized": false,
                "size": 60416,
                "unit": "bytes"
            }
        },
        "l2": {
            "fetchGranularity": {
                "confidence": 0.9690606187876243,
                "method": "p-chase",
                "randomized": false,
                "size": 32,
                "unit": "bytes"
            },
            "latency": {
                "mean": 157.73333333333332,
                "measurements": 255,
                "method": "p-chase",
                "p50": 159.0,
                "p95": 163.0,
                "sampleSize": 256,
                "stdev": 4.205295998582684,
                "unit": "cycles"
            },
            "lineSize": {
                "confidence": 0.9437125868651072,
                "method": "p-chase",
                "randomized": false,
                "size": 128,
                "unit": "bytes"
            },
            "missPenalty": {
                "unit": "cycles",
                "value": 170.30196078431376
            },
            "persistingL2CacheMaxSize": {
                "unit": "bytes",
                "value": 0
            },
            "readBandwidth": {
                "unit": "GiB/s",
                "value": 203.8965431783563
            },
            "segmentSize": {
                "confidence": 0.9843207415421947,
                "method": "p-chase",
                "randomized": false,
                "size": 1048576,
                "unit": "bytes"
            },
            "size": {
                "unit": "bytes",
                "value": 1048576
            },
            "writeBandwidth": {
                "unit": "GiB/s",
                "value": 205.03526996309242
            }
        },
        "main": {
            "latency": {
                "mean": 495.90229604298975,
                "measurements": 2047,
                "method": "p-chase",
                "p50": 486.0,
                "p95": 531.0999999999995,
                "sampleSize": 2048,
                "stdev": 66.3679075244464,
                "unit": "cycles"
            },
            "memoryBusWidth": {
                "unit": "bit",
                "value": 128
            },
            "memoryClockRate": {
                "unit": "kHz",
                "value": 5001000
            },
            "readBandwidth": {
                "unit": "GiB/s",
                "value": 139.71035788232072
            },
            "totalGlobalMem": {
                "unit": "bytes",
                "value": 3897229312
            },
            "writeBandwidth": {
                "unit": "GiB/s",
                "value": 135.26267512180644
            }
        },
        "readOnly": {
            "amountPerMultiprocessor": 1,
            "fetchGranularity": {
                "confidence": 0.99910001799964,
                "method": "p-chase",
                "randomized": false,
                "size": 32,
                "unit": "bytes"
            },
            "latency": {
                "mean": 34.0,
                "measurements": 255,
                "method": "p-chase",
                "p50": 34.0,
                "p95": 34.0,
                "sampleSize": 256,
                "stdev": 0.0,
                "unit": "cycles"
            },
            "lineSize": {
                "confidence": 0.9996676633586229,
                "method": "p-chase",
                "randomized": false,
                "size": 128,
                "unit": "bytes"
            },
            "missPenalty": {
                "unit": "cycles",
                "value": 105.89019607843136
            },
            "sharedWith": [
                "L1",
                "Texture"
            ],
            "size": {
                "confidence": 0.9739013049347532,
                "method": "p-chase",
                "randomized": false,
                "size": 60416,
                "unit": "bytes"
            }
        },
        "shared": {
            "latency": {
                "mean": 58.0,
                "measurements": 255,
                "method": "p-chase",
                "p50": 58.0,
                "p95": 58.0,
                "sampleSize": 256,
                "stdev": 0.0,
                "unit": "cycles"
            },
            "reservedSharedMemPerBlock": {
                "unit": "bytes",
                "value": 0
            },
            "sharedMemPerBlock": {
                "unit": "bytes",
                "value": 49152
            },
            "sharedMemPerMultiProcessor": {
                "unit": "bytes",
                "value": 65536
            }
        },
        "texture": {
            "amountPerMultiprocessor": 1,
            "fetchGranularity": {
                "confidence": 0.999980000399992,
                "method": "p-chase",
                "randomized": false,
                "size": 32,
                "unit": "bytes"
            },
            "latency": {
                "mean": 38.0,
                "measurements": 255,
                "method": "p-chase",
                "p50": 38.0,
                "p95": 40.0,
                "sampleSize": 256,
                "stdev": 2.0,
                "unit": "cycles"
            },
            "lineSize": {
                "confidence": 0.9998408443544974,
                "method": "p-chase",
                "randomized": false,
                "size": 128,
                "unit": "bytes"
            },
            "missPenalty": {
                "unit": "cycles",
                "value": 130.01568627450982
            },
            "sharedWith": [
                "L1",
                "Read Only"
            ],
            "size": {
                "confidence": 0.9721013949302535,
                "method": "p-chase",
                "randomized": false,
                "size": 61440,
                "unit": "bytes"
            }
        }
    },
    "meta": {
        "driver": 12080,
        "gpuCompiler": "nvcc 12.9.41",
        "hostCompiler": "gcc 13.3.0",
        "hostCpu": "Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz",
        "os": "Linux 6.8.0-63-generic",
        "runtime": 12000,
        "timestamp": "2025-08-10T00:33:19Z"
    }
}
```
