# Quadro P6000 Benchmark Report

## General

| Key | Value |
| --- | ----- |
| asicRevision | 0 |
| clockRate | 1645000 kHz |
| computeCapability | 6.1 |
| name | Quadro P6000 |
| vendor | NVIDIA |

## Compute

| Key | Value |
| --- | ----- |
| concurrentKernels | true |
| maxBlocksPerMultiProcessor | 32 |
| maxThreadsPerBlock | 1024 |
| maxThreadsPerMultiProcessor | 2048 |
| multiProcessorCount | 30 |
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
| latency | 62.9098 cycles |
| lineSize | 64 bytes |
| missPenalty | 46.0599 cycles |
| sharedWith | L1 |
| size | 2112 bytes |

#### constant l1.5

| Key | Value |
| --- | ----- |
| fetchGranularity | 256 bytes |
| latency | 168.528 cycles |
| lineSize | 256 bytes |
| size | 31488 bytes |

### l1

| Key | Value |
| --- | ----- |
| fetchGranularity | 32 bytes |
| globalL1CacheSupported | true |
| latency | 106.4 cycles |
| lineSize | 128 bytes |
| localL1CacheSupported | true |
| missPenalty | 77.4 cycles |
| sharedWith | Constant L1, Read Only, Texture |
| size | 24576 bytes |

### l2

| Key | Value |
| --- | ----- |
| fetchGranularity | 32 bytes |
| latency | 226.922 cycles |
| lineSize | 128 bytes |
| missPenalty | 197.922 cycles |
| persistingL2CacheMaxSize | 0 bytes |
| readBandwidth | 789.609 GiB/s |
| segmentSize | 3145728 bytes |
| size | 3145728 bytes |
| writeBandwidth | 601.863 GiB/s |

### main

| Key | Value |
| --- | ----- |
| latency | 415.291 cycles |
| memoryBusWidth | 384 bit |
| memoryClockRate | 4513000 kHz |
| readBandwidth | 365.374 GiB/s |
| totalGlobalMem | 25623527424 bytes |
| writeBandwidth | 377.839 GiB/s |

### readOnly

| Key | Value |
| --- | ----- |
| amountPerMultiprocessor | 2 |
| fetchGranularity | 32 bytes |
| latency | 135.988 cycles |
| lineSize | 128 bytes |
| missPenalty | 106.988 cycles |
| sharedWith | L1, Texture |
| size | 24576 bytes |

### shared

| Key | Value |
| --- | ----- |
| latency | 39.6118 cycles |
| reservedSharedMemPerBlock | 0 bytes |
| sharedMemPerBlock | 49152 bytes |
| sharedMemPerMultiProcessor | 98304 bytes |

### texture

| Key | Value |
| --- | ----- |
| amountPerMultiprocessor | 2 |
| fetchGranularity | 32 bytes |
| latency | 67.7216 cycles |
| lineSize | 128 bytes |
| missPenalty | 72.336 cycles |
| sharedWith | L1, Read Only |
| size | 24576 bytes |

## Graphs

![Quadro P6000 - Constant L1 Fetch Granularity](./Quadro%20P6000%20-%20Constant%20L1%20Fetch%20Granularity.png)
[Raw data](./Quadro%20P6000%20-%20Constant%20L1%20Fetch%20Granularity.txt)

![Quadro P6000 - Constant L1 Line Size](./Quadro%20P6000%20-%20Constant%20L1%20Line%20Size.png)
[Raw data](./Quadro%20P6000%20-%20Constant%20L1%20Line%20Size.txt)

![Quadro P6000 - Constant L1 Size](./Quadro%20P6000%20-%20Constant%20L1%20Size.png)
[Raw data](./Quadro%20P6000%20-%20Constant%20L1%20Size.txt)

![Quadro P6000 - Constant L1.5 Fetch Granularity](./Quadro%20P6000%20-%20Constant%20L1.5%20Fetch%20Granularity.png)
[Raw data](./Quadro%20P6000%20-%20Constant%20L1.5%20Fetch%20Granularity.txt)

![Quadro P6000 - Constant L1.5 Line Size](./Quadro%20P6000%20-%20Constant%20L1.5%20Line%20Size.png)
[Raw data](./Quadro%20P6000%20-%20Constant%20L1.5%20Line%20Size.txt)

![Quadro P6000 - Constant L1.5 Size](./Quadro%20P6000%20-%20Constant%20L1.5%20Size.png)
[Raw data](./Quadro%20P6000%20-%20Constant%20L1.5%20Size.txt)

![Quadro P6000 - L1 Fetch Granularity](./Quadro%20P6000%20-%20L1%20Fetch%20Granularity.png)
[Raw data](./Quadro%20P6000%20-%20L1%20Fetch%20Granularity.txt)

![Quadro P6000 - L1 Line Size](./Quadro%20P6000%20-%20L1%20Line%20Size.png)
[Raw data](./Quadro%20P6000%20-%20L1%20Line%20Size.txt)

![Quadro P6000 - L1 Size](./Quadro%20P6000%20-%20L1%20Size.png)
[Raw data](./Quadro%20P6000%20-%20L1%20Size.txt)

![Quadro P6000 - L2 Fetch Granularity](./Quadro%20P6000%20-%20L2%20Fetch%20Granularity.png)
[Raw data](./Quadro%20P6000%20-%20L2%20Fetch%20Granularity.txt)

![Quadro P6000 - L2 Line Size](./Quadro%20P6000%20-%20L2%20Line%20Size.png)
[Raw data](./Quadro%20P6000%20-%20L2%20Line%20Size.txt)

![Quadro P6000 - L2 Segment Size](./Quadro%20P6000%20-%20L2%20Segment%20Size.png)
[Raw data](./Quadro%20P6000%20-%20L2%20Segment%20Size.txt)

![Quadro P6000 - Read Only Fetch Granularity](./Quadro%20P6000%20-%20Read%20Only%20Fetch%20Granularity.png)
[Raw data](./Quadro%20P6000%20-%20Read%20Only%20Fetch%20Granularity.txt)

![Quadro P6000 - Read Only Line Size](./Quadro%20P6000%20-%20Read%20Only%20Line%20Size.png)
[Raw data](./Quadro%20P6000%20-%20Read%20Only%20Line%20Size.txt)

![Quadro P6000 - Read Only Size](./Quadro%20P6000%20-%20Read%20Only%20Size.png)
[Raw data](./Quadro%20P6000%20-%20Read%20Only%20Size.txt)

![Quadro P6000 - Texture Fetch Granularity](./Quadro%20P6000%20-%20Texture%20Fetch%20Granularity.png)
[Raw data](./Quadro%20P6000%20-%20Texture%20Fetch%20Granularity.txt)

![Quadro P6000 - Texture Line Size](./Quadro%20P6000%20-%20Texture%20Line%20Size.png)
[Raw data](./Quadro%20P6000%20-%20Texture%20Line%20Size.txt)

![Quadro P6000 - Texture Size](./Quadro%20P6000%20-%20Texture%20Size.png)
[Raw data](./Quadro%20P6000%20-%20Texture%20Size.txt)

## Raw JSON

```json
{
    "compute": {
        "concurrentKernels": true,
        "maxBlocksPerMultiProcessor": 32,
        "maxThreadsPerBlock": 1024,
        "maxThreadsPerMultiProcessor": 2048,
        "multiProcessorCount": 30,
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
            "value": 1645000
        },
        "computeCapability": {
            "major": 6,
            "minor": 1
        },
        "name": "Quadro P6000",
        "vendor": "NVIDIA"
    },
    "memory": {
        "constant": {
            "l1": {
                "amountPerMultiprocessor": 1,
                "fetchGranularity": {
                    "confidence": 0.999760004799904,
                    "method": "p-chase",
                    "randomized": false,
                    "size": 64,
                    "unit": "bytes"
                },
                "latency": {
                    "mean": 62.909803921568624,
                    "measurements": 255,
                    "method": "p-chase",
                    "p50": 49.0,
                    "p95": 171.7999999999999,
                    "sampleSize": 256,
                    "stdev": 36.61401809181208,
                    "unit": "cycles"
                },
                "lineSize": {
                    "confidence": 0.9994330843882612,
                    "method": "p-chase",
                    "randomized": false,
                    "size": 64,
                    "unit": "bytes"
                },
                "missPenalty": {
                    "unit": "cycles",
                    "value": 46.059893048128345
                },
                "sharedWith": [
                    "L1"
                ],
                "size": {
                    "confidence": 0.8994820103597928,
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
                    "mean": 168.5275590551181,
                    "measurements": 127,
                    "method": "p-chase",
                    "p50": 151.0,
                    "p95": 307.0,
                    "sampleSize": 256,
                    "stdev": 51.768996478350324,
                    "unit": "cycles"
                },
                "lineSize": {
                    "confidence": 0.9992845896493242,
                    "method": "p-chase",
                    "randomized": false,
                    "size": 256,
                    "unit": "bytes"
                },
                "size": {
                    "confidence": 0.8578828423431532,
                    "method": "p-chase",
                    "randomized": false,
                    "size": 31488,
                    "unit": "bytes"
                }
            },
            "totalConstMem": {
                "unit": "bytes",
                "value": 65536
            }
        },
        "l1": {
            "fetchGranularity": {
                "confidence": 0.999060018799624,
                "method": "p-chase",
                "randomized": false,
                "size": 32,
                "unit": "bytes"
            },
            "globalL1CacheSupported": true,
            "latency": {
                "mean": 106.4,
                "measurements": 255,
                "method": "p-chase",
                "p50": 90.0,
                "p95": 184.0,
                "sampleSize": 256,
                "stdev": 34.44904944804742,
                "unit": "cycles"
            },
            "lineSize": {
                "confidence": 0.9976305445186282,
                "method": "p-chase",
                "randomized": false,
                "size": 128,
                "unit": "bytes"
            },
            "localL1CacheSupported": true,
            "missPenalty": {
                "unit": "cycles",
                "value": 77.4
            },
            "sharedWith": [
                "Constant L1",
                "Read Only",
                "Texture"
            ],
            "size": {
                "confidence": 0.9735513224338783,
                "method": "p-chase",
                "randomized": false,
                "size": 24576,
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
                "mean": 226.92156862745097,
                "measurements": 255,
                "method": "p-chase",
                "p50": 232.0,
                "p95": 239.0,
                "sampleSize": 256,
                "stdev": 7.949751487362537,
                "unit": "cycles"
            },
            "lineSize": {
                "confidence": 0.9901896117944001,
                "method": "p-chase",
                "randomized": false,
                "size": 128,
                "unit": "bytes"
            },
            "missPenalty": {
                "unit": "cycles",
                "value": 197.92156862745097
            },
            "persistingL2CacheMaxSize": {
                "unit": "bytes",
                "value": 0
            },
            "readBandwidth": {
                "unit": "GiB/s",
                "value": 789.6093138530716
            },
            "segmentSize": {
                "confidence": 0.9573639778341012,
                "method": "p-chase",
                "randomized": false,
                "size": 3145728,
                "unit": "bytes"
            },
            "size": {
                "unit": "bytes",
                "value": 3145728
            },
            "writeBandwidth": {
                "unit": "GiB/s",
                "value": 601.8627982381842
            }
        },
        "main": {
            "latency": {
                "mean": 415.2906692721055,
                "measurements": 2047,
                "method": "p-chase",
                "p50": 413.0,
                "p95": 443.0,
                "sampleSize": 2048,
                "stdev": 75.24004612863064,
                "unit": "cycles"
            },
            "memoryBusWidth": {
                "unit": "bit",
                "value": 384
            },
            "memoryClockRate": {
                "unit": "kHz",
                "value": 4513000
            },
            "readBandwidth": {
                "unit": "GiB/s",
                "value": 365.37378666378993
            },
            "totalGlobalMem": {
                "unit": "bytes",
                "value": 25623527424
            },
            "writeBandwidth": {
                "unit": "GiB/s",
                "value": 377.83854365982234
            }
        },
        "readOnly": {
            "amountPerMultiprocessor": 2,
            "fetchGranularity": {
                "confidence": 0.999980000399992,
                "method": "p-chase",
                "randomized": false,
                "size": 32,
                "unit": "bytes"
            },
            "latency": {
                "mean": 135.98823529411766,
                "measurements": 255,
                "method": "p-chase",
                "p50": 133.0,
                "p95": 135.0,
                "sampleSize": 256,
                "stdev": 16.21180729260163,
                "unit": "cycles"
            },
            "lineSize": {
                "confidence": 0.9954946856306389,
                "method": "p-chase",
                "randomized": false,
                "size": 128,
                "unit": "bytes"
            },
            "missPenalty": {
                "unit": "cycles",
                "value": 106.98823529411766
            },
            "sharedWith": [
                "L1",
                "Texture"
            ],
            "size": {
                "confidence": 0.9732013399330034,
                "method": "p-chase",
                "randomized": false,
                "size": 24576,
                "unit": "bytes"
            }
        },
        "shared": {
            "latency": {
                "mean": 39.61176470588235,
                "measurements": 255,
                "method": "p-chase",
                "p50": 38.0,
                "p95": 39.0,
                "sampleSize": 256,
                "stdev": 12.966729792789813,
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
                "value": 98304
            }
        },
        "texture": {
            "amountPerMultiprocessor": 2,
            "fetchGranularity": {
                "confidence": 0.9683206335873282,
                "method": "p-chase",
                "randomized": false,
                "size": 32,
                "unit": "bytes"
            },
            "latency": {
                "mean": 67.72156862745098,
                "measurements": 255,
                "method": "p-chase",
                "p50": 67.0,
                "p95": 72.0,
                "sampleSize": 256,
                "stdev": 6.048762277423011,
                "unit": "cycles"
            },
            "lineSize": {
                "confidence": 0.9972849304473522,
                "method": "p-chase",
                "randomized": false,
                "size": 128,
                "unit": "bytes"
            },
            "missPenalty": {
                "unit": "cycles",
                "value": 72.33602299558567
            },
            "sharedWith": [
                "L1",
                "Read Only"
            ],
            "size": {
                "confidence": 0.9732013399330034,
                "method": "p-chase",
                "randomized": false,
                "size": 24576,
                "unit": "bytes"
            }
        }
    },
    "meta": {
        "driver": 12080,
        "gpuCompiler": "nvcc 12.8.61",
        "hostCompiler": "gcc 11.4.0",
        "hostCpu": "Intel(R) Xeon(R) Gold 6238 CPU @ 2.10GHz",
        "os": "Linux 6.8.0-52-generic",
        "runtime": 12080,
        "timestamp": "2025-08-10T09:42:18Z"
    }
}
```
