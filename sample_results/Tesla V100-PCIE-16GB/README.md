# Tesla V100-PCIE-16GB Benchmark Report

## General

| Key | Value |
| --- | ----- |
| asicRevision | 0 |
| clockRate | 1380000 kHz |
| computeCapability | 7.0 |
| name | Tesla V100-PCIE-16GB |
| vendor | NVIDIA |

## Compute

| Key | Value |
| --- | ----- |
| concurrentKernels | true |
| maxBlocksPerMultiProcessor | 32 |
| maxThreadsPerBlock | 1024 |
| maxThreadsPerMultiProcessor | 2048 |
| multiProcessorCount | 80 |
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
| latency | 41.8902 cycles |
| lineSize | 64 bytes |
| missPenalty | 76.0189 cycles |
| size | 2112 bytes |

#### constant l1.5

| Key | Value |
| --- | ----- |
| fetchGranularity | 256 bytes |
| latency | 112.835 cycles |
| size | 65537 bytes |

### l1

| Key | Value |
| --- | ----- |
| amountPerMultiprocessor | 1 |
| fetchGranularity | 32 bytes |
| globalL1CacheSupported | true |
| latency | 30.0667 cycles |
| lineSize | 128 bytes |
| localL1CacheSupported | true |
| missPenalty | 156.051 cycles |
| sharedWith | Read Only, Texture |
| size | 117760 bytes |

### l2

| Key | Value |
| --- | ----- |
| fetchGranularity | 32 bytes |
| latency | 214.482 cycles |
| lineSize | 128 bytes |
| missPenalty | 153.541 cycles |
| persistingL2CacheMaxSize | 0 bytes |
| readBandwidth | 958.072 GiB/s |
| segmentSize | 6291456 bytes |
| size | 6291456 bytes |
| writeBandwidth | 1087.12 GiB/s |

### main

| Key | Value |
| --- | ----- |
| latency | 407.907 cycles |
| memoryBusWidth | 4096 bit |
| memoryClockRate | 877000 kHz |
| readBandwidth | 647.493 GiB/s |
| totalGlobalMem | 16928342016 bytes |
| writeBandwidth | 599.831 GiB/s |

### readOnly

| Key | Value |
| --- | ----- |
| amountPerMultiprocessor | 1 |
| fetchGranularity | 32 bytes |
| latency | 30.0275 cycles |
| lineSize | 128 bytes |
| missPenalty | 154.298 cycles |
| sharedWith | L1, Texture |
| size | 117760 bytes |

### shared

| Key | Value |
| --- | ----- |
| latency | 23.4941 cycles |
| reservedSharedMemPerBlock | 0 bytes |
| sharedMemPerBlock | 49152 bytes |
| sharedMemPerMultiProcessor | 98304 bytes |

### texture

| Key | Value |
| --- | ----- |
| amountPerMultiprocessor | 1 |
| fetchGranularity | 32 bytes |
| latency | 65.0157 cycles |
| lineSize | 128 bytes |
| missPenalty | 111.42 cycles |
| sharedWith | L1, Read Only |
| size | 115712 bytes |

## Graphs

![Tesla V100-PCIE-16GB - Constant L1 Fetch Granularity](./Tesla%20V100-PCIE-16GB%20-%20Constant%20L1%20Fetch%20Granularity.png)
[Raw data](./Tesla%20V100-PCIE-16GB%20-%20Constant%20L1%20Fetch%20Granularity.txt)

![Tesla V100-PCIE-16GB - Constant L1 Line Size](./Tesla%20V100-PCIE-16GB%20-%20Constant%20L1%20Line%20Size.png)
[Raw data](./Tesla%20V100-PCIE-16GB%20-%20Constant%20L1%20Line%20Size.txt)

![Tesla V100-PCIE-16GB - Constant L1 Size](./Tesla%20V100-PCIE-16GB%20-%20Constant%20L1%20Size.png)
[Raw data](./Tesla%20V100-PCIE-16GB%20-%20Constant%20L1%20Size.txt)

![Tesla V100-PCIE-16GB - Constant L1.5 Fetch Granularity](./Tesla%20V100-PCIE-16GB%20-%20Constant%20L1.5%20Fetch%20Granularity.png)
[Raw data](./Tesla%20V100-PCIE-16GB%20-%20Constant%20L1.5%20Fetch%20Granularity.txt)

![Tesla V100-PCIE-16GB - Constant L1.5 Size](./Tesla%20V100-PCIE-16GB%20-%20Constant%20L1.5%20Size.png)
[Raw data](./Tesla%20V100-PCIE-16GB%20-%20Constant%20L1.5%20Size.txt)

![Tesla V100-PCIE-16GB - L1 Fetch Granularity](./Tesla%20V100-PCIE-16GB%20-%20L1%20Fetch%20Granularity.png)
[Raw data](./Tesla%20V100-PCIE-16GB%20-%20L1%20Fetch%20Granularity.txt)

![Tesla V100-PCIE-16GB - L1 Line Size](./Tesla%20V100-PCIE-16GB%20-%20L1%20Line%20Size.png)
[Raw data](./Tesla%20V100-PCIE-16GB%20-%20L1%20Line%20Size.txt)

![Tesla V100-PCIE-16GB - L1 Size](./Tesla%20V100-PCIE-16GB%20-%20L1%20Size.png)
[Raw data](./Tesla%20V100-PCIE-16GB%20-%20L1%20Size.txt)

![Tesla V100-PCIE-16GB - L2 Fetch Granularity](./Tesla%20V100-PCIE-16GB%20-%20L2%20Fetch%20Granularity.png)
[Raw data](./Tesla%20V100-PCIE-16GB%20-%20L2%20Fetch%20Granularity.txt)

![Tesla V100-PCIE-16GB - L2 Line Size](./Tesla%20V100-PCIE-16GB%20-%20L2%20Line%20Size.png)
[Raw data](./Tesla%20V100-PCIE-16GB%20-%20L2%20Line%20Size.txt)

![Tesla V100-PCIE-16GB - L2 Segment Size](./Tesla%20V100-PCIE-16GB%20-%20L2%20Segment%20Size.png)
[Raw data](./Tesla%20V100-PCIE-16GB%20-%20L2%20Segment%20Size.txt)

![Tesla V100-PCIE-16GB - Read Only Fetch Granularity](./Tesla%20V100-PCIE-16GB%20-%20Read%20Only%20Fetch%20Granularity.png)
[Raw data](./Tesla%20V100-PCIE-16GB%20-%20Read%20Only%20Fetch%20Granularity.txt)

![Tesla V100-PCIE-16GB - Read Only Line Size](./Tesla%20V100-PCIE-16GB%20-%20Read%20Only%20Line%20Size.png)
[Raw data](./Tesla%20V100-PCIE-16GB%20-%20Read%20Only%20Line%20Size.txt)

![Tesla V100-PCIE-16GB - Read Only Size](./Tesla%20V100-PCIE-16GB%20-%20Read%20Only%20Size.png)
[Raw data](./Tesla%20V100-PCIE-16GB%20-%20Read%20Only%20Size.txt)

![Tesla V100-PCIE-16GB - Texture Fetch Granularity](./Tesla%20V100-PCIE-16GB%20-%20Texture%20Fetch%20Granularity.png)
[Raw data](./Tesla%20V100-PCIE-16GB%20-%20Texture%20Fetch%20Granularity.txt)

![Tesla V100-PCIE-16GB - Texture Line Size](./Tesla%20V100-PCIE-16GB%20-%20Texture%20Line%20Size.png)
[Raw data](./Tesla%20V100-PCIE-16GB%20-%20Texture%20Line%20Size.txt)

![Tesla V100-PCIE-16GB - Texture Size](./Tesla%20V100-PCIE-16GB%20-%20Texture%20Size.png)
[Raw data](./Tesla%20V100-PCIE-16GB%20-%20Texture%20Size.txt)

## Raw JSON

```json
{
    "compute": {
        "concurrentKernels": true,
        "maxBlocksPerMultiProcessor": 32,
        "maxThreadsPerBlock": 1024,
        "maxThreadsPerMultiProcessor": 2048,
        "multiProcessorCount": 80,
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
            "value": 1380000
        },
        "computeCapability": {
            "major": 7,
            "minor": 0
        },
        "name": "Tesla V100-PCIE-16GB",
        "vendor": "NVIDIA"
    },
    "memory": {
        "constant": {
            "l1": {
                "amountPerMultiprocessor": 1,
                "fetchGranularity": {
                    "confidence": 0.998240035199296,
                    "method": "p-chase",
                    "randomized": false,
                    "size": 64,
                    "unit": "bytes"
                },
                "latency": {
                    "mean": 41.89019607843137,
                    "measurements": 255,
                    "method": "p-chase",
                    "p50": 40.0,
                    "p95": 44.0,
                    "sampleSize": 256,
                    "stdev": 2.659795068030917,
                    "unit": "cycles"
                },
                "lineSize": {
                    "confidence": 0.9997978656214926,
                    "method": "p-chase",
                    "randomized": false,
                    "size": 64,
                    "unit": "bytes"
                },
                "missPenalty": {
                    "unit": "cycles",
                    "value": 76.01889483065953
                },
                "size": {
                    "confidence": 0.8994220115597689,
                    "method": "p-chase",
                    "randomized": false,
                    "size": 2112,
                    "unit": "bytes"
                }
            },
            "l1.5": {
                "fetchGranularity": {
                    "confidence": 0.999950002499875,
                    "method": "p-chase",
                    "randomized": false,
                    "size": 256,
                    "unit": "bytes"
                },
                "latency": {
                    "mean": 112.83464566929133,
                    "measurements": 127,
                    "method": "p-chase",
                    "p50": 114.0,
                    "p95": 114.0,
                    "sampleSize": 256,
                    "stdev": 9.249421929193614,
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
                "mean": 30.066666666666666,
                "measurements": 255,
                "method": "p-chase",
                "p50": 30.0,
                "p95": 30.0,
                "sampleSize": 256,
                "stdev": 0.6207268620499207,
                "unit": "cycles"
            },
            "lineSize": {
                "confidence": 0.9960017114493348,
                "method": "p-chase",
                "randomized": false,
                "size": 128,
                "unit": "bytes"
            },
            "localL1CacheSupported": true,
            "missPenalty": {
                "unit": "cycles",
                "value": 156.05098039215687
            },
            "sharedWith": [
                "Read Only",
                "Texture"
            ],
            "size": {
                "confidence": 0.9735513224338783,
                "method": "p-chase",
                "randomized": false,
                "size": 117760,
                "unit": "bytes"
            }
        },
        "l2": {
            "fetchGranularity": {
                "confidence": 0.998960020799584,
                "method": "p-chase",
                "randomized": false,
                "size": 32,
                "unit": "bytes"
            },
            "latency": {
                "mean": 214.48235294117646,
                "measurements": 255,
                "method": "p-chase",
                "p50": 219.0,
                "p95": 228.0,
                "sampleSize": 256,
                "stdev": 10.114632455876087,
                "unit": "cycles"
            },
            "lineSize": {
                "confidence": 0.8325978852414344,
                "method": "p-chase",
                "randomized": false,
                "size": 128,
                "unit": "bytes"
            },
            "missPenalty": {
                "unit": "cycles",
                "value": 153.54117647058823
            },
            "persistingL2CacheMaxSize": {
                "unit": "bytes",
                "value": 0
            },
            "readBandwidth": {
                "unit": "GiB/s",
                "value": 958.0717081829157
            },
            "segmentSize": {
                "confidence": 0.991969657641477,
                "method": "p-chase",
                "randomized": false,
                "size": 6291456,
                "unit": "bytes"
            },
            "size": {
                "unit": "bytes",
                "value": 6291456
            },
            "writeBandwidth": {
                "unit": "GiB/s",
                "value": 1087.122895614547
            }
        },
        "main": {
            "latency": {
                "mean": 407.90718124084026,
                "measurements": 2047,
                "method": "p-chase",
                "p50": 395.0,
                "p95": 495.6999999999998,
                "sampleSize": 2048,
                "stdev": 77.60733545446575,
                "unit": "cycles"
            },
            "memoryBusWidth": {
                "unit": "bit",
                "value": 4096
            },
            "memoryClockRate": {
                "unit": "kHz",
                "value": 877000
            },
            "readBandwidth": {
                "unit": "GiB/s",
                "value": 647.493473710986
            },
            "totalGlobalMem": {
                "unit": "bytes",
                "value": 16928342016
            },
            "writeBandwidth": {
                "unit": "GiB/s",
                "value": 599.8307137308811
            }
        },
        "readOnly": {
            "amountPerMultiprocessor": 1,
            "fetchGranularity": {
                "confidence": 0.9691206175876482,
                "method": "p-chase",
                "randomized": false,
                "size": 32,
                "unit": "bytes"
            },
            "latency": {
                "mean": 30.027450980392157,
                "measurements": 255,
                "method": "p-chase",
                "p50": 30.0,
                "p95": 30.0,
                "sampleSize": 256,
                "stdev": 0.4383570037596046,
                "unit": "cycles"
            },
            "lineSize": {
                "confidence": 0.9998531691267499,
                "method": "p-chase",
                "randomized": false,
                "size": 128,
                "unit": "bytes"
            },
            "missPenalty": {
                "unit": "cycles",
                "value": 154.29803921568626
            },
            "sharedWith": [
                "L1",
                "Texture"
            ],
            "size": {
                "confidence": 0.9720513974301285,
                "method": "p-chase",
                "randomized": false,
                "size": 117760,
                "unit": "bytes"
            }
        },
        "shared": {
            "latency": {
                "mean": 23.494117647058822,
                "measurements": 255,
                "method": "p-chase",
                "p50": 23.0,
                "p95": 27.0,
                "sampleSize": 256,
                "stdev": 4.224852034156288,
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
            "amountPerMultiprocessor": 1,
            "fetchGranularity": {
                "confidence": 0.999960000799984,
                "method": "p-chase",
                "randomized": false,
                "size": 32,
                "unit": "bytes"
            },
            "latency": {
                "mean": 65.0156862745098,
                "measurements": 255,
                "method": "p-chase",
                "p50": 68.0,
                "p95": 69.0,
                "sampleSize": 256,
                "stdev": 4.000953259887321,
                "unit": "cycles"
            },
            "lineSize": {
                "confidence": 0.9999568093044514,
                "method": "p-chase",
                "randomized": false,
                "size": 128,
                "unit": "bytes"
            },
            "missPenalty": {
                "unit": "cycles",
                "value": 111.41960784313726
            },
            "sharedWith": [
                "L1",
                "Read Only"
            ],
            "size": {
                "confidence": 0.9730513474326283,
                "method": "p-chase",
                "randomized": false,
                "size": 115712,
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
        "timestamp": "2025-08-10T10:01:37Z"
    }
}
```
