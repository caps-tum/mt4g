# NVIDIA A100-PCIE-40GB Benchmark Report

## General

| Key | Value |
| --- | ----- |
| asicRevision | 0 |
| clockRate | 1410000 kHz |
| computeCapability | 8.0 |
| name | NVIDIA A100-PCIE-40GB |
| vendor | NVIDIA |

## Compute

| Key | Value |
| --- | ----- |
| concurrentKernels | true |
| maxBlocksPerMultiProcessor | 32 |
| maxThreadsPerBlock | 1024 |
| maxThreadsPerMultiProcessor | 2048 |
| multiProcessorCount | 108 |
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
| latency | 74.8745 cycles |
| lineSize | 64 bytes |
| missPenalty | 43.0952 cycles |
| size | 2112 bytes |

#### constant l1.5

| Key | Value |
| --- | ----- |
| fetchGranularity | 256 bytes |
| latency | 120.74 cycles |
| size | 65537 bytes |

### l1

| Key | Value |
| --- | ----- |
| amountPerMultiprocessor | 1 |
| fetchGranularity | 32 bytes |
| globalL1CacheSupported | true |
| latency | 68 cycles |
| lineSize | 128 bytes |
| localL1CacheSupported | true |
| missPenalty | 70.0667 cycles |
| sharedWith | Texture |
| size | 176128 bytes |

### l2

| Key | Value |
| --- | ----- |
| fetchGranularity | 32 bytes |
| latency | 220.953 cycles |
| lineSize | 128 bytes |
| missPenalty | 265.71 cycles |
| persistingL2CacheMaxSize | 26214400 bytes |
| readBandwidth | 2374.45 GiB/s |
| segmentSize | 20971520 bytes |
| size | 41943040 bytes |
| writeBandwidth | 2093.83 GiB/s |

### main

| Key | Value |
| --- | ----- |
| latency | 640.112 cycles |
| memoryBusWidth | 5120 bit |
| memoryClockRate | 1215000 kHz |
| readBandwidth | 1098.21 GiB/s |
| totalGlobalMem | 42406903808 bytes |
| writeBandwidth | 1163.44 GiB/s |

### readOnly

| Key | Value |
| --- | ----- |
| amountPerMultiprocessor | 1 |
| fetchGranularity | 32 bytes |
| latency | 35 cycles |
| lineSize | 128 bytes |
| missPenalty | 103.012 cycles |
| sharedWith | Texture |
| size | 176128 bytes |

### shared

| Key | Value |
| --- | ----- |
| latency | 60 cycles |
| reservedSharedMemPerBlock | 1024 bytes |
| sharedMemPerBlock | 49152 bytes |
| sharedMemPerMultiProcessor | 167936 bytes |

### texture

| Key | Value |
| --- | ----- |
| amountPerMultiprocessor | 1 |
| fetchGranularity | 32 bytes |
| latency | 39.498 cycles |
| lineSize | 128 bytes |
| missPenalty | 135.161 cycles |
| sharedWith | L1, Read Only |
| size | 176128 bytes |

## Raw JSON

```json
{
    "compute": {
        "concurrentKernels": true,
        "maxBlocksPerMultiProcessor": 32,
        "maxThreadsPerBlock": 1024,
        "maxThreadsPerMultiProcessor": 2048,
        "multiProcessorCount": 108,
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
            "value": 1410000
        },
        "computeCapability": {
            "major": 8,
            "minor": 0
        },
        "name": "NVIDIA A100-PCIE-40GB",
        "vendor": "NVIDIA"
    },
    "memory": {
        "constant": {
            "l1": {
                "amountPerMultiprocessor": 1,
                "fetchGranularity": {
                    "confidence": 0.999860002799944,
                    "method": "p-chase",
                    "randomized": false,
                    "size": 64,
                    "unit": "bytes"
                },
                "latency": {
                    "mean": 74.87450980392157,
                    "measurements": 255,
                    "method": "p-chase",
                    "p50": 75.0,
                    "p95": 75.0,
                    "sampleSize": 256,
                    "stdev": 2.0039177314724785,
                    "unit": "cycles"
                },
                "lineSize": {
                    "confidence": 0.9998272152095036,
                    "method": "p-chase",
                    "randomized": false,
                    "size": 64,
                    "unit": "bytes"
                },
                "missPenalty": {
                    "unit": "cycles",
                    "value": 43.0951871657754
                },
                "size": {
                    "confidence": 0.8987020259594808,
                    "method": "p-chase",
                    "randomized": false,
                    "size": 2112,
                    "unit": "bytes"
                }
            },
            "l1.5": {
                "fetchGranularity": {
                    "confidence": 0.9916004199790011,
                    "method": "p-chase",
                    "randomized": false,
                    "size": 256,
                    "unit": "bytes"
                },
                "latency": {
                    "mean": 120.74015748031496,
                    "measurements": 127,
                    "method": "p-chase",
                    "p50": 122.0,
                    "p95": 122.0,
                    "sampleSize": 256,
                    "stdev": 9.999375058587617,
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
                "mean": 68.0,
                "measurements": 255,
                "method": "p-chase",
                "p50": 68.0,
                "p95": 68.0,
                "sampleSize": 256,
                "stdev": 0.0,
                "unit": "cycles"
            },
            "lineSize": {
                "confidence": 0.9970289255617286,
                "method": "p-chase",
                "randomized": false,
                "size": 128,
                "unit": "bytes"
            },
            "localL1CacheSupported": true,
            "missPenalty": {
                "unit": "cycles",
                "value": 70.06666666666666
            },
            "sharedWith": [
                "Texture"
            ],
            "size": {
                "confidence": 0.9705014749262537,
                "method": "p-chase",
                "randomized": false,
                "size": 176128,
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
                "mean": 220.95294117647057,
                "measurements": 255,
                "method": "p-chase",
                "p50": 169.0,
                "p95": 303.29999999999995,
                "sampleSize": 256,
                "stdev": 64.76807306504902,
                "unit": "cycles"
            },
            "lineSize": {
                "confidence": 0.9737510676066637,
                "method": "p-chase",
                "randomized": false,
                "size": 128,
                "unit": "bytes"
            },
            "missPenalty": {
                "unit": "cycles",
                "value": 265.70980392156866
            },
            "persistingL2CacheMaxSize": {
                "unit": "bytes",
                "value": 26214400
            },
            "readBandwidth": {
                "unit": "GiB/s",
                "value": 2374.446861630516
            },
            "segmentSize": {
                "confidence": 0.9158291083145481,
                "method": "p-chase",
                "randomized": false,
                "size": 20971520,
                "unit": "bytes"
            },
            "size": {
                "unit": "bytes",
                "value": 41943040
            },
            "writeBandwidth": {
                "unit": "GiB/s",
                "value": 2093.8304102520883
            }
        },
        "main": {
            "latency": {
                "mean": 640.1118710307768,
                "measurements": 2047,
                "method": "p-chase",
                "p50": 671.0,
                "p95": 799.6999999999998,
                "sampleSize": 2048,
                "stdev": 118.11250452466896,
                "unit": "cycles"
            },
            "memoryBusWidth": {
                "unit": "bit",
                "value": 5120
            },
            "memoryClockRate": {
                "unit": "kHz",
                "value": 1215000
            },
            "readBandwidth": {
                "unit": "GiB/s",
                "value": 1098.2145096543795
            },
            "totalGlobalMem": {
                "unit": "bytes",
                "value": 42406903808
            },
            "writeBandwidth": {
                "unit": "GiB/s",
                "value": 1163.4365976853953
            }
        },
        "readOnly": {
            "amountPerMultiprocessor": 1,
            "fetchGranularity": {
                "confidence": 0.999960000799984,
                "method": "p-chase",
                "randomized": false,
                "size": 32,
                "unit": "bytes"
            },
            "latency": {
                "mean": 35.0,
                "measurements": 255,
                "method": "p-chase",
                "p50": 35.0,
                "p95": 35.0,
                "sampleSize": 256,
                "stdev": 0.0,
                "unit": "cycles"
            },
            "lineSize": {
                "confidence": 0.9998479842960238,
                "method": "p-chase",
                "randomized": false,
                "size": 128,
                "unit": "bytes"
            },
            "missPenalty": {
                "unit": "cycles",
                "value": 103.01176470588234
            },
            "sharedWith": [
                "Texture"
            ],
            "size": {
                "confidence": 0.9724513774311284,
                "method": "p-chase",
                "randomized": false,
                "size": 176128,
                "unit": "bytes"
            }
        },
        "shared": {
            "latency": {
                "mean": 60.0,
                "measurements": 255,
                "method": "p-chase",
                "p50": 60.0,
                "p95": 60.0,
                "sampleSize": 256,
                "stdev": 0.0,
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
                "value": 167936
            }
        },
        "texture": {
            "amountPerMultiprocessor": 1,
            "fetchGranularity": {
                "confidence": 0.999940001199976,
                "method": "p-chase",
                "randomized": false,
                "size": 32,
                "unit": "bytes"
            },
            "latency": {
                "mean": 39.498039215686276,
                "measurements": 255,
                "method": "p-chase",
                "p50": 39.0,
                "p95": 42.0,
                "sampleSize": 256,
                "stdev": 2.5001960707426245,
                "unit": "cycles"
            },
            "lineSize": {
                "confidence": 0.999976002261542,
                "method": "p-chase",
                "randomized": false,
                "size": 128,
                "unit": "bytes"
            },
            "missPenalty": {
                "unit": "cycles",
                "value": 135.1607843137255
            },
            "sharedWith": [
                "L1",
                "Read Only"
            ],
            "size": {
                "confidence": 0.9722013899305034,
                "method": "p-chase",
                "randomized": false,
                "size": 176128,
                "unit": "bytes"
            }
        }
    },
    "meta": {
        "driver": 12080,
        "gpuCompiler": "nvcc 12.9.41",
        "hostCompiler": "gcc 13.3.0",
        "hostCpu": "AMD Ryzen Threadripper PRO 3955WX 16-Cores",
        "os": "Linux 6.8.0-64-generic",
        "runtime": 12000,
        "timestamp": "2025-08-10T00:50:56Z"
    }
}
```
