/**
 * @brief Central include that aggregates all benchmark declarations.
 */
#pragma once

#include "benchmarks/base.hpp"

#include "typedef/cacheSizeResult.hpp"
#include "typedef/cacheLineSizeResult.hpp"
#include "typedef/cacheLatencyResult.hpp"
#include "typedef/vectorTypes.hpp"

#include "benchmarks/size/cacheSize/l1CacheSize.hpp"
#include "benchmarks/size/fetchGranularity/l1FetchGranularity.hpp"
#include "benchmarks/size/lineSize/l1LineSize.hpp"

#include "benchmarks/size/cacheSize/l2SegmentSize.hpp"
#include "benchmarks/size/fetchGranularity/l2FetchGranularity.hpp"
#include "benchmarks/size/lineSize/l2LineSize.hpp"

#include "benchmarks/latency/l1Latency.hpp"
#include "benchmarks/latency/l2Latency.hpp"
#include "benchmarks/latency/mainMemoryLatency.hpp"
#include "benchmarks/latency/sharedLatency.hpp"

#include "benchmarks/missPenalty/l1MissPenalty.hpp"
#include "benchmarks/missPenalty/l2MissPenalty.hpp"

#include "benchmarks/amount/l1Amount.hpp"

#include "benchmarks/bandwidth/mainMemoryReadBandwidth.hpp"
#include "benchmarks/bandwidth/mainMemoryWriteBandwidth.hpp"
#include "benchmarks/bandwidth/l2ReadBandwidth.hpp"
#include "benchmarks/bandwidth/l2ReadBandwidthVariants.hpp"
#include "benchmarks/bandwidth/l2WriteBandwidth.hpp"

#include "benchmarks/departureDelay/departureDelay.hpp"


#include "benchmarks/latency/amd_l3Latency.hpp"
#include "benchmarks/latency/amd_scalarL1Latency.hpp"

#include "benchmarks/missPenalty/amd_scalarL1MissPenalty.hpp"
#include "benchmarks/missPenalty/amd_l3MissPenalty.hpp"

#include "benchmarks/size/cacheSize/amd_l3CacheSize.hpp"
#include "benchmarks/size/fetchGranularity/amd_l3FetchGranularity.hpp"

#include "benchmarks/bandwidth/amd_l3ReadBandwidth.hpp"
#include "benchmarks/bandwidth/amd_l3WriteBandwidth.hpp"

#include "benchmarks/size/fetchGranularity/amd_scalarL1FetchGranularity.hpp"
#include "benchmarks/size/cacheSize/amd_scalarL1CacheSize.hpp"
#include "benchmarks/size/lineSize/amd_scalarL1LineSize.hpp"

#include "benchmarks/share/amd_cuShareScalarL1.hpp"


#include "benchmarks/amount/nvidia_constantL1Amount.hpp"
#include "benchmarks/amount/nvidia_readOnlyAmount.hpp"
#include "benchmarks/amount/nvidia_textureAmount.hpp"

#include "benchmarks/difference/nvidia_differenceL1L2.hpp"

#include "benchmarks/latency/nvidia_constantL1Latency.hpp"
#include "benchmarks/latency/nvidia_constantL15Latency.hpp"
#include "benchmarks/latency/nvidia_readOnlyLatency.hpp"
#include "benchmarks/latency/nvidia_textureLatency.hpp"

#include "benchmarks/share/nvidia_constantL1SharedWithL1.hpp"
#include "benchmarks/share/nvidia_readOnlySharedWithL1.hpp"
#include "benchmarks/share/nvidia_textureSharedWithReadOnly.hpp"
#include "benchmarks/share/nvidia_textureSharedWithL1.hpp"

#include "benchmarks/missPenalty/nvidia_constantL1MissPenalty.hpp"
#include "benchmarks/missPenalty/nvidia_readOnlyMissPenalty.hpp"
#include "benchmarks/missPenalty/nvidia_textureMissPenalty.hpp"

#include "benchmarks/size/cacheSize/nvidia_constantL1CacheSize.hpp"
#include "benchmarks/size/fetchGranularity/nvidia_constantL1FetchGranularity.hpp"
#include "benchmarks/size/lineSize/nvidia_constantL1LineSize.hpp"

#include "benchmarks/size/cacheSize/nvidia_constantL15CacheSize.hpp"
#include "benchmarks/size/fetchGranularity/nvidia_constantL15FetchGranularity.hpp"
#include "benchmarks/size/lineSize/nvidia_constantL15LineSize.hpp"

#include "benchmarks/size/cacheSize/nvidia_readOnlyCacheSize.hpp"
#include "benchmarks/size/fetchGranularity/nvidia_readOnlyFetchGranularity.hpp"
#include "benchmarks/size/lineSize/nvidia_readOnlyLineSize.hpp"

#include "benchmarks/size/cacheSize/nvidia_textureCacheSize.hpp"
#include "benchmarks/size/fetchGranularity/nvidia_textureFetchGranularity.hpp"
#include "benchmarks/size/lineSize/nvidia_textureLineSize.hpp"