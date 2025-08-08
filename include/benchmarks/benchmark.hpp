/**
 * @brief Central include that aggregates all benchmark declarations.
 */
#pragma once

#include "benchmarks/base.hpp"

#include "typedef/cacheSizeResult.hpp"
#include "typedef/cacheLineSizeResult.hpp"
#include "typedef/cacheLatencyResult.hpp"
#include "typedef/vectorTypes.hpp"

#include "benchmarks/shared/size/cacheSize/l1CacheSize.hpp"
#include "benchmarks/shared/size/fetchGranularity/l1FetchGranularity.hpp"
#include "benchmarks/shared/size/lineSize/l1LineSize.hpp"

#include "benchmarks/shared/size/cacheSize/l2SegmentSize.hpp"
#include "benchmarks/shared/size/fetchGranularity/l2FetchGranularity.hpp"
#include "benchmarks/shared/size/lineSize/l2LineSize.hpp"

#include "benchmarks/shared/latency/l1Latency.hpp"
#include "benchmarks/shared/latency/l2Latency.hpp"
#include "benchmarks/shared/latency/mainMemoryLatency.hpp"
#include "benchmarks/shared/latency/sharedLatency.hpp"

#include "benchmarks/shared/missPenalty/l1MissPenalty.hpp"
#include "benchmarks/shared/missPenalty/l2MissPenalty.hpp"

#include "benchmarks/shared/amount/l1Amount.hpp"

#include "benchmarks/shared/bandwidth/mainMemoryReadBandwidth.hpp"
#include "benchmarks/shared/bandwidth/mainMemoryWriteBandwidth.hpp"
#include "benchmarks/shared/bandwidth/l2ReadBandwidth.hpp"
#include "benchmarks/shared/bandwidth/l2WriteBandwidth.hpp"

#include "benchmarks/shared/departureDelay/departureDelay.hpp"


#include "benchmarks/amd/latency/l3Latency.hpp"
#include "benchmarks/amd/latency/scalarL1Latency.hpp"

#include "benchmarks/amd/missPenalty/scalarL1MissPenalty.hpp"
#include "benchmarks/amd/missPenalty/l3MissPenalty.hpp"

#include "benchmarks/amd/size/cacheSize/l3CacheSize.hpp"
#include "benchmarks/amd/size/fetchGranularity/l3FetchGranularity.hpp"

#include "benchmarks/amd/bandwidth/l3ReadBandwidth.hpp"
#include "benchmarks/amd/bandwidth/l3WriteBandwidth.hpp"

#include "benchmarks/amd/size/fetchGranularity/scalarL1FetchGranularity.hpp"
#include "benchmarks/amd/size/cacheSize/scalarL1CacheSize.hpp"
#include "benchmarks/amd/size/lineSize/scalarL1LineSize.hpp"

#include "benchmarks/amd/share/cuShareScalarL1.hpp"


#include "benchmarks/nvidia/amount/constantL1Amount.hpp"
#include "benchmarks/nvidia/amount/readOnlyAmount.hpp"
#include "benchmarks/nvidia/amount/textureAmount.hpp"

#include "benchmarks/nvidia/difference/differenceL1L2.hpp"

#include "benchmarks/nvidia/latency/constantL1Latency.hpp"
#include "benchmarks/nvidia/latency/constantL15Latency.hpp"
#include "benchmarks/nvidia/latency/readOnlyLatency.hpp"
#include "benchmarks/nvidia/latency/textureLatency.hpp"

#include "benchmarks/nvidia/share/constantL1SharedWithL1.hpp"
#include "benchmarks/nvidia/share/readOnlySharedWithL1.hpp"
#include "benchmarks/nvidia/share/textureSharedWithReadOnly.hpp"
#include "benchmarks/nvidia/share/textureSharedWithL1.hpp"

#include "benchmarks/nvidia/missPenalty/constantL1MissPenalty.hpp"
#include "benchmarks/nvidia/missPenalty/readOnlyMissPenalty.hpp"
#include "benchmarks/nvidia/missPenalty/textureMissPenalty.hpp"

#include "benchmarks/nvidia/size/cacheSize/constantL1CacheSize.hpp"
#include "benchmarks/nvidia/size/fetchGranularity/constantL1FetchGranularity.hpp"
#include "benchmarks/nvidia/size/lineSize/constantL1LineSize.hpp"

#include "benchmarks/nvidia/size/cacheSize/constantL15CacheSize.hpp"
#include "benchmarks/nvidia/size/fetchGranularity/constantL15FetchGranularity.hpp"
#include "benchmarks/nvidia/size/lineSize/constantL15LineSize.hpp"

#include "benchmarks/nvidia/size/cacheSize/readOnlyCacheSize.hpp"
#include "benchmarks/nvidia/size/fetchGranularity/readOnlyFetchGranularity.hpp"
#include "benchmarks/nvidia/size/lineSize/readOnlyLineSize.hpp"

#include "benchmarks/nvidia/size/cacheSize/textureCacheSize.hpp"
#include "benchmarks/nvidia/size/fetchGranularity/textureFetchGranularity.hpp"
#include "benchmarks/nvidia/size/lineSize/textureLineSize.hpp"