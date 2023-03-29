//
// Created by nick- on 8/5/2022.
//

#ifndef CUDATEST_L2DATACACHE_CUH
#define CUDATEST_L2DATACACHE_CUH

# include "l2_lat.cuh"
# include "l2LatTest.cuh"
# include "LineSize/l2_linesize.cuh"
# include "l2_segment_size.cuh"


CacheResults executeL2DataCacheChecks(unsigned int l2SizeBytes, unsigned int l1SizeBytes = L2_START_SIZE ){
    printf("EXECUTE L2 DATACACHE CHECK\n");
#ifdef IsDebug
    fprintf(out, "EXECUTE L2 DATACACHE CHECK\n");
#endif //IsDebug
    printf("\n\nMeasure L2 DataCache Segment Size\n");
#ifdef IsDebug
    fprintf(out, "\n\nMeasure L2 DataCache Segment Size\n\n");
#endif //IsDebug
    CacheSizeResult L2SegmentSizeInBytes = measure_L2_segment_size((l1SizeBytes>L2_START_SIZE) ? l1SizeBytes : L2_START_SIZE);
    printf("\n\nMeasure L2 DataCache Cache Line Size\n");
#ifdef IsDebug
    fprintf(out, "\n\nMeasure L2 DataCache  Cache Line Size\n\n");
#endif //IsDebug
    unsigned int cacheLineSize = measure_L2_LineSize_Alt(l2SizeBytes);
    printf("\n\nMeasure L2 DataCache latencyCycles\n");
#ifdef IsDebug
    fprintf(out, "\n\nMeasure L2 DataCache latencyCycles\n\n");
#endif //IsDebug
    measure_L2LatTest();
    LatencyTuple latency = measure_L2_Lat();
    CacheResults result;
    result.CacheSize = L2SegmentSizeInBytes;
    result.cacheLineSize = cacheLineSize;
    result.latencyCycles = latency.latencyCycles;
    result.latencyNano = latency.latencyNano;
    result.benchmarked = true;
    return result;
}

#endif //CUDATEST_L2DATACACHE_CUH
