//
// Created by nick- on 8/5/2022.
//

#ifndef CUDATEST_TEXTURECACHE_CPP
#define CUDATEST_TEXTURECACHE_CPP

# include "texture.h"
# include "texture_lat.h"
# include "test_functions.h"

/**
 * @brief Measures the Texture Cache latency
 * @return CacheResults
 */
CacheResults executeTextureCacheChecks(){
    CacheResults result;
    printf("\n\nMeasure Texture Cache Size\n");
#ifdef IsDebug
    fprintf(out, "\n\nMeasure Texture Cache Size\n\n");
#endif //IsDebug
    CacheSizeResult TextureSizeInBytes = measureSmallCache(3);
    printf("\n\nMeasure Texture Cache Cache Line Size\n");
#ifdef IsDebug
    fprintf(out, "\n\nMeasure Texture Cache Cache Line Size\n\n");
#endif //IsDebug
    unsigned int cacheLineSize = wrapperLineSize(TextureSizeInBytes.CacheSize, launchTextureBenchmark);
    printf("\n\nMeasure Texture Cache latencyCycles\n");
#ifdef IsDebug
    fprintf(out, "\n\nMeasure Texture Cache latencyCycles\n\n");
#endif //IsDebug
    LatencyTuple latency = measure_Texture_Lat();
    result.CacheSize = TextureSizeInBytes;
    result.cacheLineSize = cacheLineSize;
    result.latencyCycles = latency.latencyCycles;
    result.latencyNano = latency.latencyNano;
    result.benchmarked = true;
    return result;
}

#endif //CUDATEST_TEXTURECACHE_CPP
