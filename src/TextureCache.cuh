//
// Created by nick- on 8/5/2022.
//

#ifndef CUDATEST_TEXTURECACHE_CUH
#define CUDATEST_TEXTURECACHE_CUH

# include "texture.cuh"
# include "texture_lat.cuh"

CacheResults executeTextureCacheChecks(){
    CacheResults result;
    printf("\n\nMeasure Texture Cache Size\n");
#ifdef IsDebug
    fprintf(out, "\n\nMeasure Texture Cache Size\n\n");
#endif //IsDebug
    CacheSizeResult TextureSizeInBytes = measure_texture();
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

#endif //CUDATEST_TEXTURECACHE_CUH
