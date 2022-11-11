//
// Created by nick- on 8/31/2022.
//

#ifndef CUDATEST_CHKALLCORE_CUH
#define CUDATEST_CHKALLCORE_CUH

# include "chkAllTexture.cuh"
# include "chkAllRO.cuh"
# include "chkAllL1.cuh"
# include "chkAllConst.cuh"

uIntTriple checkNumberOfCachesPerSM(CacheResults textureResults, CacheResults ReadOnlyResults, CacheResults L1_results, CudaDeviceInfo cudaInfo,
                                    bool ROShareTexture, bool ROShareL1Data, bool L1ShareTexture);

unsigned int measure_TwoCoreCache(unsigned int measuredSizeCache, unsigned int sub, unsigned int numberCores, char* cacheType,
                                  bool (*reference)(int, int, double*, unsigned int*, unsigned int**, int*),
                                  bool (*benchmark)(unsigned int, double*, double*, unsigned int*, unsigned int*, unsigned int **, unsigned int**, int*, unsigned int, unsigned int, unsigned int));

unsigned int corePts(unsigned int CacheSizeInInt, double avgFlowRef, unsigned int numTestedCores, char* cacheType, int* error,
                    bool (*benchmark)(unsigned int, double*, double*, unsigned int*, unsigned int*, unsigned int **, unsigned int**, int*, unsigned int, unsigned int, unsigned int));

double wrapperLaunchTwoCore(unsigned int CacheSizeInInt, unsigned int numberCores, unsigned int testCore, double avgRef, char* cacheType, int* error,
                            bool (*benchmark)(unsigned int, double*, double*, unsigned int*, unsigned int*, unsigned int **, unsigned int**, int*, unsigned int, unsigned int, unsigned int));

uIntTriple checkNumberOfCachesPerSM(CacheResults textureResults, CacheResults ReadOnlyResults, CacheResults L1_results, CudaDeviceInfo cudaInfo,
                                    bool ROShareTexture, bool ROShareL1Data, bool L1ShareTexture) {
    uIntTriple result = {0, 0, 0};

    if (textureResults.benchmarked) {
        printf("\nCheck how many Texture Caches exist Per SM\n\n");
        char type[] = "Texture";
        unsigned int cpts = measure_TwoCoreCache(textureResults.CacheSize.CacheSize, 200, cudaInfo.maxThreadsPerBlock,
                                                 type, launchTextureBenchmarkReferenceValue, launchBenchmarkTwoCoreTexture);
        printf(" %d Texture Cache(s) in 1 SM\n\n", cpts);
        result.first = cpts;
    }

    if (ReadOnlyResults.benchmarked && !ROShareTexture) {
        //two core check for RO
        printf("\nCheck how many Read-Only Caches exist Per SM\n\n");
        char type[] = "RO";
        unsigned int cpts = measure_TwoCoreCache(ReadOnlyResults.CacheSize.CacheSize, 200, cudaInfo.maxThreadsPerBlock,
                                                 type, launchROBenchmarkReferenceValue, launchBenchmarkTwoCoreRO);
        printf("%d Read-Only Cache(s) in 1 SM\n\n", cpts);
        result.second = cpts;
    } else if (ROShareTexture) {
        result.second = result.first;
    }

    if (L1_results.benchmarked && (!ROShareL1Data || !L1ShareTexture)) {
        //two core check for L1
        printf("\nCheck how many L1 Data Caches exist Per SM\n\n");
        char type[] = "L1";
        unsigned int cpts = measure_TwoCoreCache(L1_results.CacheSize.CacheSize, 200, cudaInfo.maxThreadsPerBlock, type,
                                                 launchL1DataBenchmarkReferenceValue, launchBenchmarkTwoCoreL1);
        printf("%d L1 Data Cache(s) in 1 SM\n\n", cpts);
        result.third = cpts;
    } else if (L1ShareTexture) {
        result.third = result.first;
    } else if (ROShareL1Data) {
        result.third = result.second;
    }

    return result;
}

#define FreeMeasureTwoCoreCache() \
free(timeRef);                    \
free(potMissesFlowRef);           \
free(avgFlowRef);                 \




unsigned int measure_TwoCoreCache(unsigned int measuredSizeCache, unsigned int sub, unsigned int numberCores, char* cacheType,
                                  bool (*reference)(int, int, double*, unsigned int*, unsigned int**, int*),
                                  bool (*benchmark)(unsigned int, double*, double*, unsigned int*, unsigned int*, unsigned int **, unsigned int**, int*, unsigned int, unsigned int, unsigned int)) {

    unsigned int CacheSizeInInt = (measuredSizeCache - sub) / 4;

    double *avgFlowRef = (double *) malloc(sizeof(double));
    unsigned int *potMissesFlowRef = (unsigned int *) malloc(sizeof(unsigned int));
    unsigned int **timeRef = (unsigned int **) malloc(sizeof(unsigned int *));
    if (avgFlowRef == nullptr || potMissesFlowRef == nullptr || timeRef == nullptr) {
        FreeMeasureTwoCoreCache()
        printErrorCodeInformation(1);
        exit(1);
    }
    timeRef[0] = nullptr;

    bool dist = true;
    int n = 5;
    while (dist && n > 0) {
        int error = 0;
        dist = reference((int) CacheSizeInInt, 1, avgFlowRef, potMissesFlowRef, timeRef, &error);
        if (error != 0) {
            if (timeRef[0] != nullptr) {
                free(timeRef[0]);
            }
            FreeMeasureTwoCoreCache()
            printErrorCodeInformation(error);
            exit(error);
        }
        --n;
    }

    std::vector<unsigned int> pts;
    int error = 0;

    unsigned int CoresSameCache = corePts(CacheSizeInInt, avgFlowRef[0], numberCores, cacheType, &error, benchmark);
    if (error != 0) {
        if (timeRef[0] != nullptr) {
            free(timeRef[0]);
        }
        FreeMeasureTwoCoreCache()
        printErrorCodeInformation(error);
        exit(error);
    }

    unsigned int totalThreads = numberCores;
    while(CoresSameCache == 0) {
        printf("Number of Executed Threads too high for check - dividing by 2\n");
        numberCores = numberCores >> 1; // / 2;
        CoresSameCache = corePts(CacheSizeInInt, avgFlowRef[0], numberCores, cacheType, &error, benchmark);
        if (error != 0) {
            if (timeRef[0] != nullptr) {
                free(timeRef[0]);
            }
            FreeMeasureTwoCoreCache()
            printErrorCodeInformation(error);
            exit(error);
        }
    }

    unsigned int tmpNumberCores = numberCores;
    while (numberCores % CoresSameCache != 0 && numberCores < tmpNumberCores * 2) {
        printf("Resulting number of cores with same cache not dividable by numberOfCores/threads -> not all cores are equally used - Increasing numberOfThreads by 1: %d", ++numberCores);
        if (numberCores > totalThreads) {
            break;
        }
        CoresSameCache = corePts(CacheSizeInInt, avgFlowRef[0], numberCores, cacheType, &error, benchmark);
        if (error != 0) {
            if (timeRef[0] != nullptr) {
                free(timeRef[0]);
            }
            FreeMeasureTwoCoreCache()
            printErrorCodeInformation(error);
            exit(error);
        }
    }

    unsigned int nCaches = numberCores / CoresSameCache;

    if (timeRef[0] != nullptr) {
        free(timeRef[0]);
    }
    FreeMeasureTwoCoreCache()

    return nCaches;
}

unsigned int corePts(unsigned int CacheSizeInInt, double avgFlowRef, unsigned int numTestedCores, char* cacheType, int* error,
                    bool (*benchmark)(unsigned int, double*, double*, unsigned int*, unsigned int*, unsigned int **, unsigned int**, int*, unsigned int, unsigned int, unsigned int)) {
    printf("corePts: numTestedCores/-Threads = %d\n", numTestedCores);
    double* distances = (double*) malloc(sizeof(double) * numTestedCores-1);

    distances[0] = wrapperLaunchTwoCore(CacheSizeInInt, numTestedCores, 1, avgFlowRef, cacheType, error, benchmark);
    if (*error != 0) {
        free(distances);
        return 0;
    }
    distances[1] = wrapperLaunchTwoCore(CacheSizeInInt, numTestedCores, 2, avgFlowRef, cacheType, error, benchmark);
    if (*error != 0) {
        free(distances);
        return 0;
    }
    distances[2] = wrapperLaunchTwoCore(CacheSizeInInt, numTestedCores, 3, avgFlowRef, cacheType, error, benchmark);
    if (*error != 0) {
        free(distances);
        return 0;
    }
    distances[3] = wrapperLaunchTwoCore(CacheSizeInInt, numTestedCores, 4, avgFlowRef, cacheType, error, benchmark);
    if (*error != 0) {
        free(distances);
        return 0;
    }
    if (distances[0] < 4.0 || distances[1] < 4.0 || distances[2] < 4.0 || distances[3] < 4.0) {
        return 0;
    }
    for (int i = 5; i < numTestedCores; i++) {
        distances[i-1] = wrapperLaunchTwoCore(CacheSizeInInt, numTestedCores, i, avgFlowRef, cacheType, error, benchmark);
        if (*error != 0) {
            free(distances);
            return 0;
        }
    }

#ifdef IsDebug
    char fileName[64];
    snprintf(fileName, 64, "coreFlow_%s_%d.log", cacheType, numTestedCores);
    FILE * allCoreFile = fopen(fileName, "w");
    if (allCoreFile == nullptr) {
        free(distances);
        return 0;
    }
#endif //IsDebug

    double ref = distances[0]; //this one will definitely disturb the base core
    double tol = ref / 5.0;
    unsigned int countSameCache = 1;
    for (int i = 0; i < numTestedCores-1; ++i) {
#ifdef IsDebug
        fprintf(allCoreFile, "%d:%f\n", i + 1, distances[i]);
#endif //IsDebug
        if (distances[i] >= ref - tol) {
            ++countSameCache;
        }
    }
#ifdef IsDebug
    fclose(allCoreFile);
#endif //IsDebug

    free(distances);
    return countSameCache;
}

#define FreeWrapperLaunchTwoCore() \
if (time[0] != nullptr) {      \
    free(time[0]);             \
}                                  \
if (time[1] != nullptr) {           \
    free(time[1]);                 \
}                               \
free(time);                    \
free(potMissesFlow);           \
free(avgFlow);                 \

double wrapperLaunchTwoCore(unsigned int CacheSizeInInt, unsigned int numberCores, unsigned int testCore, double avgRef, char* cacheType, int* error,
                            bool (*benchmark)(unsigned int, double*, double*, unsigned int*, unsigned int*, unsigned int **, unsigned int**, int*, unsigned int, unsigned int, unsigned int)) {
    unsigned int baseCore = 0;
    double *avgFlow = (double *) malloc(sizeof(double) * 2);
    unsigned int *potMissesFlow = (unsigned int *) malloc(sizeof(unsigned int) * 2);
    unsigned int **time = (unsigned int **) malloc(sizeof(unsigned int *) * 2);

    dTriple result;

    bool dist = true;
    int n = 5;
    while (dist && n > 0) {
        dist = benchmark(CacheSizeInInt, &avgFlow[0], &avgFlow[1], &potMissesFlow[0], &potMissesFlow[1], &time[0], &time[1], error, numberCores, baseCore, testCore);
        if (*error != 0) {
            FreeWrapperLaunchTwoCore()
            return 0.;
        }
        --n;
    }

    result.first = avgRef;
    result.second = avgFlow[0];
    result.third = avgFlow[1];

#ifdef IsDebug
    fprintf(out, "Measured %s Avg in clean execution: %f\n", cacheType, avgRef);

    fprintf(out, "Measured %s_1 Avg While Shared With %s_2:  %f\n", cacheType, cacheType, avgFlow[0]);
    fprintf(out, "Measured %s_1 Pot Misses While Shared With %s_2:  %u\n", cacheType, cacheType, potMissesFlow[0]);

    fprintf(out, "Measured %s_2 Avg While Shared With %s_1:  %f\n", cacheType, cacheType, avgFlow[1]);
    fprintf(out, "Measured %s_2 Pot Misses While Shared With %s_1:  %u\n", cacheType, cacheType, potMissesFlow[1]);
#endif //IsDebug

    FreeWrapperLaunchTwoCore()

    return std::max(std::abs(result.second - result.first), std::abs(result.third - result.first));
}

#endif //CUDATEST_CHKALLCORE_CUH
