
#ifndef CUDATEST_L2_SEGMENTSIZE
#define CUDATEST_L2_SEGMENTSIZE

#define BLOCK_SIZE 256
#define REPEATS 32
#define L2_START_SIZE 500000

__global__ void l2_segment_size (unsigned int * my_array, int array_length, unsigned long long* time, unsigned int * last);
int detect1DChangePoint(unsigned long long* arr, int num_elems, int look_around, double threshold);
int launchL2SegmentSizeBenchmark(int lowerBound, int upperBound, unsigned long long* time_array, int arrayIncrease_elements);



CacheSizeResult measure_L2_segment_size(unsigned int l1SizeBytes) {

    int absoluteLowerBoundary = l1SizeBytes * 2 / sizeof(unsigned int); // start at 2x L1 size
    int absoluteUpperBoundary = 1024 * 1024 * 1024 / sizeof(unsigned int); // 1GB
    int num_experiments_before_doubling = 100;// (smaller sizes -> more precise, larger sizes -> less precise)

    //for each doubling of size, there will be num_experiments_before_doubling experiments
    int max_num_experiments = num_experiments_before_doubling +1;
    int lower_bound = absoluteLowerBoundary;
    while (lower_bound <= absoluteUpperBoundary){
        lower_bound = lower_bound *2;
        max_num_experiments += num_experiments_before_doubling;
    }
    unsigned long long* time_array = (unsigned long long*)malloc(sizeof(unsigned long long) * max_num_experiments);//array with results
    int cp;
    lower_bound = absoluteLowerBoundary;
    int num_experiments = 0;
    //run num_experiments_before_doubling experiments, and check if there is a change point, otherwise increase maximum boundary
    do{
        int arrayIncrease_elements = lower_bound/num_experiments_before_doubling;
        //printf("searching boundary in: %d -- %d, increasing by %d elems \n",lower_bound*sizeof(unsigned int), lower_bound*2*sizeof(unsigned int), arrayIncrease_elements);
        launchL2SegmentSizeBenchmark(lower_bound, lower_bound*2, &(time_array[num_experiments]), arrayIncrease_elements);

        num_experiments += num_experiments_before_doubling;
        lower_bound = lower_bound << 1;
        cp = detect1DChangePoint(time_array, num_experiments, 5, 1.25);

    }
    while(cp == -1 && lower_bound >> 1 < absoluteUpperBoundary);

    int cache_size = absoluteLowerBoundary << (cp/num_experiments_before_doubling); //double for each num_experiments_before_doubling experiments before
    cache_size = cache_size + (cp% num_experiments_before_doubling) * cache_size/num_experiments_before_doubling;
    cache_size = cache_size * sizeof(unsigned int);
    //printf("found boundary: %dkB (index %d) \n", cache_size/1000, cp );

    CacheSizeResult result;
    result.CacheSize = cache_size;
    result.realCP = cp > 0;
    result.maxSizeBenchmarked = absoluteUpperBoundary*sizeof(unsigned int) ;
    return result;
}

//TODO this could be united with the changepoint detection for
int detect1DChangePoint(unsigned long long* arr, int num_elems, int look_around = 5, double threshold = 1.25)
{
    if(look_around < 1) //must not happen
        return -1;
    double largest_spike = 0;
    int largest_spike_index = -1;
    for(int i = look_around; i< num_elems - look_around; i++)
    {
        double lower_avg = 0;
        double upper_avg = 0;
        for(int j = 1; j<=look_around; j++)
        {
            lower_avg += arr[i-j];
            upper_avg += arr[i+j];
        }
        lower_avg = lower_avg / look_around;
        upper_avg = upper_avg / look_around;
        if(lower_avg * threshold < upper_avg) 
        {
            //printf("-found changepoint on index %d (%f, %f) -- spike %f \n", i, lower_avg, upper_avg, upper_avg / lower_avg );
            if(upper_avg / lower_avg > largest_spike)
            {
                largest_spike = upper_avg / lower_avg;
                largest_spike_index = i;
            }
        }
    }
    //if(largest_spike_index != -1)
    //    printf("-largest spike on index %d \n", largest_spike_index);
    return largest_spike_index;
}

__global__ void l2_segment_size (unsigned int * my_array, int array_length, unsigned long long* time, unsigned int * last)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long start_time, end_time;

    // First round
    unsigned int* ptr;
    unsigned int j = 0;
    unsigned int local_sum = 0;
    for(int k = tid; k< array_length; k+=BLOCK_SIZE){
        ptr = my_array + j;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(j) : "l"(ptr) : "memory");
	    //j = my_array[j];
	}

    // Second round
    asm volatile(" .reg .u64 mem_ptr64;\n\t"
                " cvta.to.global.u64 mem_ptr64, %0;\n\t" :: "l"(last));

    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start_time));
    for(int r = 0; r<REPEATS; r++) {
        for(int k = tid; k<array_length; k+=BLOCK_SIZE){
            local_sum += my_array[k];
        }
    }
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(end_time));

    if(tid == 0)
    {
        *last = local_sum;
        *time = ((end_time - start_time)*BLOCK_SIZE*1000)/(REPEATS * array_length * sizeof(unsigned int));

        // printf("---KERNEL: size %d kB, ", (int)array_length*4/1000);
        // printf("time %llu, "  , (end_time - start_time));
        // printf(" avg %llu,  ", *time);
        // printf(" last %u \n", *last);
    }
}

int launchL2SegmentSizeBenchmark(int lowerBound, int upperBound, unsigned long long* time, int arrayIncrease_elements = 100000)
{
    unsigned int *h_a, *d_a, *d_last;
    unsigned long long *d_time;

    int stride = BLOCK_SIZE; // => each block should access its own elements

    // const char* env_p = std::getenv("CUDA_VISIBLE_DEVICES");
    // printf("CUDA_VISIBLE_DEVICES, %s \n", env_p);
    // printf("array_sz, time, time/B \n");

    int num_experiments = (upperBound - lowerBound ) / arrayIncrease_elements;

    // Allocate host memory
    h_a   = (unsigned int*)malloc(sizeof(unsigned int) * upperBound);

    // Initialize p-chase array
    for (int i = 0; i < upperBound; i++){
        h_a[i] = 1;
        //h_a[i] = (i + stride) % absoluteUpperBoundary;
    }
    // for (int i = 0; i < num_experiments; i++){
    //     time[i] = 0;
    // }

    // Allocate device memory
    cudaMalloc((void**)&d_a, sizeof(unsigned int) * upperBound);
    cudaMalloc((void**)&d_last, sizeof(unsigned int));
    cudaMalloc((void**)&d_time, sizeof(unsigned long long) * num_experiments);
    // Transfer data from host to device memory
    cudaMemcpy(d_a, h_a, sizeof(unsigned int) * upperBound, cudaMemcpyHostToDevice);
    cudaMemcpy(d_time, time, sizeof(unsigned long long) * num_experiments, cudaMemcpyHostToDevice);

    for(int i = 0; i < num_experiments; i++)
    {
        int num_elems = lowerBound + i*arrayIncrease_elements;
        l2_segment_size<<<1,stride>>>(d_a, num_elems, &(d_time[i]), d_last);
    }
    // Transfer data back to host memory
    cudaMemcpy(time, d_time, sizeof(unsigned long long) * num_experiments, cudaMemcpyDeviceToHost);

    // unsigned int** time_2d = (unsigned int**)malloc(sizeof(unsigned int*) * num_experiments);
    // for (int i = 0; i < num_experiments; i++){
    //     time_2d[i] = (unsigned int*)malloc(sizeof(unsigned int));//1 element
    //     time_2d[i][0] = (unsigned int)time[i];
    // }
    //int result = detectChangePoint(time_2d, num_experiments, 1);
    
    // for(int i = 0; i< num_experiments; i++)
    // {
    //     printf("%u,     %u\n", (lowerBound+i*arrayIncrease_elements)*sizeof(int), time[i]);
    // }

    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_last);
    cudaFree(d_time);

    // Deallocate host memory
    free(h_a);
    return 1;
}

// __global__ void l2_segment_size (unsigned int * my_array, int array_length, unsigned int * duration, unsigned int *index, bool* isDisturbed);

// bool launchL2SegmentSizeBenchmark(int N, int stride, double *avgOut, unsigned int* potMissesOut, unsigned int** time, int* error);

// CacheSizeResult measure_L2_segment_size(unsigned int l1SizeBytes) {
// // 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
//     int absoluteLowerBoundary = l1SizeBytes * 2; // start at 2x L1 size
//     int absoluteUpperBoundary = 1024 * 1024 * 1024; // 1GB
//     int widenBounds = 0;

//     //Start with 1K integers until 1000K integers
//     int bounds[2] = {absoluteLowerBoundary, absoluteUpperBoundary};
//     getBoundaries(launchL2SegmentSizeBenchmark, bounds, 5);
// #ifdef IsDebug
//     fprintf(out, "Got Boundaries: %d...%d\n", bounds[0], bounds[1]);
// #endif //IsDebug
//     printf("Got Boundaries: %d...%d\n", bounds[0], bounds[1]);

//     int cp = -1;
//     int begin = bounds[0] - widenBounds;
//     int end = bounds[1] + widenBounds;
//     int stride = 1;
//     int arrayIncrease = 1; //TODO: no need for precision here -- jump by 1000 elems (4kB), however there is an error to fix

//     while (cp == -1 && begin >= absoluteLowerBoundary / sizeof(int) - widenBounds && end <= absoluteUpperBoundary / sizeof(int) + widenBounds) {
//         cp = wrapBenchmarkLaunch(launchL2SegmentSizeBenchmark, begin, end, stride, arrayIncrease, "L2");

//         if (cp == -1) {
//             begin = begin - (end - begin);
//             end = end + (end - begin);
// #ifdef IsDebug
//             fprintf(out, "\nGot Boundaries: %d...%d\n", begin, end);
// #endif //IsDebug
//             printf("\nGot Boundaries: %d...%d\n", begin, end);
//         }
//     }

//     CacheSizeResult result;
//     int cacheSizeInInt = (begin + cp * arrayIncrease);
//     result.CacheSize = (cacheSizeInInt << 2); // * 4);
//     result.realCP = cp > 0;
//     result.maxSizeBenchmarked = end << 2; // * 4;
//     return result;
// }


// bool launchL2SegmentSizeBenchmark(int N, int stride, double *avgOut, unsigned int* potMissesOut, unsigned int** time, int* error) {
//     //cudaDeviceReset();
//     cudaError_t error_id;

//     unsigned int *h_a = nullptr, *h_index = nullptr, *h_timeinfo = nullptr,
//     *d_a = nullptr, *duration = nullptr, *d_index = nullptr;
//     bool *disturb = nullptr, *d_disturb = nullptr;

//     do {
//         // Allocate Memory on Host
//         h_a = (unsigned int *) malloc(sizeof(unsigned int) * (N));
//         if (h_a == nullptr) {
//             printf("[L2_SEGMENT_SIZE.CUH]: malloc h_a Error\n");
//             *error = 1;
//             break;
//         }

//         h_index = (unsigned int *) malloc(sizeof(unsigned int) * MEASURE_SIZE);
//         if (h_index == nullptr) {
//             printf("[L2_SEGMENT_SIZE.CUH]: malloc h_index Error\n");
//             *error = 1;
//             break;
//         }

//         h_timeinfo = (unsigned int *) malloc(sizeof(unsigned int) * MEASURE_SIZE);
//         if (h_timeinfo == nullptr) {
//             printf("[L2_SEGMENT_SIZE.CUH]: malloc h_timeinfo Error\n");
//             *error = 1;
//             break;
//         }

//         disturb = (bool *) malloc(sizeof(bool));
//         if (disturb == nullptr) {
//             printf("[L2_SEGMENT_SIZE.CUH]: malloc disturb Error\n");
//             *error = 1;
//             break;
//         }

//         // Allocate Memory on GPU
//         error_id = cudaMalloc((void **) &d_a, sizeof(unsigned int) * (N));
//         if (error_id != cudaSuccess) {
//             printf("[L2_SEGMENT_SIZE.CUH]: cudaMalloc d_a Error: %s\n", cudaGetErrorString(error_id));
//             *error = 2;
//             break;
//         }

//         error_id = cudaMalloc((void **) &duration, sizeof(unsigned int) * MEASURE_SIZE);
//         if (error_id != cudaSuccess) {
//             printf("[L2_SEGMENT_SIZE.CUH]: cudaMalloc duration Error: %s\n", cudaGetErrorString(error_id));
//             *error = 2;
//             break;
//         }

//         error_id = cudaMalloc((void **) &d_index, sizeof(unsigned int) * MEASURE_SIZE);
//         if (error_id != cudaSuccess) {
//             printf("[L2_SEGMENT_SIZE.CUH]: cudaMalloc d_index Error: %s\n", cudaGetErrorString(error_id));
//             *error = 2;
//             break;
//         }

//         error_id = cudaMalloc((void **) &d_disturb, sizeof(bool));
//         if (error_id != cudaSuccess) {
//             printf("[L2_SEGMENT_SIZE.CUH]: cudaMalloc disturb Error: %s\n", cudaGetErrorString(error_id));
//             *error = 2;
//             break;
//         }

//         // Initialize p-chase array
//         for (int i = 0; i < N; i++) {
//             h_a[i] = (i + stride) % N;
//         }

//         // Copy array from Host to GPU
//         error_id = cudaMemcpy(d_a, h_a, sizeof(unsigned int) * N, cudaMemcpyHostToDevice);
//         if (error_id != cudaSuccess) {
//             printf("[L2_SEGMENT_SIZE.CUH]: cudaMemcpy d_a Error: %s\n", cudaGetErrorString(error_id));
//             *error = 3;
//             break;
//         }
//         cudaDeviceSynchronize();

//         // Launch Kernel function
//         dim3 Db = dim3(1);
//         dim3 Dg = dim3(1, 1, 1);
//         l2_segment_size <<<Dg, Db>>>(d_a, N, duration, d_index, d_disturb);

//         cudaDeviceSynchronize();

//         error_id = cudaGetLastError();
//         if (error_id != cudaSuccess) {
//             printf("[L2_SEGMENT_SIZE.CUH]: Kernel launch/execution Error: %s\n", cudaGetErrorString(error_id));
//             *error = 5;
//             break;
//         }
//         cudaDeviceSynchronize();

//         // Copy results from GPU to Host
//         error_id = cudaMemcpy((void *) h_timeinfo, (void *) duration, sizeof(unsigned int) * MEASURE_SIZE,cudaMemcpyDeviceToHost);
//         if (error_id != cudaSuccess) {
//             printf("[L2_SEGMENT_SIZE.CUH]: cudaMemcpy duration Error: %s\n", cudaGetErrorString(error_id));
//             *error = 6;
//             break;
//         }

//         error_id = cudaMemcpy((void *) h_index, (void *) d_index, sizeof(unsigned int) * MEASURE_SIZE,cudaMemcpyDeviceToHost);
//         if (error_id != cudaSuccess) {
//             printf("[L2_SEGMENT_SIZE.CUH]: cudaMemcpy d_index Error: %s\n", cudaGetErrorString(error_id));
//             *error = 6;
//             break;
//         }

//         error_id = cudaMemcpy((void *) disturb, (void *) d_disturb, sizeof(bool), cudaMemcpyDeviceToHost);
//         if (error_id != cudaSuccess) {
//             printf("[L2_SEGMENT_SIZE.CUH]: cudaMemcpy disturb Error: %s\n", cudaGetErrorString(error_id));
//             *error = 6;
//             break;
//         }

//         cudaDeviceSynchronize();

//         if (!*disturb)
//             createOutputFile(N, MEASURE_SIZE, h_index, h_timeinfo, avgOut, potMissesOut, "L1_");

//     } while(false);

//     // Free Memory on GPU
//     if (d_a != nullptr) {
//         cudaFree(d_a);
//     }

//     if (d_index != nullptr) {
//         cudaFree(d_index);
//     }

//     if (duration != nullptr) {
//         cudaFree(duration);
//     }

//     if (d_disturb != nullptr) {
//         cudaFree(d_disturb);
//     }

//     bool ret = false;
//     if (disturb != nullptr) {
//         ret = *disturb;
//         free(disturb);
//     }

//     // Free Memory on Host
//     if (h_a != nullptr) {
//         free(h_a);
//     }

//     if (h_index != nullptr) {
//         free(h_index);
//     }

//     if (h_timeinfo != nullptr) {
//         if (time != nullptr) {
//             time[0] = h_timeinfo;
//         } else {
//             free(h_timeinfo);
//         }
//     }

//     cudaDeviceReset();
//     return ret;
// }

// __global__ void l2_segment_size (unsigned int * my_array, int array_length, unsigned int * duration, unsigned int *index, bool* isDisturbed) {

//     unsigned int start_time, end_time;
//     bool dist = false;
//     unsigned int j = 0;

//     for(int k=0; k<MEASURE_SIZE; k++){
//         s_index[k] = 0;
//         s_tvalue[k] = 0;
//     }

//     // First round
//     unsigned int* ptr;
// 	for (int k = 0; k < array_length; k++) {
//         ptr = my_array + j;
//         asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(j) : "l"(ptr) : "memory");
// 	    //j = my_array[j];
// 	}

//     // Second round
//     asm volatile(" .reg .u64 smem_ptr64;\n\t"
//                  " cvta.to.shared.u64 smem_ptr64, %0;\n\t" :: "l"(s_index));
//     for (int k = 0; k < MEASURE_SIZE; k++) {
//         ptr = my_array + j;
//         //start_time = clock();
//         asm volatile ("mov.u32 %0, %%clock;\n\t"
//                       "ld.global.cg.u32 %1, [%3];\n\t"
//                       "st.shared.u32 [smem_ptr64], %1;"
//                       "mov.u32 %2, %%clock;\n\t"
//                       "add.u64 smem_ptr64, smem_ptr64, 4;" : "=r"(start_time), "=r"(j), "=r"(end_time) : "l"(ptr) : "memory");
//             //start_time = clock();
//             //j = my_array[j];
//             //s_index[k] = j;
//             //end_time = clock();
//             s_tvalue[k] = end_time-start_time;
//     }

//     for(int k=0; k<MEASURE_SIZE; k++){
//         if (s_tvalue[k] > 2000) {
//             dist = true;
//         }
//         index[k]= s_index[k];
//         duration[k] = s_tvalue[k];
//     }
//     *isDisturbed = dist;
// }

#endif //CUDATEST_L2_SEGMENTSIZE
