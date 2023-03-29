
#ifndef CUDATEST_L2_SEGMENTSIZE
#define CUDATEST_L2_SEGMENTSIZE

#define BLOCK_SIZE 256
#define REPEATS 32
#define L2_START_SIZE 500000    //500kB
#define L2_MAX_SIZE 1000000000  //1GB
#define NUM_EXPERIMENTS_BEFORE_DOUBLING 100 //before the size doubles, divide into 100 experiments (smaller sizes -> more precise, larger sizes -> less precise)

__global__ void l2_segment_size (unsigned int * my_array, int array_length, unsigned long long* time, unsigned int * last);
int detect1DChangePoint(unsigned long long* arr, int num_elems, int look_around, double threshold);
int launchL2SegmentSizeBenchmark(int lowerBound, int upperBound, unsigned long long* time_array, int arrayIncrease_elements);

CacheSizeResult measure_L2_segment_size(unsigned int l1SizeBytes) {

    int absoluteLowerBoundary = l1SizeBytes * 2 / sizeof(unsigned int); // start at 2x L1 size (at least 1 MB)
    int absoluteUpperBoundary = L2_MAX_SIZE / sizeof(unsigned int);

    //for each doubling of size, there will be NUM_EXPERIMENTS_BEFORE_DOUBLING experiments
    int max_num_experiments = NUM_EXPERIMENTS_BEFORE_DOUBLING +1;
    int lower_bound = absoluteLowerBoundary;
    while (lower_bound <= absoluteUpperBoundary){
        lower_bound = lower_bound *2;
        max_num_experiments += NUM_EXPERIMENTS_BEFORE_DOUBLING;
    }
    unsigned long long* time_array = (unsigned long long*)malloc(sizeof(unsigned long long) * max_num_experiments);//array with results
    int cp;
    lower_bound = absoluteLowerBoundary;
    int num_experiments = 0;
    //run NUM_EXPERIMENTS_BEFORE_DOUBLING experiments, and check if there is a change point, otherwise increase maximum boundary
    do{
        int arrayIncrease_elements = lower_bound/NUM_EXPERIMENTS_BEFORE_DOUBLING;
        //printf("searching boundary in: %d -- %d, increasing by %d elems \n",lower_bound*sizeof(unsigned int), lower_bound*2*sizeof(unsigned int), arrayIncrease_elements);
        launchL2SegmentSizeBenchmark(lower_bound, lower_bound*2, &(time_array[num_experiments]), arrayIncrease_elements);

        num_experiments += NUM_EXPERIMENTS_BEFORE_DOUBLING;
        lower_bound = lower_bound << 1;
        cp = detect1DChangePoint(time_array, num_experiments, 5, 1.25);

    }
    while(cp == -1 && lower_bound >> 1 < absoluteUpperBoundary);

    int cache_size = absoluteLowerBoundary << (cp/NUM_EXPERIMENTS_BEFORE_DOUBLING); //double for each NUM_EXPERIMENTS_BEFORE_DOUBLING experiments before
    cache_size = cache_size + (cp% NUM_EXPERIMENTS_BEFORE_DOUBLING) * cache_size/NUM_EXPERIMENTS_BEFORE_DOUBLING;
    cache_size = cache_size * sizeof(unsigned int);
    printf("L2 segment: found boundary %dkB (index %d) \n", cache_size/1000, cp );

    CacheSizeResult result;
    result.CacheSize = cache_size;
    result.realCP = cp > 0;
    result.maxSizeBenchmarked = absoluteUpperBoundary*sizeof(unsigned int) ;
    return result;
}

//TODO this could be united with the changepoint detection for other benchmarks
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

    // printf("array_sz, time, time/B \n");

    int num_experiments = (upperBound - lowerBound ) / arrayIncrease_elements;

    // Allocate host memory
    h_a   = (unsigned int*)malloc(sizeof(unsigned int) * upperBound);

    // Initialize p-chase array
    for (int i = 0; i < upperBound; i++){
        h_a[i] = 1;
    }

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

#endif //CUDATEST_L2_SEGMENTSIZE
