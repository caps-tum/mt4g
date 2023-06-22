//
//
// Created by dominik on 25.05.22.
//
#ifndef CUDATEST_EVAL_H1
#define CUDATEST_EVAL_H1

#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING 1;

# include "hip/hip_runtime.h"
#include <cstdio>
#include <cstdlib>

#include <cstring>

#include <sys/stat.h>
#include <algorithm>
#include <cerrno>
#include <cassert>

#ifdef _WIN32
#include <direct.h>
#endif

#include <experimental/filesystem>
#include <iostream>
#include "ErrorHandler.h"

#ifdef IsDebug
FILE *out;
#endif

char separator() {
#ifdef _WIN32
    return '\\';
#else
    return '/';
#endif
}

#include <stdarg.h>

#include <hip/hip_runtime.h>

/** Free memory for a list of pointers
 * @param ptr_list list of pointers to free
 * @param hip if true, use hipFree (GPU), else use free (CPU-side)
 */
void FreeTestMemory(std::initializer_list<void *> ptr_list, bool hip) {
    for (auto ptr: ptr_list) {
        if (ptr != nullptr) {
            if (hip)
                hipError_t error_id = hipFree(ptr);
            else
                free(ptr);
        }
    }
}

// Function to allocate memory on the host
int allocate_host_memory(unsigned int **chunk, int size,
                         char *where, char *what, int error_code) {
    *chunk = (unsigned int *) malloc(size);
    if (*chunk == nullptr) {
        printf("[%s]: malloc %s Error\n", where, what);
        return error_code;
    }
    return 0;
}

// Function to allocate memory on the GPU
int allocate_device_memory(unsigned int **chunk, int size,
                           char *where, char *what, int error_code) {
    hipError_t error_id = hipMalloc((void **) chunk, size);
    if (error_id != hipSuccess) {
        printf("[%s]: hipMalloc %s Error: %s\n", where, what, hipGetErrorString(error_id));
        return error_code;
    }
    return 0;
}

// template for ints/doubles to calculate averages
template<typename T>
double computeAvg(const T *data, int size) {
    double avg = 0.;
    for (int i = 0; i < size; i++) {
        avg += data[i];
    }
    return (avg / (double) size);
}

/**
 * Returns the number of detected cache misses in a given time series
 * @param time
 * @param size
 * @param tolerance
 * @return
 */
unsigned int potCacheMisses(unsigned int *time, int size, double tolerance = 15) {
    int ret = 0;
    double avg = computeAvg(time, size);
    for (int i = 0; i < size; i++) {
        if (time[i] > avg + tolerance) {
            ret++;
        }
    }
    return ret;
}

/**
 * Returns the cache miss loading time values
 * @param time
 * @param size
 * @param missesOut
 * @param missesIndicesOut
 * @param avg
 * @param tolerance
 * @return number of misses
 */
unsigned int
potCacheMissesInfo(unsigned int *time, int size, unsigned int *missesOut, int *missesIndicesOut, double avg,
                   double tolerance = 15) {
    int ret = 0;
    for (int i = 0; i < size; i++) {
        if (time[i] > avg + tolerance) {
            missesOut[ret] = time[i];
            missesIndicesOut[ret] = i;
            ret++;
        }
    }
    return ret;
}

/**
 * Uses sliding window technique to use for each index the minimum of the window, in order to reduce random outliers
 * @param flow
 * @param size
 * @param slideWindow
 */
void postProcessing(double *flow, int size, int slideWindow) {
#ifdef IsDebug
    fprintf(out, "Post Processing / Filtering\n");
#endif //IsDebug
    double *outFlow = (double *) malloc(sizeof(double) * size);
    if (outFlow == nullptr) {
        printErrorCodeInformation(1);
        exit(1);
    }

    for (int i = 0; i < slideWindow / 2; i++) {
        int desiredLeftIndex = i - slideWindow / 2;
        int desiredRightIndex = i + slideWindow / 2;
        int actualLeftIndex = 0;
        desiredRightIndex = desiredRightIndex + (actualLeftIndex - desiredLeftIndex);
        double minValue = flow[actualLeftIndex];
        for (int k = actualLeftIndex + 1; k <= desiredRightIndex; ++k) {
            if (flow[k] < minValue) {
                minValue = flow[k];
            }
        }
        outFlow[i] = minValue;
    }

    for (int i = slideWindow / 2; i < size - slideWindow / 2; i++) {
        int desiredLeftIndex = i - slideWindow / 2;
        int desiredRightIndex = i + slideWindow / 2;
        double minValue = flow[desiredLeftIndex];
        for (int k = desiredLeftIndex + 1; k <= desiredRightIndex; k++) {
            if (flow[k] < minValue) {
                minValue = flow[k];
            }
        }
        outFlow[i] = minValue;
    }


    for (int i = size - slideWindow / 2; i < size; i++) {
        int desiredLeftIndex = i - slideWindow / 2;
        int desiredRightIndex = i + slideWindow / 2;
        int actualRightIndex = size - 1;
        desiredLeftIndex = desiredLeftIndex - (desiredRightIndex - actualRightIndex);
        double minValue = flow[desiredLeftIndex];
        for (int k = desiredLeftIndex + 1; k <= actualRightIndex; k++) {
            if (flow[k] < minValue) {
                minValue = flow[k];
            }
        }
        outFlow[i] = minValue;
    }
    memcpy(flow, outFlow, size * sizeof(double));
    free(outFlow);
#ifdef IsDebug
    fprintf(out, "Post Processing Done\n");
#endif //IsDebug
}

void printAvgFlow(double *flow, int size, int begin, int stride, const char *type) {
#ifdef IsDebug
    char outputFile[64];
    snprintf(outputFile, 64, "%s_avgFlow.txt", type);
    FILE *fp = fopen(outputFile, "w");
    if (fp == nullptr) {
        printErrorCodeInformation(1);
        exit(1);
    }
    fprintf(out, "******************\n");
    fprintf(out, "Flow of the average\n");
    fprintf(out, "******************\n");
    for (int i = 0; i < size; i++) {
        double sizeInKiB = ((double)((i * stride) + begin)) * (double)(sizeof(int)) / 1024.;
        fprintf(fp, "%f:%f\n", sizeInKiB, flow[i]);
        fprintf(out, "Array size: %f (int) - iteration: %f\n", sizeInKiB, flow[i]);
    }
    fclose(fp);
#endif //IsDebug
}


void printMissesFlow(unsigned int *flow, int size, int begin, int stride) {
#ifdef IsDebug
    fprintf(out, "******************\n");
    fprintf(out, "Flow of the potential misses\n");
    fprintf(out, "******************\n");
    for (int i = 0; i < size; i++) {
        fprintf(out, "Array size: %d (int) - iteration: %u\n", (i*stride)+begin, flow[i]);
    }
#endif //IsDebug
}

void cleanupOutput() {
    std::error_code errorCode;
    if (!std::experimental::filesystem::remove_all("output", errorCode)) {
        std::cout << errorCode.message() << std::endl;
    }
}

/**
 * Creates the output file BUT also assigns the average and potMisses values to the pointers!
 * @param N the size of the array
 * @param it the number of iterations
 * @param h_index the array of the indices
 * @param h_duration the array of the durations
 * @param avgOut the pointer to the average
 * @param potMissesOut the pointer to the potential misses
 * @param prefix the prefix of the file
 */
void
createOutputFile(int N, int it, unsigned int *h_index, unsigned *h_duration, double *avgOut, unsigned *potMissesOut,
                 const char *prefix) {
    const char *dirName = "output";
    char fileName[64];
    snprintf(fileName, 64, "output%c%soutput_%d.log", separator(), prefix, N);
    errno = 0;
    int ret;
#ifdef __linux__
    ret = mkdir(dirName, 0777);
#else
    ret = _mkdir(dirName);
#endif
    if (ret == -1) {
        switch (errno) {
            case EACCES :
                printf("the parent directory does not allow write");
                exit(EXIT_FAILURE);
            case EEXIST:
                //printf("pathname already exists - creation not required");
                break;
            case ENAMETOOLONG:
                printf("pathname is too long");
                exit(EXIT_FAILURE);
            default:
                perror("mkdir");
                exit(EXIT_FAILURE);
        }
    }

    double avg = computeAvg(h_duration, it);

    if (avgOut != nullptr) {
        avgOut[0] = avg;
    }
    unsigned int potMisses = potCacheMisses(h_duration, it, 30);

    unsigned int *missesValues = (unsigned int *) malloc(sizeof(unsigned) * potMisses);
    int *missesIndicesValues = (int *) malloc(sizeof(int) * potMisses);
    if (missesValues == nullptr || missesIndicesValues == nullptr) {
        free(missesValues);
        free(missesIndicesValues);
        printErrorCodeInformation(1);
        exit(1);
    }
    potCacheMissesInfo(h_duration, it, missesValues, missesIndicesValues, avg, 30);

#ifdef IsDebug
    FILE *fp;
    fp = fopen(fileName, "w");
    if (fp == nullptr) {
        free(missesValues);
        free(missesIndicesValues);
        printErrorCodeInformation(1);
        exit(1);
    }

    fprintf(fp, "=====Visiting the %f KiB array====\n", (float)(N)*sizeof(int)/1024.);
    for (int i = 0; i < it; i++){
        fprintf(fp, "%10d\t %10f\n", h_index[i], (float)h_duration[i]);
    }
    fprintf(fp, "Average load value: %f\n", avg);

    std::vector<unsigned int> missesValuesSet;
    for (int i = 0; i < potMisses; i++) {
        if (std::find(missesValuesSet.begin(), missesValuesSet.end(), missesValues[i]) == missesValuesSet.end()) {
            missesValuesSet.push_back(missesValues[i]);
        }
    }
    fprintf(fp, "Potential Cache Misses: %u\n", potMisses);
    fprintf(fp, "Detected Cache Miss Values: ");
    for (unsigned int value : missesValuesSet) {
        fprintf(fp, "%u\t", value);
    }
    fprintf(fp, "\n");
    fclose(fp);
#endif //IsDebug

    if (potMissesOut != nullptr) {
        potMissesOut[0] = potMisses;
    }

    if (potMisses != 0) {
        free(missesValues);
        free(missesIndicesValues);
    }

}

void print1DArr(FILE *f, double *arr, int size) {
    fprintf(f, "[");
    for (int i = 0; i < size - 1; i++) {
        fprintf(f, "%.3f,", arr[i]);
    }
    fprintf(f, "%.3f]\n", arr[size - 1]);
}

/**
 * Cost function for CPD
 * @param a
 * @param b
 * @param y
 *
 * @return
 */
double cost(unsigned int a, unsigned int b, double *y) {
#ifdef IsDebug
    assert(a < b);
#endif //IsDebug
    double totalSum = 0.;
    double empiricalMean = 0.;

#pragma omp parallel for reduction(+:empiricalMean)
    for (unsigned int i = a; i < b; i++) {
        empiricalMean += y[i];
    }
    empiricalMean = empiricalMean / (double) (b - a);

#pragma omp parallel for reduction(+:totalSum)
    for (unsigned int t = a + 1; t < b; t++) {
        totalSum += (y[t] - empiricalMean) * (y[t] - empiricalMean);
    }
    return totalSum;
}

/**
* This function calculates the empirical cumulative distribution function (ECDF) for a given sample.
*
* @param sample The array of double values representing the sample.
* @param sampleSize The size of the sample.
* @param x The value at which to evaluate the ECDF.
*
* @return The value of the ECDF at the given value x.
*/
double ecdf(const double *sample, unsigned int sampleSize, double x) {
    int n = 0;
    for (int i = 0; i < sampleSize; ++i) {
        if (sample[i] < x) {
            ++n;
        }
    }
    return (double) n / double(sampleSize);
}


/**
 * Kolmogorov Smirnov hypothesis test
 * @param m
 * @param n
 * @param d
 * @param size
 * @param cpt
 * @param significance_level
 * @return
 */
bool kolmogorov_smirnov(unsigned int m, unsigned int n, double *d, unsigned int size, unsigned int cpt,
                        double significance_level) {
#ifdef IsDebug
    fprintf(out, "Creating set of existing values in sample..."); fflush(stdout);
#endif //IsDebug
    std::vector<double> values_first;
    std::vector<double> values_second;
    for (int i = 1; i < n; ++i) {
        if (std::find(values_first.begin(), values_first.end(), d[i]) == values_first.end()) {
            values_first.push_back(d[i]);
        }
    }
    for (unsigned int i = cpt; i < size; ++i) {
        if (std::find(values_second.begin(), values_second.end(), d[i]) == values_second.end()) {
            values_second.push_back(d[i]);
        }
    }
    values_second.insert(values_second.end(), values_first.begin(), values_first.end());
    std::sort(values_second.begin(), values_second.end());
#ifdef IsDebug
    fprintf(out, "DONE\n");
    fprintf(out, "Create two samples for test execution..."); fflush(stdout);
#endif //IsDebug
    double *arr1 = (double *) malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) {
        arr1[i] = d[i];
    }

    double *arr2 = (double *) malloc(sizeof(double) * (size - cpt));
    for (int i = cpt; i < size; i++) {
        arr2[i - cpt] = d[i];
    }
#ifdef IsDebug
    fprintf(out, "DONE\n");
    fprintf(out, "Find supremum...");fflush(stdout);
#endif //IsDebug
    double supremum = 0.;
    for (int i = 0; i < values_second.size(); ++i) {
        double val1 = ecdf(arr1, n, values_second[i]);
        double val2 = ecdf(arr2, (size - cpt), values_second[i]);
        if (supremum < abs(val1 - val2)) {
            supremum = abs(val1 - val2);
        }
    }
#ifdef IsDebug
    fprintf(out, "DONE\n");
#endif //IsDebug

    free(arr1);
    free(arr2);

    double logarithm = -1. * log(significance_level / 2.);
    double fraction = (1. + ((double) m / (double) n)) / (2. * (double) m);
    double p_val = sqrt(logarithm * fraction);
#ifdef IsDebug
    fprintf(out, "\nComparison: %f > %f\n", supremum, p_val);
#endif //IsDebug
    if (supremum > p_val) {
#ifdef IsDebug
        fprintf(out, "H0 rejected: Change point at position %d is real changepoint!\n", cpt);
#endif //IsDebug
        return true;
    } else {
#ifdef IsDebug
        fprintf(out, "H0 accepted: No real changepoint at position %d\n", cpt);
#endif //IsDebug
        return false;
    }
}

/**
 * Opt algorithm:
 * Creates for each subvector of d the measure of 'equality', the more all the values of a subvector is equal, the lower is its cost-value.
 * Finding changepoints: get index i with cost(0, i) + cost(i+1, n) is minimal.
 *
 * @param d     The array of double values representing the input data.
 * @param size  The size of the array.
 *
 * @return      A vector containing the index of the detected changepoints.
 */
std::vector<unsigned int> Opt(double *d, int size) {
    double **C_1 = (double **) malloc(sizeof(double *) * size);
    for (int i = 0; i < size; i++) {
        C_1[i] = (double *) calloc(size, sizeof(double));
    }

    // u/v are not dependent on each other
    // + d is read-only in cost()
    // -> parallelization possible
    // compute cost for each subvector
#pragma omp parallel for
    for (unsigned int u = 0; u < size; ++u) {
        for (unsigned int v = u + 1; v < size; ++v) {
            C_1[u][v] = cost(u, v, d);
        }
    }

    // find optimal changepoints
    std::vector<unsigned int> L;
    int s = size - 1;
    unsigned int t_star = 1;
    double min_val = C_1[1][t_star] + C_1[t_star + 1][s];
    for (int t = 2; t < s; ++t) {
        if (C_1[1][t] + C_1[t + 1][s] < min_val) {
            min_val = C_1[1][t] + C_1[t + 1][s];
            t_star = t;
        }
    }
    L.push_back(t_star);

    for (int i = 0; i < size; i++) {
        free(C_1[i]);
    }
    free(C_1);
    return L;
}

/**
 * Calculates the median of an array of doubles.
 *
 * @param data The input array of doubles.
 * @param size The size of the array.
 *
 * @return The median value.
 */
double calculateMedian(double *data, int size) {
    std::sort(data, data + size);
    if (size % 2 == 0) {
        return (data[size / 2 - 1] + data[size / 2]) / 2.0;
    } else {
        return data[size / 2];
    }
}

/**
 * Performs a two-sided median test to check if the change point is significant.
 *
 * @param data          The input array of doubles.
 * @param size          The size of the array.
 * @param changePoint   The change point to be tested.
 * @warning Significance level = 5%.
 *
 * @return              True if the change point is statistically significant, false otherwise.
 */
bool twoSidedMedianTest(double *data, int size, int changePoint) {
    double *group1 = new double[size];
    double *group2 = new double[size];
    int size1 = 0, size2 = 0;

    for (int i = 0; i < size; ++i) {
        if (data[i] <= changePoint) {
            group1[size1++] = data[i];
        } else {
            group2[size2++] = data[i];
        }
    }

    double median1 = calculateMedian(group1, size1);
    double median2 = calculateMedian(group2, size2);

    // Calculate the number of observations above and below the medians
    int n1_plus = 0, n1_minus = 0, n2_plus = 0, n2_minus = 0;
    for (int i = 0; i < changePoint; i++) {
        if (data[i] > median1)
            n1_plus++;
        else if (data[i] < median1)
            n1_minus++;
    }
    for (int i = changePoint; i < size; i++) {
        if (data[i] > median2)
            n2_plus++;
        else if (data[i] < median2)
            n2_minus++;
    }

    // Calculate the test statistic Q
    double Q = std::pow(n1_plus - n1_minus, 2) / static_cast<double>(changePoint) +
               std::pow(n2_plus - n2_minus, 2) / static_cast<double>(size - changePoint);

    // Compare Q with the critical value and return the result
    // significance level = 0.05 = 5%
    double criticalValue = 0.05;
    return Q > criticalValue;
}

/**
 * This function detects a change point in a two-dimensional array of unsigned integers.
 *
 * @param y          The two-dimensional array of unsigned integers representing the data.
 * @param size       The number of rows in the array.
 * @param innerSize  The number of columns in the array.
 *
 * @return           The index of the detected change point, or -1 if no change point is detected.
 */
int detectChangePoint(unsigned int **y, int size, int innerSize) {
    // Allocate memory for y_min array
    unsigned int *y_min = (unsigned int *) malloc(sizeof(unsigned int) * innerSize);

    // Find the minimum value in each column of y and store it in y_min
    for (int j = 0; j < innerSize; j++) {
        unsigned int min = y[0][j];
        for (int i = 0; i < size; i++) {
            if (y[i][j] < min) {
                min = y[i][j];
            }
        }
        y_min[j] = min;
    }

    // Allocate memory for y_prime array
    unsigned int **y_prime = (unsigned int **) malloc(sizeof(unsigned int *) * size);

    // Compute y_prime by subtracting (y_min - 1) from each element in y
    for (int i = 0; i < size; i++) {
        y_prime[i] = (unsigned int *) malloc(sizeof(unsigned int) * innerSize);

        for (int j = 0; j < innerSize; j++) {
            y_prime[i][j] = y[i][j] - (y_min[j] - 1); // 1 -> (1 ... 1)^T
        }
    }

    // y_min no longer used, free the memory
    free(y_min);

    // Allocate memory for d array
    double *d = (double *) malloc(sizeof(double) * size);

    // Compute d array by calculating the Euclidean distance of each row in y_prime
    for (int i = 0; i < size; i++) {
        double sum_scalar = 0.;
        for (int j = 0; j < innerSize; j++) {
            sum_scalar += (y_prime[i][j] - 1) * (y_prime[i][j] - 1);
        }
        d[i] = sqrt(sum_scalar);
    }

    // Perform post-processing on d array
    postProcessing(d, size, 4);

    // Use 2-sided median test to find potential change points
    std::vector<unsigned int> result = Opt(d, size);
    int firstCP = -1;

    // Print the potential change points
//    printf("potential changepoints:\n");
//    for (unsigned int cp: result) {
//        printf("%d\n", cp);
//    }

    // Check each potential change point using the 2-sided median test
    for (unsigned int potCP: result) {
        //printf("potCP\t%d\n", potCP);
        if (potCP > 0 && potCP < size - 1) {
            bool isCP = twoSidedMedianTest(d, size, potCP);
//            unsigned int n = potCP - 1;
//            unsigned int m = size - potCP;
//            bool isCP = kolmogorov_smirnov(m, n, d, size, potCP, 0.0000001);
            if (isCP) {
                firstCP = (int) potCP;
                break;
            }
        }
    }

    free(d);
    for (int i = 0; i < size; i++) {
        free(y_prime[i]);
    }
    free(y_prime);

    // Return the detected change point index or -1 if no change point is detected
    if (firstCP == -1) {
#ifdef IsDebug
        fprintf(out, "No real changepoint detected at all!\n");
#endif //IsDebug
        return -1;
    } else {
#ifdef IsDebug
        fprintf(out, "Returning change point index %d\n", firstCP);
#endif
    }
    return firstCP;
}

#endif //CUDATEST_EVAL_H1
