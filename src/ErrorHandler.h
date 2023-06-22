//
// Created by nick- on 8/31/2022.
//

#ifndef CUDATEST_ERRORHANDLER_H
#define CUDATEST_ERRORHANDLER_H

# include <cstdio>
# include <cstdint>

int errorStatus = 0;

/**
 * Prints the error code and exits the test
 * @param errorCode int, corresponding to the error
 */
void printErrorCodeInformation(int errorCode) {
    if (errorCode != 0) {
        printf("Error occured during benchmark - Program will exit!\n");
        printf("Close other applications that might use the GPU and rerun the program\n");
    }
    printf("Errorcode %d: ", errorCode);
    switch(errorCode) {
        case 0:
            printf("No Error occured\n");
            break;
        case 1:
            printf("Error during malloc - Could not allocate memory\n");
            break;
        case 2:
            printf("Error during hipMalloc - Could not allocate memory on GPU\n");
            break;
        case 3:
            printf("Error during hipMemcpy - Could not copy memory to GPU\n");
            break;
        case 4:
            printf("Error during texture binding - Could not bind texture or create texture object with array\n");
            break;
        case 5:
            printf("Error during kernel execution\n");
            break;
        case 6:
            printf("Error during hipMemcpy - Could not copy memory from GPU\n");
            break;
        default:
            printf("Unexpected ErrorCode\n");
            break;
    }
    errorStatus = errorCode;
}

#endif //CUDATEST_ERRORHANDLER_H
