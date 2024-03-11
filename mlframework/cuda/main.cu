#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>


int main(int argc, char **argv) {
    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    cudaEventRecord(beg);
    // run kernel here
    cudaEventRecord(end);
    
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.; // Convert to seconds
    printf("Elapsed time: (%7.9f) s\n", elapsed_time);
}