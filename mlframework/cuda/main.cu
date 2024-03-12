
/*
source: https://github.com/siboehm/SGEMM_CUDA
Google Colab
!nvcc -lcublas main.cu -o main
!./main
*/

#ifndef _COLORS_
#define _COLORS_

/* FOREGROUND */
#define RST  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

#define FRED(x) KRED x RST
#define FGRN(x) KGRN x RST
#define FYEL(x) KYEL x RST
#define FBLU(x) KBLU x RST
#define FMAG(x) KMAG x RST
#define FCYN(x) KCYN x RST
#define FWHT(x) KWHT x RST

#define BOLD(x) "\x1B[1m" x RST
#define UNDL(x) "\x1B[4m" x RST

#endif  /* _COLORS_ */

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>
#include <sys/time.h>
#include <cublas_v2.h>
#include <library_types.h>
#include <cuda_runtime.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <iomanip>

void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

void randomize_matrix(float *mat, int N) {
  // NOTICE: Use gettimeofday instead of srand((unsigned)time(NULL)); the time
  // precision is too low and the same random number is generated.
  struct timeval time {};
  gettimeofday(&time, nullptr);
  srand(time.tv_usec);
  for (int i = 0; i < N; i++) {
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
    mat[i] = tmp;
  }
}

bool verify_matrix(float *matRef, float *matOut, int N) {
  double diff = 0.0;
  int i;
  for (i = 0; i < N; i++) {
    diff = std::fabs(matRef[i] - matOut[i]);
    if (diff > 0.01) {
      printf(FRED("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n"),
             matRef[i], matOut[i], diff, i);
      return false;
    }
  }
  return true;
}

void print_matrix(const float *A, int M, int N, std::ofstream &fs) {
  int i;
  fs << std::setprecision(2)
     << std::fixed; // Set floating-point precision and fixed notation
  fs << "[";
  for (i = 0; i < M * N; i++) {
    if ((i + 1) % N == 0)
      fs << std::setw(5) << A[i]; // Set field width and write the value
    else
      fs << std::setw(5) << A[i] << ", ";
    if ((i + 1) % N == 0) {
      if (i + 1 < M * N)
        fs << ";\n";
    }
  }
  fs << "]\n";
}

void runCublasFP32(cublasHandle_t handle, int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C) {
  // cuBLAS uses column-major order. So we change the order of our row-major A &
  // B, since (B^T*A^T)^T = (A*B)
  // This runs cuBLAS in full fp32 mode
  if (CUBLAS_STATUS_SUCCESS != cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F,
               N, A, CUDA_R_16F, K, &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP))  {
                  printf("cublasGemmEx failed\n");
                  exit(-1);
               }
}

void run_kernel(int kernel_num, int M, int N, int K, float alpha, float *A,
                float *B, float beta, float *C, cublasHandle_t handle) {
  switch (kernel_num) {
  case 0:
    runCublasFP32(handle, M, N, K, alpha, A, B, beta, C);
    break;
  // case 1:
  //   run_sgemm_naive(M, N, K, alpha, A, B, beta, C);
    // break;
  // case 2:
  //   run_sgemm_coalesce(M, N, K, alpha, A, B, beta, C);
  //   break;
  // case 3:
  //   run_sgemm_shared_mem_block(M, N, K, alpha, A, B, beta, C);
  //   break;
  // case 4:
  //   runSgemm1DBlocktiling(M, N, K, alpha, A, B, beta, C);
  //   break;
  // case 5:
  //   runSgemm2DBlocktiling(M, N, K, alpha, A, B, beta, C);
  //   break;
  // case 6:
  //   runSgemmVectorize(M, N, K, alpha, A, B, beta, C);
  //   break;
  // case 7:
  //   runSgemmResolveBankConflicts(M, N, K, alpha, A, B, beta, C);
  //   break;
  // case 8:
  //   runSgemmResolveBankExtraCol(M, N, K, alpha, A, B, beta, C);
  //   break;
  // case 9:
  //   runSgemmAutotuned(M, N, K, alpha, A, B, beta, C);
  //   break;
  // case 10:
  //   runSgemmWarptiling(M, N, K, alpha, A, B, beta, C);
  //   break;
  // case 11:
  //   runSgemmDoubleBuffering(M, N, K, alpha, A, B, beta, C);
  //   break;
  // case 12:
  //   runSgemmDoubleBuffering2(M, N, K, alpha, A, B, beta, C);
  //   break;
  default:
    throw std::invalid_argument("Unknown kernel number");
  }
}

const std::string errLogFile = "matrixValidationFailure.txt";

int main(int argc, char **argv) {
    cublasHandle_t handle;
    if (cublasCreate(&handle)) {
      std::cerr << "Create cublas handle error." << std::endl;
      exit(EXIT_FAILURE);
    };

    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    
    long m, n, k, max_size;
    max_size = 4096;

    float alpha = 0.5, beta = 3.0; // GEMM input parameters, C=α*AB+β*C

    float *A = nullptr, *B = nullptr, *C = nullptr,
      *C_ref = nullptr; // host matrices
    float *dA = nullptr, *dB = nullptr, *dC = nullptr,
      *dC_ref = nullptr; // device matrices

    A = (float *)malloc(sizeof(float) * max_size * max_size);
    B = (float *)malloc(sizeof(float) * max_size * max_size);
    C = (float *)malloc(sizeof(float) * max_size * max_size);
    C_ref = (float *)malloc(sizeof(float) * max_size * max_size);

    randomize_matrix(A, max_size * max_size);
    randomize_matrix(B, max_size * max_size);
    randomize_matrix(C, max_size * max_size);

    cudaCheck(cudaMalloc((void **)&dA, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&dB, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&dC, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(float) * max_size * max_size));
  
    cudaCheck(cudaMemcpy(dA, A, sizeof(float) * max_size * max_size,
                        cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB, B, sizeof(float) * max_size * max_size,
                        cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size,
                        cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC_ref, C, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));

    int kernel_num = 0;

    int size = 10;
    m = n = k = size;
    run_kernel(0, m, n, k, alpha, dA, dB, beta, dC_ref, handle); // cuBLAS
    run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle); // Executes the kernel, modifies the result matrix
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
    cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    if (!verify_matrix(C_ref, C, m * n)) {
      std::cout << FRED("Failed to pass the correctness verification against NVIDIA cuBLAS.") << std::endl;
      if (m <= 128) {
        std::cout << " Logging faulty output into " << errLogFile << "\n";
        std::ofstream fs;
        fs.open(errLogFile);
        fs << "A:\n";
        print_matrix(A, m, n, fs);
        fs << "B:\n";
        print_matrix(B, m, n, fs);
        fs << "C:\n";
        print_matrix(C, m, n, fs);
        fs << "Should:\n";
        print_matrix(C_ref, m, n, fs);
      }
      exit(EXIT_FAILURE);
    }

    cudaEventRecord(beg);
    // run kernel here
    run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
    cudaEventRecord(end);
    
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.; // Convert to seconds
    printf("Elapsed time: (%7.9f) s\n", elapsed_time);

    free(A);
    free(B);
    free(C);
    free(C_ref);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dC_ref);
    cublasDestroy(handle);

    return 0;
}