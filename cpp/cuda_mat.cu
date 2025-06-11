#include "cuda_mat.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define IDX2C(i, j, ld) (((j) * (ld)) + i)

void calc_tau_cublas(int n_nodes, const double *t, double *tau,
                     double coeff_theta, int n_iterations, int debug_level) {
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;

  double *devPtrA;
  double *devPtrLeft;
  double *devPtrLeftTemp;
  double *devPtrRight;
  double *devPtrRightTemp;
  double d_1 = 1.0;
  double d_0 = 0.0;

  double *t_theta = new double[n_nodes * n_nodes];
  double *B = new double[n_nodes * n_nodes];
  double *I = new double[n_nodes * n_nodes];
  double *print = new double[n_nodes * n_nodes];

  // Initialize t_array to be loaded as A matrix
  for (int j = 0; j < n_nodes; j++) {
    for (int i = 0; i < n_nodes; i++) {
      if (t[i * n_nodes + j] > 0) {
        t_theta[IDX2C(i, j, n_nodes)] =
            std::pow(t[i * n_nodes + j], -1 * coeff_theta);
      } else {
        t_theta[IDX2C(i, j, n_nodes)] = 0;
      }
    }
  }

  for (int j = 0; j < n_nodes; j++) {
    for (int i = 0; i < n_nodes; i++) {
      if (i == j) {
        I[IDX2C(i, j, n_nodes)] = 1;
      } else {
        I[IDX2C(i, j, n_nodes)] = 0;
      }
    }
  }

  cudaStat = cudaMalloc((void **)&devPtrA, n_nodes * n_nodes * sizeof(*t));
  if (cudaStat != cudaSuccess) {
    printf("device memory allocation for devPrtA failed\n");
  }
  cudaStat = cudaMalloc((void **)&devPtrLeft, n_nodes * n_nodes * sizeof(*t));
  if (cudaStat != cudaSuccess) {
    printf("device memory allocation for devPrtLeft failed\n");
  }
  cudaStat = cudaMalloc((void **)&devPtrRight, n_nodes * n_nodes * sizeof(*t));
  if (cudaStat != cudaSuccess) {
    printf("device memory allocation for devPtrRight failed\n");
  }
  cudaStat =
      cudaMalloc((void **)&devPtrLeftTemp, n_nodes * n_nodes * sizeof(*t));
  if (cudaStat != cudaSuccess) {
    printf("device memory allocation for devPtrLeftTemp failed\n");
  }
  cudaStat =
      cudaMalloc((void **)&devPtrRightTemp, n_nodes * n_nodes * sizeof(*t));
  if (cudaStat != cudaSuccess) {
    printf("device memory allocation for devPtrRightTemp failed\n");
  }

  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("CUBLAS initialization failed\n");
  }

  // load matrix A into GPU memory
  stat = cublasSetMatrix(n_nodes, n_nodes, sizeof(*t), t_theta, n_nodes,
                         devPtrA, n_nodes);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("data download to devPtrA failed\n");
  }
  stat = cublasSetMatrix(n_nodes, n_nodes, sizeof(*t), I, n_nodes, devPtrLeft,
                         n_nodes);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("data download to devPtrLeft failed\n");
  }
  stat = cublasSetMatrix(n_nodes, n_nodes, sizeof(*t), t_theta, n_nodes,
                         devPtrRight, n_nodes);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("data download to devPtrRight failed\n");
  }

  for (int iter = 0; iter < n_iterations; iter++) {

    stat = cublasGetMatrix(n_nodes, n_nodes, sizeof(*t), devPtrRight, n_nodes,
                           print, n_nodes);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      printf("data upload from devPtrRight failed\n");
    }

    // std::cout << "A_n matrix:" << std::endl;
    // print_cu_matrix(n_nodes, print);

    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_nodes, n_nodes, &d_1,
                devPtrLeft, n_nodes, &d_1, devPtrRight, n_nodes, devPtrLeftTemp,
                n_nodes);

    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_nodes, n_nodes, &d_1,
                devPtrLeftTemp, n_nodes, &d_0, devPtrLeftTemp, n_nodes,
                devPtrLeft, n_nodes);

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_nodes, n_nodes, n_nodes,
                &d_1, devPtrRight, n_nodes, devPtrA, n_nodes, &d_0,
                devPtrRightTemp, n_nodes);

    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_nodes, n_nodes, &d_1,
                devPtrRightTemp, n_nodes, &d_0, devPtrRightTemp, n_nodes,
                devPtrRight, n_nodes);
  }

  stat = cublasGetMatrix(n_nodes, n_nodes, sizeof(*t), devPtrLeft, n_nodes, B,
                         n_nodes);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("data upload from devPtrLeft failed\n");
  }

  for (int j = 0; j < n_nodes; j++) {
    for (int i = 0; i < n_nodes; i++) {
      if (B[IDX2C(i, j, n_nodes)] > 0) {
        tau[i * n_nodes + j] =
            std::pow(B[IDX2C(i, j, n_nodes)], -1 / coeff_theta);
      } else {
        tau[i * n_nodes + j] =
            std::pow(B[IDX2C(i, j, n_nodes)], -1 / coeff_theta);
        // tau[i * n_nodes + j] = 0;
      }
    }
  }

  delete[] t_theta;
  delete[] B;
  delete[] I;

  cudaFree(devPtrA);
  cudaFree(devPtrLeft);
  cudaFree(devPtrRight);
  cudaFree(devPtrLeftTemp);
  cudaFree(devPtrRightTemp);
}
