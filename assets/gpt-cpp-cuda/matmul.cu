// The matrix multiplication kernels of the CUDA C++ Best Practices Guide, the tiled kernel for any N, and cuBLAS,
// with bf16 values: their runtimes, and their differences to cuBLAS
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cublas_v2.h>
#include <cuda_bf16.h>

using dt = __nv_bfloat16;
constexpr int TILE_DIM = 32;   // tiles of TILE_DIM x TILE_DIM values, with one thread per value

// C (M x O) = A (M x N) B (N x O) for N = TILE_DIM, with one thread per value of C, that reads A and B from global memory
__global__ void simpleMultiply(const dt* A, const dt* B, dt* C, int O) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0;
  for (int i = 0; i < TILE_DIM; i++)
    sum += (float) A[row*TILE_DIM + i] * (float) B[i*O + col];
  C[row*O + col] = sum;
}

// the same, with the tile of A read into shared memory, with coalesced reads
__global__ void coalescedMultiply(const dt* A, const dt* B, dt* C, int O) {
  __shared__ dt aTile[TILE_DIM][TILE_DIM];
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0;
  aTile[threadIdx.y][threadIdx.x] = A[row*TILE_DIM + threadIdx.x];
  __syncwarp();   // each warp reads only the row of aTile that it wrote
  for (int i = 0; i < TILE_DIM; i++)
    sum += (float) aTile[threadIdx.y][i] * (float) B[i*O + col];
  C[row*O + col] = sum;
}

// the same, with the tiles of A and B read into shared memory
__global__ void sharedABMultiply(const dt* A, const dt* B, dt* C, int O) {
  __shared__ dt aTile[TILE_DIM][TILE_DIM], bTile[TILE_DIM][TILE_DIM];
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0;
  aTile[threadIdx.y][threadIdx.x] = A[row*TILE_DIM + threadIdx.x];
  bTile[threadIdx.y][threadIdx.x] = B[threadIdx.y*O + col];
  __syncthreads();
  for (int i = 0; i < TILE_DIM; i++)
    sum += (float) aTile[threadIdx.y][i] * (float) bTile[i][threadIdx.x];
  C[row*O + col] = sum;
}

// C (M x O) = A (M x N) B (N x O) for M, N and O multiples of TILE_DIM: each thread block computes a tile of C as
// the sum of the products of the N / TILE_DIM pairs of tiles of A and B, read one pair at a time into shared memory
__global__ void tiledMultiply(const dt* A, const dt* B, dt* C, int N, int O) {
  __shared__ dt aTile[TILE_DIM][TILE_DIM], bTile[TILE_DIM][TILE_DIM];
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0;
  for (int t = 0; t < N; t += TILE_DIM) {
    aTile[threadIdx.y][threadIdx.x] = A[row*N + t + threadIdx.x];
    bTile[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y)*O + col];
    __syncthreads();
    for (int i = 0; i < TILE_DIM; i++)
      sum += (float) aTile[threadIdx.y][i] * (float) bTile[i][threadIdx.x];
    __syncthreads();   // all threads have read the tiles before the next ones overwrite them
  }
  C[row*O + col] = sum;
}

// C (M x O) = A (M x N) B (N x O) in row-major order with cuBLAS, as in gpt_cuda.cu
void cublasMultiply(cublasHandle_t handle, const dt* A, const dt* B, dt* C, int M, int N, int O) {
  float alpha = 1, beta = 0;
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, O, M, N, &alpha, B, CUDA_R_16BF, O, A, CUDA_R_16BF, N,
               &beta, C, CUDA_R_16BF, O, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

// average runtime of f, in milliseconds, after a first run that is not timed
template <typename F> double runtime(F f, int iters = 10) {
  f();
  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();
  for (int it = 0; it < iters; it++)
    f();
  cudaDeviceSynchronize();
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / iters;
}

// the largest difference between the values of C and ref (n values on the GPU), relative to the largest value of ref
double difference(const dt* C, const dt* ref, long n) {
  std::vector<dt> c(n), r(n);
  cudaMemcpy(c.data(), C, n * sizeof(dt), cudaMemcpyDeviceToHost);
  cudaMemcpy(r.data(), ref, n * sizeof(dt), cudaMemcpyDeviceToHost);
  double diff = 0, max_ref = 0;
  for (long i = 0; i < n; i++) {
    diff = std::max(diff, (double) std::abs((float) c[i] - (float) r[i]));
    max_ref = std::max(max_ref, (double) std::abs((float) r[i]));
  }
  return diff / max_ref;
}

int main(int argc, char* argv[]) {
  // the sizes M, N and O of the tiled kernel and of cuBLAS, multiples of TILE_DIM
  int M = argc > 3 ? std::atoi(argv[1]) : 4096, N = argc > 3 ? std::atoi(argv[2]) : 4096, O = argc > 3 ? std::atoi(argv[3]) : 4096;
  cublasHandle_t handle;
  cublasCreate(&handle);
  std::vector<dt> a((long) M*N), b((long) N*O);
  for (dt& v : a) v = (float) std::rand() / RAND_MAX - 0.5f;
  for (dt& v : b) v = (float) std::rand() / RAND_MAX - 0.5f;
  dt *A, *B, *C, *ref;
  cudaMalloc(&A, a.size() * sizeof(dt));
  cudaMalloc(&B, b.size() * sizeof(dt));
  cudaMalloc(&C, (long) M*O * sizeof(dt));
  cudaMalloc(&ref, (long) M*O * sizeof(dt));
  cudaMemcpy(A, a.data(), a.size() * sizeof(dt), cudaMemcpyHostToDevice);
  cudaMemcpy(B, b.data(), b.size() * sizeof(dt), cudaMemcpyHostToDevice);
  dim3 threads(TILE_DIM, TILE_DIM), tiles(O / TILE_DIM, M / TILE_DIM);

  // the three kernels of the guide, for N = TILE_DIM: the first M*TILE_DIM values of A are read as an M x TILE_DIM
  // matrix, and the first TILE_DIM rows of B as a TILE_DIM x O matrix
  cublasMultiply(handle, A, B, ref, M, TILE_DIM, O);
  double ms = runtime([&] { simpleMultiply<<<tiles, threads>>>(A, B, C, O); });
  std::printf("simpleMultiply,    N = %d: %.3f ms, difference to cuBLAS %.4f\n", TILE_DIM, ms, difference(C, ref, (long) M*O));
  ms = runtime([&] { coalescedMultiply<<<tiles, threads>>>(A, B, C, O); });
  std::printf("coalescedMultiply, N = %d: %.3f ms, difference to cuBLAS %.4f\n", TILE_DIM, ms, difference(C, ref, (long) M*O));
  ms = runtime([&] { sharedABMultiply<<<tiles, threads>>>(A, B, C, O); });
  std::printf("sharedABMultiply,  N = %d: %.3f ms, difference to cuBLAS %.4f\n", TILE_DIM, ms, difference(C, ref, (long) M*O));
  ms = runtime([&] { cublasMultiply(handle, A, B, ref, M, TILE_DIM, O); });
  std::printf("cuBLAS,            N = %d: %.3f ms\n", TILE_DIM, ms);

  // the tiled kernel and cuBLAS, for any N
  double tflop = 2.0 * M * N * O / 1e12;
  ms = runtime([&] { cublasMultiply(handle, A, B, ref, M, N, O); });
  std::printf("cuBLAS,        M = %d, N = %d, O = %d: %.3f ms, %.1f TFLOP/s\n", M, N, O, ms, tflop / ms * 1e3);
  ms = runtime([&] { tiledMultiply<<<tiles, threads>>>(A, B, C, N, O); });
  std::printf("tiledMultiply, M = %d, N = %d, O = %d: %.3f ms, %.1f TFLOP/s, difference to cuBLAS %.4f\n",
              M, N, O, ms, tflop / ms * 1e3, difference(C, ref, (long) M*O));
}
