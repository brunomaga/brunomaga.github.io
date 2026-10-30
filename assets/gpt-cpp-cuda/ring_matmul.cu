// The ring algorithm of the distributed matrix multiplication on the GPUs of a node, with NCCL and cuBLAS: each GPU
// multiplies its current block of B while it sends this block to the previous GPU and receives the next one
#include <algorithm>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <nccl.h>

using dt = __nv_bfloat16;

// C_r (M/p x O) = A_r (M/p x N) B (N x O) on the rank r of a ring of p ranks, where the rank r holds the row blocks r
// of A and C, and starts with the row block r of B in B_cur. C accumulates the p products in fp32
void ring_matmul(const dt* A, dt* B_cur, dt* B_next, float* C, int M, int N, int O, int r, int p,
                 cublasHandle_t handle, ncclComm_t comm, cudaStream_t comm_stream) {
  int m = M / p, n = N / p;           // rows of the blocks of A and C, and of the blocks of B
  for (int step = 0; step < p; step++) {
    int k = (r + step) % p;           // the rank holds the row block k of B at this step
    if (step < p - 1) {               // it sends this block to the rank r-1, and receives the next one from the rank r+1
      ncclGroupStart();
      ncclSend(B_cur, (size_t) n*O, ncclBfloat16, (r - 1 + p) % p, comm, comm_stream);
      ncclRecv(B_next, (size_t) n*O, ncclBfloat16, (r + 1) % p, comm, comm_stream);
      ncclGroupEnd();
    }
    // C_r = C_r + A_rk B_k, where A_rk is the column block k of A_r (n columns, in rows of N values)
    float alpha = 1, beta = step == 0 ? 0 : 1;
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, O, m, n, &alpha, B_cur, CUDA_R_16BF, O, A + k*n, CUDA_R_16BF, N,
                 &beta, C, CUDA_R_32F, O, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize();          // waits for the product and for the transfers of this step
    std::swap(B_cur, B_next);
  }
}

int main(int argc, char* argv[]) {
  int p;
  cudaGetDeviceCount(&p);
  int size = argc > 1 ? std::atoi(argv[1]) : 8192, M = size, N = size, O = size;   // multiples of p
  std::vector<dt> a((long) M*N), b((long) N*O);
  for (dt& v : a) v = (float) std::rand() / RAND_MAX - 0.5f;
  for (dt& v : b) v = (float) std::rand() / RAND_MAX - 0.5f;
  std::vector<float> c((long) M*O);
  std::vector<ncclComm_t> comms(p);
  ncclCommInitAll(comms.data(), p, nullptr);

  // one thread per GPU, that runs the ring twice, and times the second run
  double ms = 0;
  std::barrier sync(p);
  std::vector<std::thread> ranks;
  for (int r = 0; r < p; r++)
    ranks.emplace_back([&, r] {
      int m = M / p, n = N / p;
      cudaSetDevice(r);
      dt *A, *B_cur, *B_next;
      float* C;
      cudaMalloc(&A, (long) m*N * sizeof(dt));
      cudaMalloc(&B_cur, (long) n*O * sizeof(dt));
      cudaMalloc(&B_next, (long) n*O * sizeof(dt));
      cudaMalloc(&C, (long) m*O * sizeof(float));
      cudaMemcpy(A, a.data() + (long) r*m*N, (long) m*N * sizeof(dt), cudaMemcpyHostToDevice);
      cublasHandle_t handle;
      cublasCreate(&handle);
      cudaStream_t compute, comm;
      cudaStreamCreate(&compute);
      cudaStreamCreate(&comm);
      cublasSetStream(handle, compute);
      for (int run = 0; run < 2; run++) {
        cudaMemcpy(B_cur, b.data() + (long) r*n*O, (long) n*O * sizeof(dt), cudaMemcpyHostToDevice);
        sync.arrive_and_wait();
        auto start = std::chrono::steady_clock::now();
        ring_matmul(A, B_cur, B_next, C, M, N, O, r, p, handle, comms[r], comm);
        sync.arrive_and_wait();
        if (r == 0)
          ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
      }
      cudaMemcpy(c.data() + (long) r*m*O, C, (long) m*O * sizeof(float), cudaMemcpyDeviceToHost);
    });
  for (std::thread& t : ranks)
    t.join();

  // the same product on the GPU 0 alone, and the largest difference to the ring, relative to the largest value
  cudaSetDevice(0);
  dt *A, *B;
  float* C;
  cudaMalloc(&A, a.size() * sizeof(dt));
  cudaMalloc(&B, b.size() * sizeof(dt));
  cudaMalloc(&C, c.size() * sizeof(float));
  cudaMemcpy(A, a.data(), a.size() * sizeof(dt), cudaMemcpyHostToDevice);
  cudaMemcpy(B, b.data(), b.size() * sizeof(dt), cudaMemcpyHostToDevice);
  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1, beta = 0;
  double ms_one = 0;
  for (int run = 0; run < 2; run++) {
    auto start = std::chrono::steady_clock::now();
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, O, M, N, &alpha, B, CUDA_R_16BF, O, A, CUDA_R_16BF, N,
                 &beta, C, CUDA_R_32F, O, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize();
    ms_one = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  }
  std::vector<float> ref(c.size());
  cudaMemcpy(ref.data(), C, ref.size() * sizeof(float), cudaMemcpyDeviceToHost);
  double diff = 0, max_ref = 0;
  for (long i = 0; i < (long) c.size(); i++) {
    diff = std::max(diff, (double) std::abs(c[i] - ref[i]));
    max_ref = std::max(max_ref, (double) std::abs(ref[i]));
  }
  std::printf("ring on %d GPUs: %.2f ms, one GPU: %.2f ms, M = N = O = %d, difference %.2e\n", p, ms, ms_one, size, diff / max_ref);
}
