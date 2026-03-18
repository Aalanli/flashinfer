#pragma once
#include <cuda_fp16.h>
#include <stdio.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                << cudaGetErrorString(err) << std::endl;                    \
      exit(EXIT_FAILURE);                                                   \
    }                                                                       \
  } while (0)

__device__ __forceinline__ auto convert_to_uint32_v2(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key =
      (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ inline int ld_acquire(const int32_t* __restrict__ addr) {
  int res;
  asm volatile("ld.acquire.cta.b32 %0, [%1];" : "=r"(res) : "l"(addr) : "memory");
  return res;
}

// topk kernel v2
template <typename T>
__host__ __device__ __forceinline__ T divup(T a, T b) {
  return (a + b - 1) / b;
}

// PTX functions
__device__ __forceinline__ uint32_t getLaneId() {
  uint32_t laneId;
  asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
  return laneId;
}

__device__ __forceinline__ uint32_t getWarpId() {
  uint32_t warpid;
  asm("mov.u32 %0, %%warpid;" : "=r"(warpid));
  return warpid;
}

__device__ __forceinline__ unsigned getLaneMaskLt() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
  return mask;
}

__device__ __forceinline__ unsigned getLaneMaskLe() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_le;" : "=r"(mask));
  return mask;
}
__device__ __forceinline__ unsigned getLaneMaskGt() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_gt;" : "=r"(mask));
  return mask;
}

__device__ __forceinline__ unsigned getLaneMaskGe() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_ge;" : "=r"(mask));
  return mask;
}

// Warp scans
__device__ __forceinline__ uint32_t InclusiveWarpScan(uint32_t val) {
#pragma unroll
  for (int i = 1; i <= 16; i <<= 1)  // 16 = LANE_COUNT >> 1
  {
    const uint32_t t = __shfl_up_sync(0xffffffff, val, i, 32);
    if (getLaneId() >= i) val += t;
  }

  return val;
}

template <int num_threads>
__device__ __forceinline__ uint32_t InclusiveWarpDownScan(uint32_t val) {
#pragma unroll
  for (int i = 1; i <= (num_threads >> 1); i <<= 1)  // 16 = LANE_COUNT >> 1
  {
    const uint32_t t = __shfl_down_sync(0xffffffff, val, i, 32);
    if (getLaneId() < num_threads - i) val += t;
  }

  return val;
}

__device__ __forceinline__ void WLMS(uint8_t val, int* shared_hist, bool active_mask = true) {
  if (active_mask) {
    atomicAdd(shared_hist + val, 1);
  }
  return;
  unsigned warpFlags = __ballot_sync(0xffffffff, active_mask);
  for (int k = 0; k < 8; ++k) {
    const bool t2 = val >> k & 1;
    warpFlags &= (t2 ? 0 : 0xffffffff) ^ __ballot_sync(0xffffffff, t2);
  }

  // now warpFlags contains the bit pattern of all lanes, where 1 = the val of
  // that lane is the same as the current val bits == 0 if this is the first
  // lane holding val (out of the lanes that hold the same val)
  const uint32_t bits = __popc(warpFlags & getLaneMaskLt());

  // __popc(warpFlags) counts the number of lanes that hold the same val, only
  // the first such lane increments the shared histogram
  if (bits == 0 && active_mask) {
    atomicAdd(shared_hist + val, __popc(warpFlags));
  }
}

__device__ inline float2 explicit_load_float2(const float2* ptr) {
  float2 res;
  asm("ld.global.nc.L1::no_allocate.L2::256B.v2.f32 {%0,%1}, [%2];"
      : "=f"(res.x), "=f"(res.y)
      : "l"(ptr)
      : "memory");
  return res;
}

__device__ __forceinline__ void reduce_shared(int* addr, int val) {
  asm("red.relaxed.cta.shared::cta.add.s32 [%0], %1;" ::"l"(addr), "r"(val) : "memory");
}
template <auto* kernel_func, size_t smem_bytes>
void setup_kernel_smem_once() {
  static const cudaError_t result = []() -> cudaError_t {
    auto func_ptr = kernel_func;

    return cudaFuncSetAttribute(func_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  }();
  CUDA_CHECK(result);
}
template <auto* kernel_func>
void setup_non_portable_clusters_once() {
  static const cudaError_t result = []() -> cudaError_t {
    auto func_ptr = kernel_func;
    return cudaFuncSetAttribute(func_ptr, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  }();
  if (result != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s \n", cudaGetErrorString(result));
    throw std::runtime_error("cuda set non portable cluster error");
  }
}

__device__ inline void mbarrier_wait(uint64_t* mbar_addr, int phase) {
  uint32_t ticks = 0x989680;  // this is optional
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "LAB_WAIT:\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], "
      "%1, %2;\n\t"
      "@P1 bra.uni DONE;\n\t"
      "bra.uni LAB_WAIT;\n\t"
      "DONE:\n\t"
      "}" ::"l"(mbar_addr),
      "r"(phase), "r"(ticks));
}

__device__ inline void mbarrier_init(uint64_t* mbar_addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"l"(mbar_addr), "r"(count));
}

__device__ inline void mbarrier_arrive(uint64_t* mbar_addr, int expect_size) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"l"(mbar_addr),
      "r"(expect_size)
      : "memory");
}

__device__ inline void mbarrier_cpy(int* src, int* dst, int num_bytes, uint64_t* mbar) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];" ::"l"(dst),
      "l"(src), "r"(num_bytes), "l"(mbar)
      : "memory");
}

__device__ inline int mbarrier_arrive_no_expect_tx(uint64_t* mbar) {
  int tok;
  asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 %0, [%1];"
               : "=r"(tok)
               : "l"(mbar)
               : "memory");
  return tok;
}

__host__ __device__ inline void print_bits(uint32_t bits) {
  for (int i = 0; i < 32; i++) {
    printf("%d", (bits >> (31 - i)) & 1);
  }
}
