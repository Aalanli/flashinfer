#include <flashinfer/mqa_histogram/common.cuh>
#include <cassert>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
constexpr int TopK = 2048;

__device__ __forceinline__ int cum_sum(int *s_hist_buf) {
  constexpr int RADIX = 256;
  const int warp_idx = threadIdx.x / 32;

  __shared__ int reduce_buf[8];
  int val = 0;
  if (threadIdx.x < RADIX) {
    val = s_hist_buf[threadIdx.x];
    val = InclusiveWarpDownScan<32>(val);
    if (getLaneId() == 0 && warp_idx < 8) {
      reduce_buf[warp_idx] = val;
    }
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    int cum_val =
        InclusiveWarpDownScan<8>(threadIdx.x < 8 ? reduce_buf[threadIdx.x] : 0);
    __syncwarp();
    if (threadIdx.x < 8) {
      reduce_buf[threadIdx.x] = cum_val;
    }
  }
  __syncthreads();
  if (warp_idx < 7) {
    val += reduce_buf[warp_idx + 1];
  }
  return val;
}

template <int NumBlocks, int Threads, int UnRollFactor, typename F>
__device__ __forceinline__ void run_vectorized_v2(const float2 *logits,
                                                  const int seq_len,
                                                  const int block_id, F f) {

  constexpr int ElemPerBlock = Threads;
  constexpr int GridStride = NumBlocks * ElemPerBlock;

  for (int t = 0; t < seq_len / (2 * GridStride * UnRollFactor); t++) {
#pragma unroll
    for (int j = 0; j < UnRollFactor; j++) {

      int offset = t * GridStride * UnRollFactor + j * GridStride +
                   block_id * ElemPerBlock + threadIdx.x;
      float2 val = logits[offset];
      f(val.x, offset * 2);
      f(val.y, offset * 2 + 1);
    }
  }
}

template <int NumBlocks, int Threads, int UnRollFactor, typename F>
__device__ __forceinline__ void run_vectorized_v4(const float4 *logits,
                                                  const int seq_len,
                                                  const int block_id, F f) {

  constexpr int ElemPerBlock = Threads;
  constexpr int GridStride = NumBlocks * ElemPerBlock;

  for (int t = 0; t < seq_len / (4 * GridStride * UnRollFactor); t++) {
#pragma unroll
    for (int j = 0; j < UnRollFactor; j++) {

      int offset = t * GridStride * UnRollFactor + j * GridStride +
                   block_id * ElemPerBlock + threadIdx.x;
      float4 val = logits[offset];
      f(val.x, offset * 4);
      f(val.y, offset * 4 + 1);
      f(val.z, offset * 4 + 2);
      f(val.w, offset * 4 + 3);
    }
  }
}
template <int NumBlocks, int Threads, int UnrollFactor, int VecType, typename F>
__device__ __forceinline__ void run_vectorized(const float *logits,
                                               const int seq_len,
                                               const int block_id, F f) {

  static_assert(VecType == 2 || VecType == 4, "expected VecType == 2 or 4");
  if (VecType == 2) {
    run_vectorized_v2<NumBlocks, Threads, UnrollFactor>((float2 *)logits,
                                                        seq_len, block_id, f);
  } else if (VecType == 4) {

    run_vectorized_v4<NumBlocks, Threads, UnrollFactor>((float4 *)logits,
                                                        seq_len, block_id, f);
  }

  constexpr int ElemPerBlock = VecType * Threads;
  constexpr int GridStride = NumBlocks * ElemPerBlock;
  int leftover_offset =
      (seq_len / (GridStride * UnrollFactor)) * GridStride * UnrollFactor;
  for (int i = leftover_offset + threadIdx.x + block_id * Threads; i < seq_len;
       i += Threads * NumBlocks) {
    f(logits[i], i);
  }
}

template <int NClusters, bool PDL_ENABLED>
__device__ __forceinline__ void fast_topk_cuda_v4(
    const float *__restrict__ logits,   // Input logits [max_num_pages * 64]
    const int *__restrict__ first_hist, // [256]
    int *__restrict__ output_indices,   // Output top-k indices [TopK]
    const int seq_len, const int num_cached) {
  constexpr int RADIX = 256;

  __shared__ int shared_hist[3][RADIX];

  // we may assume logits is aligned to 64 * 4 bytes, each warp handles a page
  //
  // First scan entire logits, processing bits [0-8), filling in histogram
  // from histogram, fill in cached indices, and compute the next histogram
  //
  // Then for bits [8-16), [16-24), [24-32), we have the histogram and cached
  // indices, compute next histogram and cached indices.

  // cache 2 * num_cached uint16_t (indices), 2 * num_cached float (logits),
  // TopK uint16_t (final indices)
  extern __shared__ uint8_t shared_cache[];
  alignas(128) __shared__ int
      shared_final_idx_count; // number of topk indices in s_topk_inds
  alignas(128)
      __shared__ int shared_num_cached_count[2]; // number of cached indices
  alignas(128) __shared__ int shared_threshold_bin;
  uint32_t *s_cached_logit_bits = (uint32_t *)shared_cache;
  int *s_cached_indices = (int *)(2 * num_cached + s_cached_logit_bits);
  int *s_topk_inds = (int *)(2 * num_cached + s_cached_indices);

  auto cluster = cg::this_cluster();
  const int block_id = cluster.block_rank();
  const bool radix_thread = threadIdx.x < RADIX;
  constexpr bool DEBUG = false;
  constexpr bool RUN_PHASE1 = true;
  constexpr bool ENABLE_CLUSTER_SAFETY = true;

  auto get_threshold_bin = [&](int hist_idx, int &k_remaining,
                               bool sum_hist = true) {
    __syncthreads();
    // first reduce cum sum locally
    int cum_val = cum_sum(shared_hist[hist_idx]);
    if (NClusters > 1 && sum_hist) {
      if (radix_thread) {
        shared_hist[hist_idx][threadIdx.x] = cum_val;
      }
      cluster.sync(); // now first block in cluster has its local cum sum

      if (radix_thread) {
#pragma unroll
        for (int cl = 0; cl < NClusters - 1; cl++) {
          cum_val +=
              cluster.map_shared_rank(&shared_hist[hist_idx][threadIdx.x],
                                      (cl + block_id + 1) % NClusters)[0];
        }
        shared_hist[2][threadIdx.x] = cum_val;
      }
    } else {
      if (radix_thread) {
        shared_hist[2][threadIdx.x] = cum_val;
      }
    }

    __syncthreads();

    int cum_val1 =
        threadIdx.x < RADIX - 1 ? shared_hist[2][threadIdx.x + 1] : 0;

    if (radix_thread && cum_val > k_remaining && cum_val1 <= k_remaining) {
      shared_threshold_bin = threadIdx.x;
      if (DEBUG) {
        printf(
            "block_id %d: threshold_bin %d. cum_sum_thres %d, cum_sum_thres+1 "
            "%d, topk_val %d\n",
            block_id, threadIdx.x, cum_val, cum_val1, shared_final_idx_count);
      }
    }

    __syncthreads();
    const int threshold_bin = shared_threshold_bin;
    if (threshold_bin < RADIX - 1) {
      k_remaining -= shared_hist[2][threshold_bin + 1];
    }
    return threshold_bin;
  };
  if (PDL_ENABLED)
    cudaGridDependencySynchronize();

  if (radix_thread) {
    shared_hist[0][threadIdx.x] = first_hist[threadIdx.x];
    shared_hist[1][threadIdx.x] = 0;
  }
  if (threadIdx.x == 0) {
    shared_final_idx_count = 0;
    shared_num_cached_count[0] = 0;
  }

  int top_k_remaining = TopK;
  if (RUN_PHASE1) {
    const int threshold_bin = get_threshold_bin(0, top_k_remaining, false);

    auto compute_phase1 = [&](float logit, int i) {
      uint32_t bits = convert_to_uint32_v2(logit);
      int bin = (bits >> 24);

      if (bin > threshold_bin) {
        int topk_offset = atomicAdd(&shared_final_idx_count, 1);
        if (topk_offset < TopK) {
          s_topk_inds[topk_offset] = i;
        }
      }

      if (bin == threshold_bin) {
        int cached_offset = atomicAdd(&shared_num_cached_count[0], 1);
        if (cached_offset < num_cached) {
          s_cached_indices[cached_offset] = i;
          s_cached_logit_bits[cached_offset] = bits;
          atomicAdd(shared_hist[1] + (bits >> 16 & 0xff), 1);
        }
      }
    };
    if (ENABLE_CLUSTER_SAFETY && NClusters > 1) {

      cluster.barrier_arrive(); // arrive at this point to signal that we are
                                // done with shared_hist[0], which we need in
                                // distributed shared memory to communicate
                                // histograms within the cluster
    }

    run_vectorized<NClusters, 1024, 4, 2>(logits, seq_len, block_id,
                                          compute_phase1);
  }
#pragma unroll
  for (int t = 1; t <= 3; t++) {
    const int phase = t % 2;
    // we are now using histogram buffers at phase, and cached_offsets at phase
    // ^ 1 clear the next stage to prepare

    if (ENABLE_CLUSTER_SAFETY && NClusters > 1) {
      cluster
          .barrier_wait(); // wait because we need to clear shared_hist for this
                           // next histogram; there is a dependency between
                           // get_threshold_bin and phase ^ 1 of the histogram
    }
    if (radix_thread) {
      shared_hist[phase ^ 1][threadIdx.x] = 0;
    }
    if (threadIdx.x == 0) {
      shared_num_cached_count[phase] = 0;
    }
    // get_threshold_bin synchronizes the block
    const int threshold_bin = get_threshold_bin(phase, top_k_remaining);
    if (ENABLE_CLUSTER_SAFETY && NClusters > 1 && t < 3) {
      cluster.barrier_arrive(); // same reasoning as above, for the last
                                // iteration there is no dependency because no
                                // more calls to get_threshold_bin
    }
    int buf_len = min(num_cached, shared_num_cached_count[phase ^ 1]);
    if (DEBUG && threadIdx.x == 0) {
      printf("block_id %d, num_cached %d, \n", block_id, buf_len);
    }
    // using cached indices, it's a local slice so don't partition between
    // blocks
    for (int i = threadIdx.x; i < buf_len; i += 1024) {
      uint32_t bits = s_cached_logit_bits[i + (phase ^ 1) * num_cached];
      int cached_idx = s_cached_indices[i + (phase ^ 1) * num_cached];
      int bin = (bits >> (24 - t * 8)) & 0xff;

      if (bin > threshold_bin) {
        int topk_offset = atomicAdd(&shared_final_idx_count, 1);
        if (topk_offset < TopK) {
          s_topk_inds[topk_offset] = cached_idx;
        } else {
          break;
        }
      }
      if (bin == threshold_bin && t < 3) {
        int cached_offset = atomicAdd(&shared_num_cached_count[phase], 1);
        if (cached_offset < num_cached) {
          s_cached_indices[cached_offset + phase * num_cached] = cached_idx;
          s_cached_logit_bits[cached_offset + phase * num_cached] = bits;
          atomicAdd(
              shared_hist[phase ^ 1] + (bits >> (24 - (t + 1) * 8) & 0xff), 1);
        }
      }
    }

    // it could be that at the last stage, we have say S topk indices, T indices
    // above the threshold_bin and S + T < TopK. Then the rest of the topk items
    // must come from threshold_bin
    if (t == 3) {
      if (top_k_remaining > 0) {
        for (int i = threadIdx.x; i < buf_len; i += blockDim.x) {
          int bin = s_cached_logit_bits[i];
          int cached_idx = s_cached_indices[i];
          if (bin == threshold_bin) {
            int topk_offset = atomicAdd(&shared_final_idx_count, 1);
            if (topk_offset < TopK) {
              s_topk_inds[topk_offset] = cached_idx;
            } else {
              break;
            }
          }
        }
      }
    }
  }
  __syncthreads();
  // now shared_final_idx_count contains the local topK in each
  // block in a cluster, and
  // s_topk_inds[0:shared_final_idx_count] contains the local
  // slice.
  if (NClusters > 1) {

    int topk_start = 0;
    int topk_num = shared_final_idx_count;

    cluster.sync(); // sync to get the shared_final_idx_count across CTAs in
                    // current cluster
    if (block_id > 0) {
      if (threadIdx.x == 0) {
        topk_start += atomicAdd(
            cluster.map_shared_rank(&shared_final_idx_count, 0), topk_num);
        shared_final_idx_count = topk_start;
      }
      __syncthreads();
      topk_start = shared_final_idx_count;
    }

    if (DEBUG && threadIdx.x == 0) {
      printf("block rank %d, topk start %d, topk num %d\n", block_id,
             topk_start, topk_num);
    }
    cluster.sync(); // we must wait at the end so that blocks in a cluster don't
                    // exit when we read their remote shared memory, removing
                    // this causes a subtle bug for large batch size
    for (int i = threadIdx.x; i < min(TopK, topk_num); i += 1024) {
      if (i + topk_start < TopK) {
        output_indices[i + topk_start] = s_topk_inds[i];
      }
    }

  } else {
    for (int i = threadIdx.x; i < TopK; i += 1024) {
      output_indices[i] = s_topk_inds[i];
    }
  }
}

template <int NClusters, bool PDL_ENABLED>
__global__ __launch_bounds__(1024) void __cluster_dims__(NClusters, 1, 1)
    fast_topk_v3_cluster_kernel_fused_prologue(
        const float *__restrict__ logits,   // [batchsize, logit_stride]
        const int *__restrict__ first_hist, // [batch_size, 256] (contiguous)
        int *__restrict__ output_indices, int *__restrict__ seq_lens,
        int logit_stride, int num_cached, int ind_batch_stride) {

  int logit_offset = blockIdx.x / NClusters * logit_stride;
  int ind_offset = blockIdx.x / NClusters * ind_batch_stride;
  int hist_offset = blockIdx.x / NClusters * 256;
  int seq_len = seq_lens[blockIdx.x / NClusters];
  if (seq_len <= TopK) {
    for (int i = threadIdx.x + (blockIdx.x % NClusters) * 1024; i < TopK;
         i += 1024 * NClusters) {
      if (i < seq_len) {
        output_indices[ind_offset + i] = i;
      } else {
        output_indices[ind_offset + i] = -1;
      }
    }

  } else {

    fast_topk_cuda_v4<NClusters, PDL_ENABLED>(
        logits + logit_offset, first_hist + hist_offset,
        output_indices + ind_offset, seq_len, num_cached);
  }
}

extern "C" void launch_fast_topk_clusters_fused_prologue(
    const float *logits, const int *first_hist, int *indices, int *seq_lens,
    int batch_size, int logit_stride, int num_cached, int ind_batch_stride,
    bool pdl_enabled, cudaStream_t stream) {

  int extern_shared_mem =
      (num_cached * 2 * sizeof(float) + num_cached * 2 * sizeof(int) +
       TopK *
           sizeof(int)); // 2 * num_cached float, 2 * num_cached int, topk int

  assert(extern_shared_mem <= 4096 * 16 + 2048 * 4 && "too much shared memory");
  setup_kernel_smem_once<fast_topk_v3_cluster_kernel_fused_prologue<8, true>,
                         4096 * 16 + 2048 * 4>();
  setup_kernel_smem_once<fast_topk_v3_cluster_kernel_fused_prologue<8, false>,
                         4096 * 16 + 2048 * 4>();

  assert(logit_stride % 4 == 0 && "logit_stride must be divisible by 4");
  assert(reinterpret_cast<uint64_t>(logits) % 4 == 0 &&
         "logits must be 16 byte aligned");

  cudaLaunchConfig_t config;
  config.numAttrs = 0;
  cudaLaunchAttribute attribute[1];
  if (pdl_enabled) {
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;
    config.numAttrs += 1;
  }

  config.blockDim = 1024;
  config.dynamicSmemBytes = extern_shared_mem;
  config.gridDim = batch_size * 8;
  config.stream = stream;
  config.attrs = attribute;

  if (pdl_enabled)
    cudaLaunchKernelEx(&config,
                       &fast_topk_v3_cluster_kernel_fused_prologue<8, true>,
                       logits, first_hist, indices, seq_lens, logit_stride,
                       num_cached, ind_batch_stride);
  else
    fast_topk_v3_cluster_kernel_fused_prologue<8, false>
        <<<batch_size * 8, 1024, extern_shared_mem, stream>>>(
            logits, first_hist, indices, seq_lens, logit_stride, num_cached,
            ind_batch_stride);
}
