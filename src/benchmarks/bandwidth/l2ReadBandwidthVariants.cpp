#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <algorithm>
#include <bit>       // std::bit_floor (C++20)
#include <cstdio>
#include <vector>

static constexpr int EXPLORE_ROUNDS = 5;
static constexpr int EXPLORE_PASSES = 128;

// ── Kernel 1: =v + s_waitcnt ──────────────────────────────────────────────────
__global__ void l2RBW_v_wait(uint32v4* __restrict__ dst,
                              uint32v4* __restrict__ src, size_t n)
{
    size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x  * blockDim.x;
    uint32v4 dummy = {0, 0, 0, 0};
    for (int rep = 0; rep < EXPLORE_PASSES; ++rep) {
        for (size_t i = tid; i < n; i += stride) {
            uint32v4 loaded;
#ifdef __HIP_PLATFORM_AMD__
            uint64_t __addr = reinterpret_cast<uint64_t>(src + i);
            asm volatile(
                "global_load_dwordx4 %0, %1, off " GLC "\n\t"
                "s_waitcnt vmcnt(0)\n\t"
                : "=v"(loaded) : "v"(__addr) : "memory");
#else
            loaded = src[i];
#endif
            dummy.x ^= loaded.x;
        }
    }
    dst[threadIdx.x] = dummy;
}

// ── Kernel 3: =s + s_waitcnt ──────────────────────────────────────────────────
__global__ void l2RBW_s_wait(uint32v4* __restrict__ dst,
                              uint32_t* __restrict__ src32, size_t n32)
{
    size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x  * blockDim.x;
    uint32_t dummy = 0;
    for (int rep = 0; rep < EXPLORE_PASSES; ++rep) {
        for (size_t i = tid; i < n32; i += stride) {
#ifdef __HIP_PLATFORM_AMD__
            uint32_t loaded_s, vgpr_tmp;
            uint64_t __addr = reinterpret_cast<uint64_t>(src32 + i);
            asm volatile(
                "global_load_dword %1, %2, off\n\t"
                "s_waitcnt vmcnt(0)\n\t"
                "v_readfirstlane_b32 %0, %1\n\t"
                : "=s"(loaded_s), "=&v"(vgpr_tmp)
                : "v"(__addr)
                : "memory");
            dummy ^= loaded_s;
#else
            dummy ^= src32[i];
#endif
        }
    }
    dst[threadIdx.x].x = dummy;
}

// ── Kernel 4: =s + no waitcnt ─────────────────────────────────────────────────
__global__ void l2RBW_s_nowait(uint32v4* __restrict__ dst,
                                uint32_t* __restrict__ src32, size_t n32)
{
    size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x  * blockDim.x;
    uint32_t dummy = 0;
    for (int rep = 0; rep < EXPLORE_PASSES; ++rep) {
        for (size_t i = tid; i < n32; i += stride) {
#ifdef __HIP_PLATFORM_AMD__
            uint32_t loaded_s, vgpr_tmp;
            uint64_t __addr = reinterpret_cast<uint64_t>(src32 + i);
            asm volatile(
                "global_load_dword %1, %2, off\n\t"
                "v_readfirstlane_b32 %0, %1\n\t"
                : "=s"(loaded_s), "=&v"(vgpr_tmp)
                : "v"(__addr)
                : "memory");
            dummy ^= loaded_s;
#else
            dummy ^= src32[i];
#endif
        }
    }
    dst[threadIdx.x].x = dummy;
}

// ── Kernel 5: plain C++/HIP, no inline asm ───────────────────────────────────
__global__ void l2RBW_cpp(uint32v4* __restrict__ dst,
                           uint32v4* __restrict__ src, size_t n)
{
    size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x  * blockDim.x;
    uint32v4 dummy = {0, 0, 0, 0};
    for (int rep = 0; rep < EXPLORE_PASSES; ++rep) {
        for (size_t i = tid; i < n; i += stride) {
            uint32v4 loaded = src[i];
            dummy.x ^= loaded.x;
        }
    }
    dst[threadIdx.x] = dummy;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Batched-load kernels
//  ─────────────────────────────────────────────────────────────────────────────
//  Array layout: n_mask = N_PO2 - 1.  Thread 'tid' loads from BATCH addresses
//
//      addr_b = src + ( (tid + b * step) & n_mask )   b = 0 .. BATCH-1
//
//  step = N_PO2 / BATCH  →  the BATCH addresses are evenly spread across the
//  16 MiB power-of-2 array, hitting different cache sets.
//
//  All BATCH global_load_dwordx4 instructions are issued WITHOUT any
//  intermediate s_waitcnt, so the hardware keeps BATCH requests in-flight per
//  wavefront simultaneously.  One s_waitcnt vmcnt(0) waits for all of them.
//
//  "Can two blocks address the same element?"  YES — with 7 296 × 256 = 1 867 776
//  threads but only N_PO2 = 1 048 576 elements, ~78 % of elements are accessed
//  by 2 threads after masking.  This is intentional: each CU issues its own
//  independent L2 request even for the same cache line (the GPU memory system
//  does NOT coalesce requests across different wavefronts or CUs).  Reads never
//  conflict, so correctness is guaranteed.
//
//  Addresses are pre-computed OUTSIDE the rep-loop.  The hardware loads the same
//  lines every rep — guaranteed L2 hits after warmup — maximising L2 bandwidth
//  stress without needing an array larger than the L2.
// ─────────────────────────────────────────────────────────────────────────────

// ── 1 load in-flight ──────────────────────────────────────────────────────────
__global__ void l2RBW_1ld(uint32v4* dst, const uint32v4* src, size_t n_mask)
{
    const size_t tid  = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t base = tid & n_mask;
    uint32v4 dummy = {0, 0, 0, 0};
#ifdef __HIP_PLATFORM_AMD__
    const uint64_t a0 = reinterpret_cast<uint64_t>(src + base);
#endif
    for (int rep = 0; rep < EXPLORE_PASSES; ++rep) {
        uint32v4 v0;
#ifdef __HIP_PLATFORM_AMD__
        asm volatile(
            "global_load_dwordx4 %0, %1, off " GLC "\n\t"
            "s_waitcnt vmcnt(0)\n\t"
            : "=v"(v0) : "v"(a0) : "memory");
#else
        v0 = src[base];
#endif
        dummy.x ^= v0.x;
    }
    dst[threadIdx.x] = dummy;
}

// ── 2 loads in-flight ─────────────────────────────────────────────────────────
__global__ void l2RBW_2ld(uint32v4* dst, const uint32v4* src, size_t n_mask)
{
    const size_t tid  = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t step = (n_mask + 1) / 2;
    const size_t b0   =  tid          & n_mask;
    const size_t b1   = (tid +   step) & n_mask;
    uint32v4 dummy = {0, 0, 0, 0};
#ifdef __HIP_PLATFORM_AMD__
    const uint64_t a0 = reinterpret_cast<uint64_t>(src + b0);
    const uint64_t a1 = reinterpret_cast<uint64_t>(src + b1);
#endif
    for (int rep = 0; rep < EXPLORE_PASSES; ++rep) {
        uint32v4 v0, v1;
#ifdef __HIP_PLATFORM_AMD__
        asm volatile(
            "global_load_dwordx4 %0, %2, off " GLC "\n\t"
            "global_load_dwordx4 %1, %3, off " GLC "\n\t"
            "s_waitcnt vmcnt(0)\n\t"
            : "=v"(v0), "=v"(v1)
            : "v"(a0), "v"(a1)
            : "memory");
#else
        v0 = src[b0]; v1 = src[b1];
#endif
        dummy.x ^= v0.x ^ v1.x;
    }
    dst[threadIdx.x] = dummy;
}

// ── 4 loads in-flight ─────────────────────────────────────────────────────────
__global__ void l2RBW_4ld(uint32v4* dst, const uint32v4* src, size_t n_mask)
{
    const size_t tid  = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t step = (n_mask + 1) / 4;
    const size_t b0   =  tid            & n_mask;
    const size_t b1   = (tid +   step)  & n_mask;
    const size_t b2   = (tid + 2*step)  & n_mask;
    const size_t b3   = (tid + 3*step)  & n_mask;
    uint32v4 dummy = {0, 0, 0, 0};
#ifdef __HIP_PLATFORM_AMD__
    const uint64_t a0 = reinterpret_cast<uint64_t>(src + b0);
    const uint64_t a1 = reinterpret_cast<uint64_t>(src + b1);
    const uint64_t a2 = reinterpret_cast<uint64_t>(src + b2);
    const uint64_t a3 = reinterpret_cast<uint64_t>(src + b3);
#endif
    for (int rep = 0; rep < EXPLORE_PASSES; ++rep) {
        uint32v4 v0, v1, v2, v3;
#ifdef __HIP_PLATFORM_AMD__
        asm volatile(
            "global_load_dwordx4 %0, %4, off " GLC "\n\t"
            "global_load_dwordx4 %1, %5, off " GLC "\n\t"
            "global_load_dwordx4 %2, %6, off " GLC "\n\t"
            "global_load_dwordx4 %3, %7, off " GLC "\n\t"
            "s_waitcnt vmcnt(0)\n\t"
            : "=v"(v0), "=v"(v1), "=v"(v2), "=v"(v3)
            : "v"(a0), "v"(a1), "v"(a2), "v"(a3)
            : "memory");
#else
        v0 = src[b0]; v1 = src[b1]; v2 = src[b2]; v3 = src[b3];
#endif
        dummy.x ^= v0.x ^ v1.x ^ v2.x ^ v3.x;
    }
    dst[threadIdx.x] = dummy;
}

// ── 8 loads in-flight ─────────────────────────────────────────────────────────
__global__ void l2RBW_8ld(uint32v4* dst, const uint32v4* src, size_t n_mask)
{
    const size_t tid  = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t step = (n_mask + 1) / 8;
    const size_t b0   =  tid            & n_mask;
    const size_t b1   = (tid +   step)  & n_mask;
    const size_t b2   = (tid + 2*step)  & n_mask;
    const size_t b3   = (tid + 3*step)  & n_mask;
    const size_t b4   = (tid + 4*step)  & n_mask;
    const size_t b5   = (tid + 5*step)  & n_mask;
    const size_t b6   = (tid + 6*step)  & n_mask;
    const size_t b7   = (tid + 7*step)  & n_mask;
    uint32v4 dummy = {0, 0, 0, 0};
#ifdef __HIP_PLATFORM_AMD__
    const uint64_t a0 = reinterpret_cast<uint64_t>(src + b0);
    const uint64_t a1 = reinterpret_cast<uint64_t>(src + b1);
    const uint64_t a2 = reinterpret_cast<uint64_t>(src + b2);
    const uint64_t a3 = reinterpret_cast<uint64_t>(src + b3);
    const uint64_t a4 = reinterpret_cast<uint64_t>(src + b4);
    const uint64_t a5 = reinterpret_cast<uint64_t>(src + b5);
    const uint64_t a6 = reinterpret_cast<uint64_t>(src + b6);
    const uint64_t a7 = reinterpret_cast<uint64_t>(src + b7);
#endif
    for (int rep = 0; rep < EXPLORE_PASSES; ++rep) {
        uint32v4 v0, v1, v2, v3, v4, v5, v6, v7;
#ifdef __HIP_PLATFORM_AMD__
        asm volatile(
            "global_load_dwordx4 %0, %8, off " GLC "\n\t"
            "global_load_dwordx4 %1, %9, off " GLC "\n\t"
            "global_load_dwordx4 %2, %10, off " GLC "\n\t"
            "global_load_dwordx4 %3, %11, off " GLC "\n\t"
            "global_load_dwordx4 %4, %12, off " GLC "\n\t"
            "global_load_dwordx4 %5, %13, off " GLC "\n\t"
            "global_load_dwordx4 %6, %14, off " GLC "\n\t"
            "global_load_dwordx4 %7, %15, off " GLC "\n\t"
            "s_waitcnt vmcnt(0)\n\t"
            : "=v"(v0),"=v"(v1),"=v"(v2),"=v"(v3),
              "=v"(v4),"=v"(v5),"=v"(v6),"=v"(v7)
            : "v"(a0),"v"(a1),"v"(a2),"v"(a3),
              "v"(a4),"v"(a5),"v"(a6),"v"(a7)
            : "memory");
#else
        v0=src[b0]; v1=src[b1]; v2=src[b2]; v3=src[b3];
        v4=src[b4]; v5=src[b5]; v6=src[b6]; v7=src[b7];
#endif
        dummy.x ^= v0.x^v1.x^v2.x^v3.x^v4.x^v5.x^v6.x^v7.x;
    }
    dst[threadIdx.x] = dummy;
}

// ── 16 loads in-flight ────────────────────────────────────────────────────────
// Split into two asm blocks of 8 with NO wait between them so all 16 loads are
// in-flight simultaneously when the final s_waitcnt vmcnt(0) fires.
__global__ void l2RBW_16ld(uint32v4* dst, const uint32v4* src, size_t n_mask)
{
    const size_t tid  = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t step = (n_mask + 1) / 16;
    const size_t b0   =  tid              & n_mask;
    const size_t b1   = (tid +    step)   & n_mask;
    const size_t b2   = (tid +  2*step)   & n_mask;
    const size_t b3   = (tid +  3*step)   & n_mask;
    const size_t b4   = (tid +  4*step)   & n_mask;
    const size_t b5   = (tid +  5*step)   & n_mask;
    const size_t b6   = (tid +  6*step)   & n_mask;
    const size_t b7   = (tid +  7*step)   & n_mask;
    const size_t b8   = (tid +  8*step)   & n_mask;
    const size_t b9   = (tid +  9*step)   & n_mask;
    const size_t b10  = (tid + 10*step)   & n_mask;
    const size_t b11  = (tid + 11*step)   & n_mask;
    const size_t b12  = (tid + 12*step)   & n_mask;
    const size_t b13  = (tid + 13*step)   & n_mask;
    const size_t b14  = (tid + 14*step)   & n_mask;
    const size_t b15  = (tid + 15*step)   & n_mask;
    uint32v4 dummy = {0, 0, 0, 0};
#ifdef __HIP_PLATFORM_AMD__
    const uint64_t a0  = reinterpret_cast<uint64_t>(src + b0);
    const uint64_t a1  = reinterpret_cast<uint64_t>(src + b1);
    const uint64_t a2  = reinterpret_cast<uint64_t>(src + b2);
    const uint64_t a3  = reinterpret_cast<uint64_t>(src + b3);
    const uint64_t a4  = reinterpret_cast<uint64_t>(src + b4);
    const uint64_t a5  = reinterpret_cast<uint64_t>(src + b5);
    const uint64_t a6  = reinterpret_cast<uint64_t>(src + b6);
    const uint64_t a7  = reinterpret_cast<uint64_t>(src + b7);
    const uint64_t a8  = reinterpret_cast<uint64_t>(src + b8);
    const uint64_t a9  = reinterpret_cast<uint64_t>(src + b9);
    const uint64_t a10 = reinterpret_cast<uint64_t>(src + b10);
    const uint64_t a11 = reinterpret_cast<uint64_t>(src + b11);
    const uint64_t a12 = reinterpret_cast<uint64_t>(src + b12);
    const uint64_t a13 = reinterpret_cast<uint64_t>(src + b13);
    const uint64_t a14 = reinterpret_cast<uint64_t>(src + b14);
    const uint64_t a15 = reinterpret_cast<uint64_t>(src + b15);
#endif
    for (int rep = 0; rep < EXPLORE_PASSES; ++rep) {
        uint32v4 v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15;
#ifdef __HIP_PLATFORM_AMD__
        // First 8 loads — no s_waitcnt, so all 16 are in-flight after second block.
        asm volatile(
            "global_load_dwordx4 %0, %8, off " GLC "\n\t"
            "global_load_dwordx4 %1, %9, off " GLC "\n\t"
            "global_load_dwordx4 %2, %10, off " GLC "\n\t"
            "global_load_dwordx4 %3, %11, off " GLC "\n\t"
            "global_load_dwordx4 %4, %12, off " GLC "\n\t"
            "global_load_dwordx4 %5, %13, off " GLC "\n\t"
            "global_load_dwordx4 %6, %14, off " GLC "\n\t"
            "global_load_dwordx4 %7, %15, off " GLC "\n\t"
            : "=v"(v0),"=v"(v1),"=v"(v2),"=v"(v3),
              "=v"(v4),"=v"(v5),"=v"(v6),"=v"(v7)
            : "v"(a0),"v"(a1),"v"(a2),"v"(a3),
              "v"(a4),"v"(a5),"v"(a6),"v"(a7)
            : "memory");
        // Second 8 loads + wait for all 16.
        asm volatile(
            "global_load_dwordx4 %0, %8, off " GLC "\n\t"
            "global_load_dwordx4 %1, %9, off " GLC "\n\t"
            "global_load_dwordx4 %2, %10, off " GLC "\n\t"
            "global_load_dwordx4 %3, %11, off " GLC "\n\t"
            "global_load_dwordx4 %4, %12, off " GLC "\n\t"
            "global_load_dwordx4 %5, %13, off " GLC "\n\t"
            "global_load_dwordx4 %6, %14, off " GLC "\n\t"
            "global_load_dwordx4 %7, %15, off " GLC "\n\t"
            "s_waitcnt vmcnt(0)\n\t"
            : "=v"(v8),"=v"(v9),"=v"(v10),"=v"(v11),
              "=v"(v12),"=v"(v13),"=v"(v14),"=v"(v15)
            : "v"(a8),"v"(a9),"v"(a10),"v"(a11),
              "v"(a12),"v"(a13),"v"(a14),"v"(a15)
            : "memory");
#else
        v0=src[b0];  v1=src[b1];  v2=src[b2];  v3=src[b3];
        v4=src[b4];  v5=src[b5];  v6=src[b6];  v7=src[b7];
        v8=src[b8];  v9=src[b9];  v10=src[b10]; v11=src[b11];
        v12=src[b12]; v13=src[b13]; v14=src[b14]; v15=src[b15];
#endif
        dummy.x ^= v0.x^v1.x^v2.x^v3.x^v4.x^v5.x^v6.x^v7.x
                ^v8.x^v9.x^v10.x^v11.x^v12.x^v13.x^v14.x^v15.x;
    }
    dst[threadIdx.x] = dummy;
}

// ─────────────────────────────────────────────────────────────────────────────

namespace benchmark {

// Shared try_run / run_with helpers — declared as static lambdas are not
// re-usable across functions, so factor them out as free helpers here.

template<typename F>
static double bw_try_run(F& launch_fn) {
    auto start = util::createHipEvent();
    auto end   = util::createHipEvent();
    util::hipCheck(hipEventRecord(start));
    launch_fn();
    util::hipCheck(hipEventRecord(end));
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        // On ROCm 6.x hipDeviceReset() itself fails after a GPU fault.
        (void)hipGetLastError();
        return -1.0;
    }
    return util::getElapsedTimeMs(start, end);
}

template<typename F>
static void bw_run_with(const char* label, double tGiB, F& launch_fn) {
    if (bw_try_run(launch_fn) < 0.0) {
        printf("  %-55s  CRASH (warmup)\n", label);
        return;
    }
    std::vector<double> bws;
    for (int r = 0; r < EXPLORE_ROUNDS; ++r) {
        double ms = bw_try_run(launch_fn);
        if (ms < 0.0) {
            printf("  %-55s  CRASH (round %d/%d)\n", label, r+1, EXPLORE_ROUNDS);
            return;
        }
        bws.push_back(tGiB / (ms / 1000.0));
    }
    std::sort(bws.begin(), bws.end());
    printf("  %-55s  %9.1f GiB/s\n", label, bws[EXPLORE_ROUNDS / 2]);
}

// ─────────────────────────────────────────────────────────────────────────────
//  exploreBatchedBandwidth
//  ─────────────────────────────────────────────────────────────────────────────
//  General batched-load stress that works for any memory level.
//
//  arraySizeBytes must fit in the TARGET memory level so that data stays warm:
//    L2  →  l2CacheSize * numXCDs * 0.9
//    L3  →  l3CacheSize * 0.9
//    HBM →  a large array that exceeds all caches (pair with GLC/SLC bypass)
//
//  arraySizeBytes is rounded DOWN to the nearest power of 2 for fast masking.
//  All available blocks are launched regardless of array size: with
//  (blocks × threads) > n_po2 / BATCH, multiple threads address the same
//  element after masking.  Each CU issues its own independent L2/L3 request —
//  the GPU memory system does NOT coalesce across different wavefronts or CUs —
//  so overlapping is safe and maximises concurrent outstanding requests.
//
//  Optimal BATCH varies by GPU (VGPR file size, wavefront size, CU count).
//  All five are tried; the plateau or peak in the output indicates the best.
//  On gfx942 (512 VGPRs/SIMD, wf=64):
//    BATCH  VGPRs/wf (est.)  wf/SIMD (est.)
//     1        ~15              ~32  (full occupancy)
//     2        ~25              ~20
//     4        ~40              ~12
//     8        ~70              ~7
//    16       ~130              ~4   (occupancy drop may hurt)
// ─────────────────────────────────────────────────────────────────────────────
void exploreBatchedBandwidth(const char* level, size_t arraySizeBytes)
{
    const uint32_t maxTPB = util::min(util::getMaxThreadsPerBlock(),
                                      util::getWarpSize() * util::getSIMDsPerCU());
    const uint32_t totalMaxBlocks = util::getNumberOfComputeUnits() *
                                    util::getDeviceProperties().maxBlocksPerMultiProcessor;

    // Round array down to a power-of-2 number of uint32v4 elements.
    const size_t n_elems = arraySizeBytes / sizeof(uint32v4);
    const size_t n_po2   = (n_elems > 0) ? std::bit_floor(n_elems) : size_t{1};
    const size_t n_mask  = n_po2 - 1;

    // batchGiB: actual bytes loaded per run = threads × BATCH × 16B × passes.
    // This counts every independent L2/memory request, including overlapping ones.
    auto batchGiB = [&](int batch) -> double {
        return (double)totalMaxBlocks * maxTPB * batch
               * (double)sizeof(uint32v4) * EXPLORE_PASSES / (1.0 * GiB);
    };

    uint32v4* d_src = util::allocateGPUMemory<uint32v4>(n_po2);
    uint32v4* d_dst = util::allocateGPUMemory<uint32v4>(maxTPB);

    printf("\n=== %s Read BW: batched loads"
           "  n_po2=2^%zu (%.0f MiB)  blocks=%u  threads=%u  passes=%d ===\n",
           level,
           static_cast<size_t>(__builtin_ctzll(static_cast<unsigned long long>(n_po2))),
           static_cast<double>(n_po2) * sizeof(uint32v4) / (1024.0 * 1024.0),
           totalMaxBlocks, maxTPB, EXPLORE_PASSES);
    printf("    overlap ratio: %.1fx (threads/elements)\n",
           static_cast<double>(totalMaxBlocks) * maxTPB / static_cast<double>(n_po2));

    auto go = [&](const char* lbl, int batch, auto kern) {
        auto fn = [&]{ kern<<<totalMaxBlocks, maxTPB>>>(d_dst, d_src, n_mask); };
        bw_run_with(lbl, batchGiB(batch), fn);
    };

    go("[ 1LD]  1 dwordx4 in-flight / wavefront step",  1, l2RBW_1ld);
    go("[ 2LD]  2 dwordx4 in-flight / wavefront step",  2, l2RBW_2ld);
    go("[ 4LD]  4 dwordx4 in-flight / wavefront step",  4, l2RBW_4ld);
    go("[ 8LD]  8 dwordx4 in-flight / wavefront step",  8, l2RBW_8ld);
    go("[16LD] 16 dwordx4 in-flight / wavefront step", 16, l2RBW_16ld);

    util::hipCheck(hipFree(d_src));
    util::hipCheck(hipFree(d_dst));
}

// ─────────────────────────────────────────────────────────────────────────────
//  exploreL2ReadBandwidth
// ─────────────────────────────────────────────────────────────────────────────
void exploreL2ReadBandwidth(size_t l2SizeBytes) {
    util::hipDeviceReset();

    auto xcdOpt = util::getNumXCDs();
    if (!xcdOpt) xcdOpt = util::getNumXCCs();
    const size_t numXCDs = static_cast<size_t>(xcdOpt.value_or(1));

    const size_t arraySizeBytes = static_cast<size_t>(l2SizeBytes * numXCDs * 0.9);
    const double testSizeGiB    = static_cast<double>(arraySizeBytes) / (1.0 * GiB);
    const double totalGiB       = testSizeGiB * EXPLORE_PASSES;

    const uint32_t maxTPB = util::min(util::getMaxThreadsPerBlock(),
                                       util::getWarpSize() * util::getSIMDsPerCU());
    const uint32_t totalMaxBlocks = util::getNumberOfComputeUnits() *
                                     util::getDeviceProperties().maxBlocksPerMultiProcessor;

    const size_t n   = arraySizeBytes / sizeof(uint32v4);
    const size_t n32 = arraySizeBytes / sizeof(uint32_t);
    // Cap blocks so stride ≤ n (avoids modulo bottleneck in inner loop).
    const uint32_t cappedBlocks = std::min(totalMaxBlocks,
                                           static_cast<uint32_t>(n / maxTPB));

    uint32v4* d_src   = util::allocateGPUMemory<uint32v4>(n);
    uint32v4* d_dst   = util::allocateGPUMemory<uint32v4>(maxTPB);
    uint32_t* d_src32 = reinterpret_cast<uint32_t*>(d_src);

    // ── Section 1: single-load variants (stride-based, capped blocks) ────────
    printf("\n=== L2 Read BW: single-load variants"
           "  array=%.1f MiB  blocks=%u/%u  threads=%u  XCDs=%zu  passes=%d ===\n",
           static_cast<double>(arraySizeBytes) / (1024.0 * 1024.0),
           cappedBlocks, totalMaxBlocks, maxTPB, numXCDs, EXPLORE_PASSES);

    auto run = [&](const char* label, auto launch_fn) {
        bw_run_with(label, totalGiB, launch_fn);
    };

    run("[1] global_load_dwordx4  =v  s_waitcnt",
        [&]{ l2RBW_v_wait  <<<cappedBlocks, maxTPB>>>(d_dst, d_src,   n);   });
    run("[3] global_load_dword    =s  s_waitcnt   (4 B/iter)",
        [&]{ l2RBW_s_wait  <<<cappedBlocks, maxTPB>>>(d_dst, d_src32, n32); });
    run("[4] global_load_dword    =s  no waitcnt  (4 B/iter)",
        [&]{ l2RBW_s_nowait<<<cappedBlocks, maxTPB>>>(d_dst, d_src32, n32); });
    run("[5] C++/HIP  no asm",
        [&]{ l2RBW_cpp    <<<cappedBlocks, maxTPB>>>(d_dst, d_src,   n);    });

    // Kernel 2: confirmed VGPR WAW hardware fault on gfx942.
    // Not run: on ROCm 6.x the fault leaves the HIP driver unrecoverable.
    printf("  %-55s  SKIP  (VGPR WAW fault — confirmed crash in prior run)\n",
           "[2] global_load_dwordx4  =v  no waitcnt");

    util::hipCheck(hipFree(d_src));
    util::hipCheck(hipFree(d_dst));

    // ── Section 2: batched loads (power-of-2 array, all blocks) ─────────────
    exploreBatchedBandwidth("L2", arraySizeBytes);
}

} // namespace benchmark
