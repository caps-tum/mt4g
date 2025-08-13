#pragma once

#include <stdexcept>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

#ifdef __HIP_PLATFORM_AMD__
  /*
   * On CDNA‑3 GPUs (gfx940, gfx941, gfx942), the older “glc” flag no longer
   * reflects the new encoded memory model: LLVM/ROCm now encodes:
   *   - only “sc1” bit if glc was requested
   *   - both “sc0 sc1” if glc + slc was requested
   * Older CDNA‑2 (e.g. gfx90a) and earlier still expect the legacy syntax:
   *   - “glc” → sc0+sc1 semantics
   *   - “glc slc” → explicit sc0+sc1
   *
   * The official AMDGPUUsage memory-model tables for GFX940, GFX941 and GFX942
   * confirm this encoding shift:
   *   - GFX940/941: flat_load/flat_store use sc0=1 sc1=1 for non‑atomic (“glc slc”)
   *   - GFX942: the flag differences now produce only sc1, or explicit “sc0 sc1”
   *     when both cache bypass flags were used (see Memory Model section,
   *     table for GFX940–942)
   *
   * Thus we branch using __gfx94[0‑2] to select "sc1" vs "glc" so our inline ASM
   * always matches the AMD backend's expected modifier syntax.
   */
    #if defined(__gfx942__) || defined(__gfx941__) || defined(__gfx940__)
        #define GLC     "sc1"
        #define GLC_SLC "sc0 sc1"
    #else
        #define GLC     "glc"
        #define GLC_SLC "glc slc"
    #endif
#endif

/**
 * @brief Query the physical compute unit (SM) identifier of the calling thread.
 *
 * @return Hardware compute unit ID.
 */
__device__ __forceinline__ uint32_t __getPhysicalCUId() {
    #ifdef __HIP_PLATFORM_NVIDIA__
    uint32_t smid;
    asm volatile ("mov.u32 %0, %smid;" : "=r"(smid));
    return smid;
    #endif

    #ifdef __HIP_PLATFORM_AMD__
return __smid();
#endif
}

/**
 * @brief Query the warp identifier of the calling thread.
 *
 * @return Warp ID Register content.
 */
__device__ __forceinline__ uint32_t __getWarpId() {
    #ifdef __HIP_PLATFORM_NVIDIA__
    uint32_t wid;
    asm volatile ("mov.u32 %0, %warpid;" : "=r"(wid));
    return wid;
    #endif
    #ifdef __HIP_PLATFORM_AMD__
    // Read HW_ID into an SGPR, then extract fields.
    uint32_t hwid;
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s"(hwid));

    // wave_id = bits [3:0]  (scheduler's wave slot within a SIMD)
    // simd_id = bits [5:4]  (which SIMD within the CU)
    const uint32_t wave_id =  hwid        & 0xF;   // [0..15]
    //const uint32_t simd_id = (hwid >> 4)  & 0x3;   // [0..3]

    return wave_id;  
    #endif

    // Portable fallback
    const uint32_t linear_tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    return linear_tid / warpSize;
}

// While a bit sketchy it is sufficient for our purposes
// Proper way of doing this would be using cooperative groups.
__device__ uint32_t gBarrierCount = 0; 
__device__ uint32_t gBarrierSense = 0; 

/**
 * @brief Lightweight global barrier across all thread blocks.
 *
 * Uses atomics to synchronize all participating blocks without
 * relying on cooperative groups.
 *
 * @param threshold Number of threads expected to arrive at the barrier.
 */
__device__ __forceinline__ void __globalBarrier(unsigned int threshold) {
    // Shared per-block: store the sense for this barrier phase
    __shared__ unsigned int localSense;

    // Only one thread per block initializes localSense
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        // Flip sense: threads will wait for this new value
        localSense = 1 - atomicAdd(&gBarrierSense, 0);
    }
    // Ensure localSense is visible to all threads in this block
    __syncthreads();

    // Arrival: get a unique ticket in [0..threshold-1]
    unsigned int ticket = atomicAdd(&gBarrierCount, 1);

    if (ticket == threshold - 1) {
        // Last thread to arrive:
        // Reset counter for next barrier and release all waiters
        gBarrierCount = 0;  // reset for next use
        atomicExch(&gBarrierSense, localSense);  
    } else {
        // Busy-wait until sense matches localSense
        while (atomicAdd(&gBarrierSense, 0) != localSense) {
            // spin
        }
    }

    // Optional: synchronize threads within a block
    __syncthreads();
}

/**
 * @brief Block-local barrier without using __syncthreads().
 *
 * @param threshold Number of threads in the block.
 */
__device__ __forceinline__ void __localBarrier(unsigned int threshold) {
    // Shared per-block counter and sense flag
    __shared__ unsigned int counter;      // counter lives in shared memory
    __shared__ unsigned int sense;        // sense flag lives in shared memory

    // Compute this thread's sense for the current barrier phase
    unsigned int mySense = 1 - sense;     // sense-reversal technique

    // Atomic arrival: each thread fetches-and-increments the counter
    unsigned int ticket = atomicAdd(&counter, 1); // shared-memory atomic 

    if (ticket == threshold - 1) {
        // Last arriving thread: reset counter and flip the sense to release waiters
        counter = 0;                       // reset for next barrier iteration
        sense = mySense;                   // release all spin-waiters
    } else {
        // Other threads: spin-wait until sense matches this thread’s sense
        while (sense != mySense) {         // pure busy-wait, no syncthreads 
            // spin
        }
    }
    // At this point, all 'threshold' threads have synchronized
}


/**
 * @brief Read the device's global timer register.
 *
 * @return 64-bit global timer value.
 */
__device__ __forceinline__ uint64_t __globaltimer() {
    uint64_t globaltimer;
    #ifdef __HIP_PLATFORM_NVIDIA__
    asm volatile ("mov.u64 %0, %globaltimer;\n\t"   : "=l"(globaltimer));
    #endif

    #ifdef __HIP_PLATFORM_AMD__
    __asm__ volatile (
        "s_memrealtime %0\n\t" 
        "s_waitcnt lgkmcnt(0)\n\t" 
        : "=s" (globaltimer) );
    #endif

    return globaltimer;
}

/**
 * @brief Read the current cycle counter on the device.
 *
 * @return Timer value in cycles.
 */
__device__ __forceinline__ uint64_t __timer() {
    #ifdef __HIP_PLATFORM_NVIDIA__
    uint32_t clock;
    asm volatile ("mov.u32 %0, %%clock;\n\t" : "=r"(clock));
    return clock;
    #endif

    #ifdef __HIP_PLATFORM_AMD__
    uint64_t clock;
    __asm__ volatile (
        "s_memtime %0\n\t" 
        "s_waitcnt lgkmcnt(0)\n\t" 
        : "=s" (clock) );
    return clock;
    #endif
}

/**
 * @brief Perform a load that bypasses all caches.
 *
 * Useful for measuring memory latency without cache effects.
 *
 * @param baseAddress Base address to load from.
 * @param index       Index within the array to read.
 * @return Loaded value.
 */
__device__ __forceinline__ uint32_t __forceBypassAllCacheReads(uint32_t *baseAddress, uint32_t index) {
    uint32_t result;
    // Calculate 64-Bit-Byte-Offset
    uint64_t addr = reinterpret_cast<uint64_t>(baseAddress) + uint64_t(index) * sizeof(uint32_t);

    #ifdef __HIP_PLATFORM_AMD__
    __asm__ volatile(
        // Flat-Load with GLC=1 and SLC=1: Bypasses L1 and L2
        "flat_load_dword %0, %1 " GLC_SLC 
        #if defined(__gfx942__) || defined(__gfx941__) || defined(__gfx940__)
        " nt" // Only on CDNA3(+)
        #endif
         " \n\t"
        // Wait for VMEM
        "s_waitcnt vmcnt(0)\n\t"
        : "=v"(result) 
        : "v"(addr) 
    );
    #endif
    #ifdef __HIP_PLATFORM_NVIDIA__
    asm volatile(
        // ld.global.cv.u32: .cv invalidates L2 and forces L2 pull. Sadly, there is no way of always pulling from VMEM directly.
        "ld.global.cv.u32 %0, [%1];\n\t"
        "membar.gl;" // Ensure all global memory operations are complete
        : "=r"(result)  
        : "l"(addr)  
        : "memory"
    );
    #endif

    return result;
}

/**
 * @brief Perform a load that guarantees an L1 cache miss.
 *
 * @param baseAddress Base address to load from.
 * @param index       Index within the array to read.
 * @return Loaded value fetched from L2/VRAM.
 */
__device__ __forceinline__ uint32_t __forceL1MissRead(uint32_t *baseAddress, uint32_t index) {
    uint32_t result;
    // Calc 64 Bit addr
    uint64_t addr = reinterpret_cast<uint64_t>(baseAddress) + uint64_t(index) * sizeof(uint32_t);

    #ifdef __HIP_PLATFORM_AMD__
    __asm__ volatile(
        // Flat loading with GLC=1 forces L1 miss, therefore reading from L2
        "flat_load_dword %0, %1 " GLC "\n\t"
        // Wait untill VMEM is finished
        "s_waitcnt vmcnt(0)\n\t"
        : "=v"(result)              // Output in VGPR
        : "v"(addr)                 // Address in VGPR
        : "memory"
    );

    #endif

    #ifdef __HIP_PLATFORM_NVIDIA__
    asm volatile(
        "ld.global.cg.u32 %0, [%1];\n\t"  
        : "=r"(result)
        : "l"(addr)
        : "memory"
    );
    #endif

    return result;
}


/**
 * @brief Perform a normal load that may hit in L1.
 *
 * @param baseAddress Base address to load from.
 * @param index       Index within the array to read.
 * @return Loaded value.
 */
__device__ __forceinline__ uint32_t __allowL1Read(uint32_t *baseAddress, uint32_t index) {
    uint32_t result;
    uint64_t addr = reinterpret_cast<uint64_t>(baseAddress) + uint64_t(index) * sizeof(uint32_t);

    #ifdef __HIP_PLATFORM_AMD__
    __asm__ volatile(
        // Flat loading with GLC=0, allows reading from L1 (if the value is present there, of course)
        "flat_load_dword %0, %1\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : "=v"(result)
        : "v"(addr)
        : "memory"
    );
    #endif

    #ifdef __HIP_PLATFORM_NVIDIA__
    asm volatile(
        "ld.global.ca.u32 %0, [%1];\n\t" 
        : "=r"(result)
        : "l"(addr)
        : "memory"
    );
    #endif

    return result;
}

/**
 * @brief Load a value while bypassing L1 and L2 caches.
 *
 * On AMD the instruction reads with " GLC_SLC " flags, on NVIDIA a cache-invalidating
 * load is performed.
 *
 * @param baseAddress Base address to load from.
 * @param index       Index within the array to read.
 * @return Loaded value.
 */
__device__ __forceinline__ uint32_t __l3Read(uint32_t *baseAddress, uint32_t index) {
    uint32_t result;
    uint64_t addr = reinterpret_cast<uint64_t>(baseAddress) + uint64_t(index) * sizeof(uint32_t);

    #ifdef __HIP_PLATFORM_AMD__
    __asm__ volatile(
        "flat_load_dword %0, %1 " GLC_SLC "\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : "=v"(result)
        : "v"(addr)
        : "memory"
    );
    #endif

    // Not yet applicable for NVIDIA, as it does not have an L3 cache. Placeholder.
    #ifdef __HIP_PLATFORM_NVIDIA__
    asm volatile(
        "ld.global.cv.u32 %0, [%1];\n\t"
        : "=r"(result)
        : "l"(addr)
        : "memory"
    );
    #endif

    return result;
}


#define V_TO_SGPR32(dest_s, src_var) \
    asm volatile ( \
        "v_readfirstlane_b32 " #dest_s ", %0\n\t" \
        "s_waitcnt lgkmcnt(0)\n\t" \
        "s_waitcnt vmcnt(0)\n\t" \
        : /* No Outputs */ \
        : "v"(src_var) \
        : #dest_s /* prevent compiler from reusing this register */)

#define V_TO_SGPR64(dest_s_lo, dest_s_hi, src_v64)  \
  do { \
    uint32_t _v2s_tmp_lo = static_cast<uint32_t>(src_v64); \
    uint32_t _v2s_tmp_hi = static_cast<uint32_t>((src_v64) >> 32); \
    asm volatile ( \
        "v_readfirstlane_b32 " #dest_s_lo ", %0\n\t" \
        "v_readfirstlane_b32 " #dest_s_hi ", %1\n\t" \
        "s_waitcnt lgkmcnt(0)\n\t" \
        "s_waitcnt vmcnt(0)\n\t" \
        : /* No Outputs */ \
        : "v"(_v2s_tmp_lo), "v"(_v2s_tmp_hi) \
        : #dest_s_lo, #dest_s_hi); \
  } while (0)


#define SGPR_TO_VAR32(var32, dest_s) \
    asm volatile( \
        "s_mov_b32 %0, " #dest_s "\n\t" \
        "s_waitcnt lgkmcnt(0)\n\t" \
        "s_waitcnt vmcnt(0)\n\t"\
        : "=s"(var32) \
        : \
        : \
    )

#define SGPR_TO_VAR64(var64, dest_s_lo, dest_s_hi) \
    do { \
        uint32_t _lo, _hi; \
        asm volatile( \
            "s_mov_b32 %0, " #dest_s_lo "\n\t" \
            "s_mov_b32 %1, " #dest_s_hi "\n\t" \
            "s_waitcnt lgkmcnt(0)\n\t" \
            "s_waitcnt vmcnt(0)\n\t"\
            : "=s"(_lo), "=s"(_hi) \
            : \
            : \
        ); \
        var64 = (static_cast<uint64_t>(_hi) << 32) | static_cast<uint64_t>(_lo); \
    } while (0)

