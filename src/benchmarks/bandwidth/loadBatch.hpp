#pragma once

#include "benchmarks/base.hpp"
#include "config.hpp"
#include "typedef/vectorTypes.hpp"
#include "utils/util.hpp"

#include <cstdint>
#include <cstddef>

// Read bandwidth kernels issue a batch of N loads per thread and only then wait for them, so N loads are in flight.
// N is a template parameter of the kernels. The sweeps try N = 1, 2, 4, ... up to these limits, more than today's GPUs
// keep in flight, so the results show the plateau and leave headroom for future GPUs.
static constexpr uint32_t MAX_LOADS_PER_BATCH_SINGLE_BLOCK = 128;   // L1, shared memory, scalar L1, read-only, texture
static constexpr uint32_t MAX_LOADS_PER_BATCH_MULTIPLE_BLOCKS = 16; // L2, L3, main memory

#define CHECK_LOADS_PER_BATCH(N, MAX) \
    static_assert((N) <= (MAX) && ((N) & ((N) - 1)) == 0, "loads per batch must be a power of two up to " #MAX)

// Single block: load k of a batch reads k * 128 bytes after the first one, so the loads hit different cache lines
// (on AMD this repeats after 32 loads, as the offset field of the load instruction has 12 bits). Arrays get this padding.
static constexpr size_t BATCH_STRIDE_BYTES = 128;
static constexpr size_t BATCH_PADDING_BYTES = MAX_LOADS_PER_BATCH_SINGLE_BLOCK * BATCH_STRIDE_BYTES;

// Shared memory: load k reads k * 128 bytes further on AMD (also repeating after 32 loads) and k * 16 bytes on NVIDIA,
// without bank conflicts on either
static constexpr uint32_t SHARED_BATCH_PADDING_BYTES = 4096;


#ifdef __HIP_PLATFORM_AMD__
// AMD batches are inline asm: ONE asm statement with the N loads followed by one s_waitcnt.
// All loads write the same registers (the data is not used), and these registers are listed as clobbers.
// As the wait is inside the same asm, the compiler never uses them while a load can still write them.
// The asm text has to be a string literal, so the macros below write the N loads out.

// LOAD(offset) at the byte offsets 0, 128, 256, ... (single block: one address, the loads differ in the offset)
#define AMD_OFFSETS_1(LOAD)   LOAD(0)
#define AMD_OFFSETS_2(LOAD)   AMD_OFFSETS_1(LOAD) LOAD(128)
#define AMD_OFFSETS_4(LOAD)   AMD_OFFSETS_2(LOAD) LOAD(256) LOAD(384)
#define AMD_OFFSETS_8(LOAD)   AMD_OFFSETS_4(LOAD) LOAD(512) LOAD(640) LOAD(768) LOAD(896)
#define AMD_OFFSETS_16(LOAD)  AMD_OFFSETS_8(LOAD) LOAD(1024) LOAD(1152) LOAD(1280) LOAD(1408) \
                              LOAD(1536) LOAD(1664) LOAD(1792) LOAD(1920)
#define AMD_OFFSETS_32(LOAD)  AMD_OFFSETS_16(LOAD) LOAD(2048) LOAD(2176) LOAD(2304) LOAD(2432) \
                              LOAD(2560) LOAD(2688) LOAD(2816) LOAD(2944) \
                              LOAD(3072) LOAD(3200) LOAD(3328) LOAD(3456) \
                              LOAD(3584) LOAD(3712) LOAD(3840) LOAD(3968)
#define AMD_OFFSETS_64(LOAD)  AMD_OFFSETS_32(LOAD) AMD_OFFSETS_32(LOAD)
#define AMD_OFFSETS_128(LOAD) AMD_OFFSETS_64(LOAD) AMD_OFFSETS_64(LOAD)

// asm with N loads LOAD(offset) from address (operand %0), then WAIT; the clobbers are the registers the loads write
#define AMD_SINGLE_BLOCK_BATCH(N, LOAD, WAIT, address, ...) \
    if constexpr ((N) == 1)   { asm volatile (AMD_OFFSETS_1(LOAD) WAIT : : address : "memory", __VA_ARGS__); } \
    if constexpr ((N) == 2)   { asm volatile (AMD_OFFSETS_2(LOAD) WAIT : : address : "memory", __VA_ARGS__); } \
    if constexpr ((N) == 4)   { asm volatile (AMD_OFFSETS_4(LOAD) WAIT : : address : "memory", __VA_ARGS__); } \
    if constexpr ((N) == 8)   { asm volatile (AMD_OFFSETS_8(LOAD) WAIT : : address : "memory", __VA_ARGS__); } \
    if constexpr ((N) == 16)  { asm volatile (AMD_OFFSETS_16(LOAD) WAIT : : address : "memory", __VA_ARGS__); } \
    if constexpr ((N) == 32)  { asm volatile (AMD_OFFSETS_32(LOAD) WAIT : : address : "memory", __VA_ARGS__); } \
    if constexpr ((N) == 64)  { asm volatile (AMD_OFFSETS_64(LOAD) WAIT : : address : "memory", __VA_ARGS__); } \
    if constexpr ((N) == 128) { asm volatile (AMD_OFFSETS_128(LOAD) WAIT : : address : "memory", __VA_ARGS__); }

// Multiple blocks: load k reads from its own address p[k] (operand %k)
#define AMD_GRID_LOAD(k, CACHE) "flat_load_dwordx4 v[20:23], %" #k " " CACHE "\n\t"

#define AMD_GRID_LOADS_1(CACHE)  AMD_GRID_LOAD(0, CACHE)
#define AMD_GRID_LOADS_2(CACHE)  AMD_GRID_LOADS_1(CACHE) AMD_GRID_LOAD(1, CACHE)
#define AMD_GRID_LOADS_4(CACHE)  AMD_GRID_LOADS_2(CACHE) AMD_GRID_LOAD(2, CACHE) AMD_GRID_LOAD(3, CACHE)
#define AMD_GRID_LOADS_8(CACHE)  AMD_GRID_LOADS_4(CACHE) AMD_GRID_LOAD(4, CACHE) AMD_GRID_LOAD(5, CACHE) \
                                 AMD_GRID_LOAD(6, CACHE) AMD_GRID_LOAD(7, CACHE)
#define AMD_GRID_LOADS_16(CACHE) AMD_GRID_LOADS_8(CACHE) AMD_GRID_LOAD(8, CACHE) AMD_GRID_LOAD(9, CACHE) \
                                 AMD_GRID_LOAD(10, CACHE) AMD_GRID_LOAD(11, CACHE) AMD_GRID_LOAD(12, CACHE) \
                                 AMD_GRID_LOAD(13, CACHE) AMD_GRID_LOAD(14, CACHE) AMD_GRID_LOAD(15, CACHE)

#define AMD_GRID_ADDRESSES_1(p)  "v"(p[0])
#define AMD_GRID_ADDRESSES_2(p)  AMD_GRID_ADDRESSES_1(p), "v"(p[1])
#define AMD_GRID_ADDRESSES_4(p)  AMD_GRID_ADDRESSES_2(p), "v"(p[2]), "v"(p[3])
#define AMD_GRID_ADDRESSES_8(p)  AMD_GRID_ADDRESSES_4(p), "v"(p[4]), "v"(p[5]), "v"(p[6]), "v"(p[7])
#define AMD_GRID_ADDRESSES_16(p) AMD_GRID_ADDRESSES_8(p), "v"(p[8]), "v"(p[9]), "v"(p[10]), "v"(p[11]), \
                                 "v"(p[12]), "v"(p[13]), "v"(p[14]), "v"(p[15])

// asm with N loads from the addresses p[0..N-1], then the wait; CACHE is the cache policy of the loads
#define AMD_GRID_BATCH(N, CACHE, p) \
    if constexpr ((N) == 1)  { asm volatile (AMD_GRID_LOADS_1(CACHE) "s_waitcnt vmcnt(0)\n\t" : : AMD_GRID_ADDRESSES_1(p) : "memory", "v20", "v21", "v22", "v23"); } \
    if constexpr ((N) == 2)  { asm volatile (AMD_GRID_LOADS_2(CACHE) "s_waitcnt vmcnt(0)\n\t" : : AMD_GRID_ADDRESSES_2(p) : "memory", "v20", "v21", "v22", "v23"); } \
    if constexpr ((N) == 4)  { asm volatile (AMD_GRID_LOADS_4(CACHE) "s_waitcnt vmcnt(0)\n\t" : : AMD_GRID_ADDRESSES_4(p) : "memory", "v20", "v21", "v22", "v23"); } \
    if constexpr ((N) == 8)  { asm volatile (AMD_GRID_LOADS_8(CACHE) "s_waitcnt vmcnt(0)\n\t" : : AMD_GRID_ADDRESSES_8(p) : "memory", "v20", "v21", "v22", "v23"); } \
    if constexpr ((N) == 16) { asm volatile (AMD_GRID_LOADS_16(CACHE) "s_waitcnt vmcnt(0)\n\t" : : AMD_GRID_ADDRESSES_16(p) : "memory", "v20", "v21", "v22", "v23"); }
#endif


// Shared memory batch. AMD: address is the byte address in the LDS. NVIDIA: address is in the shared state space
// (cvta.to.shared); all four words are used and all addresses differ, otherwise the compiler narrows or merges the loads.
#ifdef __HIP_PLATFORM_AMD__
#define AMD_SHARED_LOAD(offset) "ds_read_b128 v[20:23], %0 offset:" #offset "\n\t"

template <uint32_t LoadsPerBatch>
__device__ __forceinline__ void amdSharedLoadBatch(uint32_t address)
{
    AMD_SINGLE_BLOCK_BATCH(LoadsPerBatch, AMD_SHARED_LOAD, "s_waitcnt lgkmcnt(0)\n\t", "v"(address), "v20", "v21", "v22", "v23")
}
#endif

#ifdef __HIP_PLATFORM_NVIDIA__
template <uint32_t LoadsPerBatch>
__device__ __forceinline__ uint32_t nvidiaSharedLoadBatch(uint64_t address)
{
    uint32_t dummy = 0;

    #pragma unroll
    for (uint32_t k = 0; k < LoadsPerBatch; ++k)
    {
        uint32_t x, y, z, w;
        asm volatile ("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(address + k * 16) : "memory");
        dummy ^= x ^ y ^ z ^ w;
    }

    return dummy;
}
#endif


// Multiple blocks: load k of a batch reads element i + k * gridStride, so every element is still read once per rep.
// (With the single-block layout, neighbouring threads would load the same lines within a batch, from a closer cache.)
// BypassL1 reads past L1 for the L2 benchmark: GLC bit on AMD, ld.global.cg on NVIDIA.
#ifdef __HIP_PLATFORM_AMD__
template <uint32_t LoadsPerBatch, bool BypassL1>
__device__ __forceinline__ void amdGridLoadBatch(const uint32v4 *src, size_t gridStride)
{
    const uint32v4 *p[LoadsPerBatch];

    #pragma unroll
    for (uint32_t k = 0; k < LoadsPerBatch; ++k) { p[k] = src + k * gridStride; }

    if constexpr (BypassL1) { AMD_GRID_BATCH(LoadsPerBatch, GLC, p) }
    else                    { AMD_GRID_BATCH(LoadsPerBatch, "", p) }
}
#endif

#ifdef __HIP_PLATFORM_NVIDIA__
// All four words are used, otherwise the compiler narrows the loads
template <uint32_t LoadsPerBatch, bool BypassL1>
__device__ __forceinline__ uint32_t nvidiaGridLoadBatch(const uint32v4 *src, size_t gridStride)
{
    uint32_t dummy = 0;

    #pragma unroll
    for (uint32_t k = 0; k < LoadsPerBatch; ++k)
    {
        uint32_t x, y, z, w;
        if constexpr (BypassL1)
        {
            asm volatile ("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(src + k * gridStride) : "memory");
        }
        else
        {
            asm volatile ("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(src + k * gridStride) : "memory");
        }
        dummy ^= x ^ y ^ z ^ w;
    }

    return dummy;
}
#endif

// Reads every element of src[0..n) once with this thread's share of the grid-stride loop: batches of N loads,
// then single loads for the rest
template <uint32_t LoadsPerBatch, bool BypassL1>
__device__ __forceinline__ uint32_t gridReadPass(const uint32v4 *src, size_t first, size_t n, size_t gridStride)
{
    uint32_t dummy = 0;
    size_t i = first;

    for (; i + (LoadsPerBatch - 1) * gridStride < n; i += LoadsPerBatch * gridStride)
    {
        #ifdef __HIP_PLATFORM_AMD__
        amdGridLoadBatch<LoadsPerBatch, BypassL1>(src + i, gridStride);
        #endif

        #ifdef __HIP_PLATFORM_NVIDIA__
        dummy ^= nvidiaGridLoadBatch<LoadsPerBatch, BypassL1>(src + i, gridStride);
        #endif
    }

    for (; i < n; i += gridStride)
    {
        #ifdef __HIP_PLATFORM_AMD__
        amdGridLoadBatch<1, BypassL1>(src + i, gridStride);
        #endif

        #ifdef __HIP_PLATFORM_NVIDIA__
        dummy ^= nvidiaGridLoadBatch<1, BypassL1>(src + i, gridStride);
        #endif
    }

    return dummy;
}
