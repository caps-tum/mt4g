#pragma once
#include <stdint.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
  #define HD __host__ __device__
#else
  #define HD
#endif

/* ---------- Clang-Pfad: echte VGPR-Vektoren ---------- */
#if defined(__clang__) && __has_attribute(ext_vector_type)
  typedef uint32_t uint32v4 __attribute__((ext_vector_type(4)));
  typedef uint32_t uint32v8 __attribute__((ext_vector_type(8)));
  typedef uint32_t uint32v16 __attribute__((ext_vector_type(16)));

/* ---------- NVCC-Fallback: POD-Struct + Operatoren --- */
#else
  struct uint32v4 {
      uint32_t x, y, z, w;

      /* no defaulted ctor → no #20012 */
      HD constexpr uint32v4() : x(0), y(0), z(0), w(0) {}
      HD constexpr uint32v4(uint32_t a,uint32_t b,
                            uint32_t c,uint32_t d)
          : x(a), y(b), z(c), w(d) {}

      /* element access */
      HD uint32_t&       operator[](int i)       { return (&x)[i]; }
      HD const uint32_t& operator[](int i) const { return (&x)[i]; }

      /* a few basic ops */
      HD uint32v4 operator+(const uint32v4& o) const { return {x+o.x,y+o.y,z+o.z,w+o.w}; }
      HD uint32v4 operator-(const uint32v4& o) const { return {x-o.x,y-o.y,z-o.z,w-o.w}; }
      HD uint32v4 operator&(const uint32v4& o) const { return {x&o.x,y&o.y,z&o.z,w&o.w}; }
      HD uint32v4 operator|(const uint32v4& o) const { return {x|o.x,y|o.y,z|o.z,w|o.w}; }
  };

  struct uint32v8 {
      uint32_t v[8];

      HD constexpr uint32v8() : v{0,0,0,0,0,0,0,0} {}
      HD uint32_t&       operator[](int i)       { return v[i]; }
      HD const uint32_t& operator[](int i) const { return v[i]; }
  };

  struct uint32v16 {
      uint32_t v[16];

      HD constexpr uint32v16() : v{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0} {}
      HD uint32_t&       operator[](int i)       { return v[i]; }
      HD const uint32_t& operator[](int i) const { return v[i]; }
  };
#endif

#undef HD
