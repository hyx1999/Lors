#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <assert.h>

#ifdef DEBUG
#define DEBUG_CODE(block)                                                           \
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) \
    {                                                                               \
        do                                                                          \
        {                                                                           \
            block                                                                   \
        } while (0);                                                                \
    }
#else
#define DEBUG_CODE(block)
#endif

namespace lors
{

    template <typename T>
    __device__ void swap(T &a, T &b)
    {
        T c = a;
        a = b;
        b = c;
    }

    template <typename T>
    __device__ T init_t(float x)
    {
        return x;
    }

    template <>
    __device__ __half init_t(float x)
    {
        return __float2half(x);
    }

    template <>
    __device__ __nv_bfloat16 init_t(float x)
    {
        return __float2bfloat16(x);
    }

    __device__ float fp32_t(float x)
    {
        return x;
    }

    __device__ float fp32_t(__half x)
    {
        return __half2float(x);
    }

    __device__ float fp32_t(__nv_bfloat16 x)
    {
        return __bfloat162float(x);
    }

    template <typename T>
    __device__ T abs_t(T x)
    {
        DEBUG_CODE(
            printf("__habs: %.2f => %.2f\n", fp32_t(x), fp32_t(__habs(x)));)
        return __habs(x);
    }

    template <>
    __device__ float abs_t(float x)
    {
        DEBUG_CODE(
            printf("abs: %.2f => %.2f\n", x, abs(x));)
        return abs(x);
    }

    template <typename T>
    __device__ bool lt_t(T x, T y)
    {
        return __hlt(x, y);
    }

    template <>
    __device__ bool lt_t(float x, float y)
    {
        return x < y;
    }

    template <typename T>
    __device__ T mul_t(T a, T b)
    {
        return __hmul(a, b);
    }

    template <>
    __device__ float mul_t(float a, float b)
    {
        return a * b;
    }

}
