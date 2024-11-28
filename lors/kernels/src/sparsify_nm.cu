#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <assert.h>
#include "helper_cuda.cuh"
#include "lors_utils.cuh"

constexpr int BlockDimX = 16;
constexpr int BlockDimY = 16;

// 内核函数
template <typename T, int N, int M>
__global__ void sparsify_nm_kernel(T *W, T *Ws, T *scale,
                                   int stride_W_0, int stride_W_1,
                                   int stride_Ws_0, int stride_Ws_1,
                                   int stride_scale,
                                   int width, int height)
{
    extern __shared__ __align__(sizeof(T)) unsigned char __tile[];
    T *tile_w = reinterpret_cast<T *>(__tile);
    T *tile_s = tile_w + (blockDim.x * blockDim.y * M);

    // 将数据加载到共享内存
    for (int i = 0; i < M; i++)
    {
        int x_id = blockIdx.x * blockDim.x * M + i * blockDim.x + threadIdx.x;
        int y_id = blockIdx.y * blockDim.y + threadIdx.y;
        int id = x_id * stride_W_0 + y_id * stride_W_1;
        int tile_id = blockDim.x * M * threadIdx.y + i * blockDim.x + threadIdx.x;
        int s_id = x_id * stride_scale;
        int tile_s_id = i * blockDim.x + threadIdx.x;
        if (x_id < width && y_id < height)
        {
            tile_w[tile_id] = W[id];
        }
        else
        {
            tile_w[tile_id] = lors::init_t<T>(0.0f);
        }
        if (threadIdx.y == 0)
        {
            if (x_id < width)
            {
                tile_s[tile_s_id] = scale[s_id];
            }
            else
            {
                tile_s[tile_s_id] = lors::init_t<T>(1.0f);
            }
        }
    }

    __syncthreads();

    DEBUG_CODE(
        printf("print tile_w:\n");
        for (int i = 0; i < BlockDimY; i++) {
            for (int j = 0; j < BlockDimX * M; j++)
            {
                int tile_id = i * (BlockDimX * M) + j;
                printf("%.2f ", lors::fp32_t(tile_w[tile_id]));
            }
            printf("\n");
        })

    // 找到最大的N个值及其索引
    T values[M];
    T sorted_values[M];
    int sorted_indices[M];
    int base = blockDim.x * M * threadIdx.y + threadIdx.x * M;
    int base_s = threadIdx.x * M;
    for (int i = 0; i < M; ++i)
    {
        sorted_values[i] = lors::abs_t(lors::mul_t(tile_w[base + i], tile_s[base_s + i]));
        sorted_indices[i] = i;
    }
    for (int i = 0; i < M; ++i)
    {
        for (int j = i; j > 0; j--)
        {
            if (lors::lt_t(sorted_values[j - 1], sorted_values[j]))
            {
                lors::swap(sorted_values[j], sorted_values[j - 1]);
                lors::swap(sorted_indices[j], sorted_indices[j - 1]);
            }
        }
    }

    for (int i = 0; i < N; i++)
        values[sorted_indices[i]] = tile_w[base + sorted_indices[i]];
    for (int i = N; i < M; i++)
        values[sorted_indices[i]] = lors::init_t<T>(0.0f);
    for (int i = 0; i < M; i++)
        tile_w[base + i] = values[i];

    DEBUG_CODE(
        printf("tile_values:\n");
        for (int i = 0; i < M; i++)
            printf("%.2f ", tile_w[base + i]);
        printf("\n");
        printf("sorted_values:\n");
        for (int i = 0; i < M; i++)
            printf("%.2f ", sorted_values[i]);
        printf("\n");
        printf("sorted_indices:\n");
        for (int i = 0; i < M; i++)
            printf("%d ", sorted_indices[i]);
        printf("\n");)

    __syncthreads();

    DEBUG_CODE(
        printf("print tile_w:\n");
        for (int i = 0; i < BlockDimY; i++) {
            for (int j = 0; j < BlockDimX * M; j++)
            {
                int tile_id = i * (BlockDimX * M) + j;
                printf("%.2f ", lors::fp32_t(tile_w[tile_id]));
            }
            printf("\n");
        })

    // 将结果写回到全局内存
    for (int i = 0; i < M; i++)
    {
        int x_id = blockIdx.x * blockDim.x * M + i * blockDim.x + threadIdx.x;
        int y_id = blockIdx.y * blockDim.y + threadIdx.y;
        if (x_id < width && y_id < height)
        {
            int id = x_id * stride_Ws_0 + y_id * stride_Ws_1;
            int tile_id = blockDim.x * M * threadIdx.y + i * blockDim.x + threadIdx.x;
            Ws[id] = tile_w[tile_id];
        }
    }
}

template <typename T, typename Tp, int N, int M>
torch::Tensor sparsify_nm(torch::Tensor x, torch::Tensor scale)
{
    TORCH_CHECK(x.dim() == 2);
    TORCH_CHECK(scale.dim() == 1);
    TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(scale.device().type() == at::DeviceType::CUDA);
    auto y = torch::empty_like(x);
    int width = x.size(1);
    int height = x.size(0);

    DEBUG_CODE(
        printf("width  = %d\n", width);
        printf("height = %d\n", height);
        printf("x.sride(1): %d\n", x.stride(1));
        printf("x.sride(0): %d\n", x.stride(0));
        printf("y.sride(1): %d\n", y.stride(1));
        printf("y.sride(0): %d\n", y.stride(0));)
    dim3 threadsPerBlock(BlockDimX, BlockDimY); // 每个block处理4 * 4 * M个元素
    dim3 numBlocks((width + BlockDimX * M - 1) / (BlockDimX * M),
                   (height + BlockDimY - 1) / BlockDimY);
    uint32_t sharedMemSize = (BlockDimX * M * BlockDimY +
                              BlockDimX * M) *
                             sizeof(T);

    // 调用CUDA内核
    sparsify_nm_kernel<T, N, M><<<numBlocks, threadsPerBlock, sharedMemSize>>>(
        reinterpret_cast<T *>(x.data_ptr<Tp>()),
        reinterpret_cast<T *>(y.data_ptr<Tp>()),
        reinterpret_cast<T *>(scale.data_ptr<Tp>()),
        x.stride(1), x.stride(0),
        y.stride(1), y.stride(0),
        scale.stride(0),
        width, height);
    checkCudaErrors(cudaGetLastError());
    return y;
}

// 6by8
extern "C" torch::Tensor sparsify_6by8_fp32(torch::Tensor x, torch::Tensor scale)
{
    return sparsify_nm<float, float, 6, 8>(x, scale);
}

extern "C" torch::Tensor sparsify_6by8_fp16(torch::Tensor x, torch::Tensor scale)
{
    return sparsify_nm<__half, at::Half, 6, 8>(x, scale);
}

extern "C" torch::Tensor sparsify_6by8_bf16(torch::Tensor x, torch::Tensor scale)
{
    return sparsify_nm<__nv_bfloat16, at::BFloat16, 6, 8>(x, scale);
}


// 3by4
extern "C" torch::Tensor sparsify_3by4_fp32(torch::Tensor x, torch::Tensor scale)
{
    return sparsify_nm<float, float, 3, 4>(x, scale);
}

extern "C" torch::Tensor sparsify_3by4_fp16(torch::Tensor x, torch::Tensor scale)
{
    return sparsify_nm<__half, at::Half, 3, 4>(x, scale);
}

extern "C" torch::Tensor sparsify_3by4_bf16(torch::Tensor x, torch::Tensor scale)
{
    return sparsify_nm<__nv_bfloat16, at::BFloat16, 3, 4>(x, scale);
}


// 2by4
extern "C" torch::Tensor sparsify_2by4_fp32(torch::Tensor x, torch::Tensor scale)
{
    return sparsify_nm<float, float, 2, 4>(x, scale);
}

extern "C" torch::Tensor sparsify_2by4_fp16(torch::Tensor x, torch::Tensor scale)
{
    return sparsify_nm<__half, at::Half, 2, 4>(x, scale);
}

extern "C" torch::Tensor sparsify_2by4_bf16(torch::Tensor x, torch::Tensor scale)
{
    return sparsify_nm<__nv_bfloat16, at::BFloat16, 2, 4>(x, scale);
}
