#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>

extern "C" torch::Tensor sparsify_6by8_fp32(torch::Tensor x, torch::Tensor scale);
extern "C" torch::Tensor sparsify_6by8_fp16(torch::Tensor x, torch::Tensor scale);
extern "C" torch::Tensor sparsify_6by8_bf16(torch::Tensor x, torch::Tensor scale);

extern "C" torch::Tensor sparsify_3by4_fp32(torch::Tensor x, torch::Tensor scale);
extern "C" torch::Tensor sparsify_3by4_fp16(torch::Tensor x, torch::Tensor scale);
extern "C" torch::Tensor sparsify_3by4_bf16(torch::Tensor x, torch::Tensor scale);

extern "C" torch::Tensor sparsify_2by4_fp32(torch::Tensor x, torch::Tensor scale);
extern "C" torch::Tensor sparsify_2by4_fp16(torch::Tensor x, torch::Tensor scale);
extern "C" torch::Tensor sparsify_2by4_bf16(torch::Tensor x, torch::Tensor scale);

torch::Tensor sparsify_6by8(torch::Tensor x, torch::Tensor scale) {
    TORCH_CHECK(x.dtype() == scale.dtype());
    if (x.dtype() == torch::kF32)
    {
        // printf("brach torch::kF32\n");
        return sparsify_6by8_fp32(x, scale);
    }
    else if (x.dtype() == torch::kF16)
    {
        // printf("brach torch::kF16\n");
        return sparsify_6by8_fp16(x, scale);
    }
    else if (x.dtype() == torch::kBFloat16)
    {
        // printf("brach torch::kBFloat16\n");
        return sparsify_6by8_bf16(x, scale);
    }
    else
        throw torch::TypeError("sparsify_6by8 does not support the dtype of input tensor.");
}

torch::Tensor sparsify_3by4(torch::Tensor x, torch::Tensor scale) {
    TORCH_CHECK(x.dtype() == scale.dtype());
    if (x.dtype() == torch::kF32)
    {
        // printf("brach torch::kF32\n");
        return sparsify_3by4_fp32(x, scale);
    }
    else if (x.dtype() == torch::kF16)
    {
        // printf("brach torch::kF16\n");
        return sparsify_3by4_fp16(x, scale);
    }
    else if (x.dtype() == torch::kBFloat16)
    {
        // printf("brach torch::kBFloat16\n");
        return sparsify_3by4_bf16(x, scale);
    }
    else
        throw torch::TypeError("sparsify_3by4 does not support the dtype of input tensor.");
}

torch::Tensor sparsify_2by4(torch::Tensor x, torch::Tensor scale) {
    TORCH_CHECK(x.dtype() == scale.dtype());
    if (x.dtype() == torch::kF32)
    {
        // printf("brach torch::kF32\n");
        return sparsify_2by4_fp32(x, scale);
    }
    else if (x.dtype() == torch::kF16)
    {
        // printf("brach torch::kF16\n");
        return sparsify_2by4_fp16(x, scale);
    }
    else if (x.dtype() == torch::kBFloat16)
    {
        // printf("brach torch::kBFloat16\n");
        return sparsify_2by4_bf16(x, scale);
    }
    else
        throw torch::TypeError("sparsify_2by4 does not support the dtype of input tensor.");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparsify_6by8", &sparsify_2by4, "6:8 Sparify Kernel (CUDA)");
    m.def("sparsify_3by4", &sparsify_2by4, "3:4 Sparify Kernel (CUDA)");
    m.def("sparsify_2by4", &sparsify_2by4, "2:4 Sparify Kernel (CUDA)");
}
