// swiglu_fp32_final.cu
// SwiGLU FP32 with PTX L2 prefetching - optimized for H100
// Performance: 14.0 ms (ranked on swiglu leaderboard)

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__device__ __forceinline__ void prefetch_l2(const void* ptr) {
    asm volatile("prefetch.global.L2 [%0];" : : "l"(ptr));
}

__global__ __launch_bounds__(256, 4) void swish_mul_ptx(
    const float* __restrict__ gate,
    const float* __restrict__ b,
    const float* __restrict__ c,
    const float beta,
    float* __restrict__ output,
    const int total,
    const int N
) {
    const int base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    const int prefetch_offset = gridDim.x * blockDim.x * 4;
    if (base + prefetch_offset < total) {
        prefetch_l2(gate + base + prefetch_offset);
        prefetch_l2(output + base + prefetch_offset);
    }
    
    if (base + 3 < total) {
        float4 g = __ldg(reinterpret_cast<const float4*>(gate + base));
        float4 v = *reinterpret_cast<float4*>(output + base);
        
        const int h = base % N;
        
        if (h + 3 < N) {
            float4 bv = __ldg(reinterpret_cast<const float4*>(b + h));
            float4 cv = __ldg(reinterpret_cast<const float4*>(c + h));
            g.x += bv.x; g.y += bv.y; g.z += bv.z; g.w += bv.w;
            v.x += cv.x; v.y += cv.y; v.z += cv.z; v.w += cv.w;
        } else {
            g.x += __ldg(b + h);
            g.y += __ldg(b + (h+1) % N);
            g.z += __ldg(b + (h+2) % N);
            g.w += __ldg(b + (h+3) % N);
            v.x += __ldg(c + h);
            v.y += __ldg(c + (h+1) % N);
            v.z += __ldg(c + (h+2) % N);
            v.w += __ldg(c + (h+3) % N);
        }
        
        float4 out;
        out.x = g.x / (1.0f + __expf(-beta * g.x)) * v.x;
        out.y = g.y / (1.0f + __expf(-beta * g.y)) * v.y;
        out.z = g.z / (1.0f + __expf(-beta * g.z)) * v.z;
        out.w = g.w / (1.0f + __expf(-beta * g.w)) * v.w;
        
        *reinterpret_cast<float4*>(output + base) = out;
        
    } else if (base < total) {
        for (int i = 0; i < 4 && base + i < total; i++) {
            const int idx = base + i;
            const int h = idx % N;
            float gv = __ldg(gate + idx) + __ldg(b + h);
            float sw = gv / (1.0f + __expf(-beta * gv));
            output[idx] = sw * (output[idx] + __ldg(c + h));
        }
    }
}

static cublasHandle_t handle = nullptr;
static bool ready = false;

torch::Tensor custom_kernel(
    torch::Tensor x, torch::Tensor W, torch::Tensor V,
    torch::Tensor b, torch::Tensor c, float beta
) {
    if (!ready) { cublasCreate(&handle); ready = true; }
    
    const int bs = x.size(0), sl = x.size(1), K = x.size(2), N = W.size(1);
    const int M = bs * sl, total = M * N;
    
    auto x2d = x.view({M, K}).contiguous();
    auto gate = torch::empty({M, N}, x.options());
    auto out = torch::empty({M, N}, x.options());
    
    const float one = 1.0f, zero = 0.0f;
    
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one,
                W.data_ptr<float>(), N, x2d.data_ptr<float>(), K, &zero, 
                gate.data_ptr<float>(), N);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one,
                V.data_ptr<float>(), N, x2d.data_ptr<float>(), K, &zero, 
                out.data_ptr<float>(), N);
    
    swish_mul_ptx<<<(total + 1023) / 1024, 256>>>(
        gate.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
        beta, out.data_ptr<float>(), total, N);
    
    return out.view({bs, sl, N});
}

