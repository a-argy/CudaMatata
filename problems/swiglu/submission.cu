// submission.cu
// Baseline SwiGLU CUDA Kernel Implementation
//
// SwiGLU(x, W, V, b, c, beta) = Swish(xW + b) ⊙ (xV + c)
// where Swish(x) = x * sigmoid(beta * x)

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Baseline CUDA kernel for SwiGLU
// Each thread computes one output element
template <typename scalar_t>
__global__ void kernel_body(
    const scalar_t* __restrict__ x,      // [batch_size, seq_len, in_features]
    const scalar_t* __restrict__ W,      // [in_features, hidden_size]
    const scalar_t* __restrict__ V,      // [in_features, hidden_size]
    const scalar_t* __restrict__ b,      // [hidden_size]
    const scalar_t* __restrict__ c,      // [hidden_size]
    float beta,
    scalar_t* __restrict__ output,       // [batch_size, seq_len, hidden_size]
    int batch_size,
    int seq_len,
    int in_features,
    int hidden_size
) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Total number of output elements
    int total_outputs = batch_size * seq_len * hidden_size;
    
    if (idx >= total_outputs) return;
    
    // Decompose idx into (batch, seq, hidden) indices
    int h = idx % hidden_size;
    int s = (idx / hidden_size) % seq_len;
    int batch = idx / (hidden_size * seq_len);
    
    // Compute base index for x[batch][s][:]
    int x_base = batch * seq_len * in_features + s * in_features;
    
    // Compute gate = xW + b (dot product of x[batch][s][:] with W[:][h])
    scalar_t gate = static_cast<scalar_t>(0);
    for (int i = 0; i < in_features; i++) {
        gate += x[x_base + i] * W[i * hidden_size + h];
    }
    gate += b[h];
    
    // Apply Swish activation: swish(gate) = gate * sigmoid(beta * gate)
    scalar_t sigmoid_val = static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + exp(-beta * gate));
    scalar_t swish_gate = gate * sigmoid_val;
    
    // Compute value = xV + c (dot product of x[batch][s][:] with V[:][h])
    scalar_t value = static_cast<scalar_t>(0);
    for (int i = 0; i < in_features; i++) {
        value += x[x_base + i] * V[i * hidden_size + h];
    }
    value += c[h];
    
    // Output = swish_gate * value
    output[idx] = swish_gate * value;
}

// Main function called from Python
torch::Tensor custom_kernel(
    torch::Tensor x,      // [batch_size, seq_len, in_features]
    torch::Tensor W,      // [in_features, hidden_size]
    torch::Tensor V,      // [in_features, hidden_size]
    torch::Tensor b,      // [hidden_size]
    torch::Tensor c,      // [hidden_size]
    float beta
) {
    // Get dimensions
    int batch_size = x.size(0);
    int seq_len = x.size(1);
    int in_features = x.size(2);
    int hidden_size = W.size(1);
    
    // Allocate output tensor [batch_size, seq_len, hidden_size]
    auto output = torch::empty({batch_size, seq_len, hidden_size}, x.options());
    
    // Calculate grid and block dimensions
    int total_elements = batch_size * seq_len * hidden_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "kernel_body", ([&] {
        kernel_body<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            W.data_ptr<scalar_t>(),
            V.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            beta,
            output.data_ptr<scalar_t>(),
            batch_size,
            seq_len,
            in_features,
            hidden_size
        );
    }));

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();

    return output;
}
