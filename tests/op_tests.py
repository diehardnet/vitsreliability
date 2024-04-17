#!/usr/bin/python3


import torch


def profile_linear():
    """
    73% of the call is ampere_sgemm_32x128_tn,
    27% is the void splitKreduce_kernel
    """
    m = torch.nn.Linear(512, 32).to("cuda")
    input_tensor = torch.randn(1024, 256).to("cuda")
    output = m(input_tensor)
    print(output.size())


def profile_layer_norm():
    """
    Kernel at low level
    at::native::<unnamed>::vectorized_layer_norm_kernel
    """
    # Image Example
    n, c, h, w = 32, 128, 256, 512
    input_tensor = torch.randn(n, c, h, w).to("cuda")
    # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
    # as shown in the image below
    layer_norm = torch.nn.LayerNorm([c, h, w]).to("cuda")
    output = layer_norm(input_tensor)
    print(output.size())


def profile_conv2d():
    """
    Gemm kernel is 94% of the cycles --> Can vary according to the input matrix 91% if square matrix
    void implicit_convolve_sgemm
    at::native::elementwise_kernel is the rest 6%
    """
    # # With square kernels and equal stride
    # m = torch.nn.Conv2d(16, 33, 3, stride=2).to("cuda")
    # # non-square kernels and unequal stride and with padding
    m = torch.nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2)).to("cuda")
    # non-square kernels and unequal stride and with padding and dilation
    # m = torch.nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1)).to("cuda")
    input_tensor = torch.randn(20, 16, 50, 100).to("cuda")
    output = m(input_tensor)
    print(output.size())


# Test linear
# profile_linear()

# Test layer norm
# profile_layer_norm()

# Test layer conv2d
profile_conv2d()
