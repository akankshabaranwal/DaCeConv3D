import numpy as np
import torch
import torch.nn.functional as F
import tensorflow as tf

# Layout NHWC 
indepth = 5
inheight = 5
inwidth = 5
inchannels = 3
batchsize = 1
outchannels = 2
stride = 1
pad = 0
kdim = 3
outdepth = indepth - kdim + 1
outheight = inheight - kdim + 1
outwidth = inwidth - kdim + 1

# TODO: Maybe add a feature in the original benchmarking code to make it accept both layouts
# TODO: Replace the gemm formulation with the implicit gemm formulation
# Gemm formulation of 3D convolution
# Layout: NDHWC

def direct_conv3d(Input, kernel, Output):
    for n in range(0, batchsize):
        for d in range(0, outdepth):
            for h in range(0, outheight):
                for w in range(0, outwidth):
                    for oc in range(0, outchannels):
                        tmp = 0
                        for kd in range(0, kdim):
                            for kh in range(0, kdim):
                                for kw in range(0, kdim):
                                    for ic in range(0, inchannels):
                                        tmp += Input[ n, d+kd, h+kh, w+kw, ic]*kernel[oc, kd, kh, kw, ic]
                        Output[n, d, h, w, oc] = tmp


def implicit_gemm_conv3d( Input, kernel, Output ):
    GEMM_M = batchsize * outdepth * outheight * outwidth
    GEMM_N = outchannels
    GEMM_K = inchannels * kdim * kdim * kdim

    for gemm_i in range(0,GEMM_M):
        for gemm_j in range(0,GEMM_N):
            print(f'Implicit gemm computing: {gemm_i}, {gemm_j}')
            n = int(gemm_i/(outdepth*outheight*outwidth))
            nopq_residual = int(gemm_i % (outdepth*outheight*outwidth))
            
            o = int(nopq_residual/(outheight*outwidth))
            opq_residual = int(nopq_residual%(outheight*outwidth))
            
            p = int(opq_residual/outwidth)
            q = int(opq_residual%outwidth)
           
            accum = np.zeros([1])
            k = gemm_j

            for gemm_k in range(0, GEMM_K):
                
                c = int(gemm_k/(kdim*kdim*kdim))
                ctrs_residual = int(gemm_k%(kdim*kdim*kdim))

                t = int(ctrs_residual/(kdim*kdim))
                trs_residual = int(ctrs_residual%(kdim*kdim))

                r = int(trs_residual/kdim)
                s = int(trs_residual%kdim)
                
                d = o + t
                h = p + r
                w = q + s

                a = Input[n, d, h, w, c]
                b = kernel[k, t, r, s, c]
                accum[0] = accum[0] + a*b

            Output[ n, o, p, q, k] = accum[0]


Input = torch.rand(batchsize, indepth, inheight, inwidth, inchannels).cuda()
kernel = torch.rand(outchannels, kdim, kdim, kdim, inchannels).cuda()
Output = torch.zeros(batchsize, outdepth, outheight, outwidth, outchannels).cuda()

cudnn_input = Input.detach().clone()
cudnn_kernel = kernel.detach().clone()
cudnn_output = Output.detach().clone()

direct_input = Input.detach().clone()
direct_kernel = kernel.detach().clone()
direct_output = Output.detach().clone()

# 3D convolution code for verification
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
import libcudnn, ctypes

## cudnn fixed parameters init
cudnn_context = libcudnn.cudnnCreate()
tensor_format = libcudnn.cudnnTensorFormat['CUDNN_TENSOR_NHWC']
data_type = libcudnn.cudnnDataType['CUDNN_DATA_FLOAT']
tensor_dim = 5
conv_dim = tensor_dim-2
convolution_mode = libcudnn.cudnnConvolutionMode['CUDNN_CROSS_CORRELATION']
convolution_algo = libcudnn.cudnnConvolutionFwdAlgo['CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM']
''' Available algorithms for 3d convolution cudnn are: 
CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING
'''
alpha = 1.0
beta = 0
c_int_p = ctypes.POINTER(ctypes.c_int)
outdimsinit = [0, 0, 0, 0, 0]
# cudnn convolution descriptor
conv_desc = libcudnn.cudnnCreateConvolutionDescriptor()
pad = 0
dil = 1
stride = 1
convpad = [pad, pad, pad]
filtstr = [stride, stride, stride]
convdil = [dil, dil, dil]
libcudnn.cudnnSetConvolutionNdDescriptor(conv_desc, conv_dim, convpad, filtstr, convdil, convolution_mode, data_type)

# Input descriptor
dims = [batchsize, inchannels, indepth, inheight, inwidth]
in_desc = libcudnn.cudnnCreateTensorDescriptor()
libcudnn.cudnnSetTensorNdDescriptorEx(in_desc, tensor_format, data_type,  tensor_dim, dims)
print(f'Input dims: {dims}')
# TODO: Maybe simplify this conversion to gpuarray ??

in_data_g = gpuarray.to_gpu(cudnn_input.cpu().numpy().astype(np.float32))
in_data = ctypes.c_void_p(int(in_data_g.gpudata))

# Cudnn filter descriptor
filt_desc = libcudnn.cudnnCreateFilterDescriptor()
filt_dims = [outchannels, inchannels, kdim, kdim, kdim]
print(f'Filter dims: {filt_dims}')
libcudnn.cudnnSetFilterNdDescriptor(filt_desc, data_type, tensor_format, tensor_dim, filt_dims)
filt_data_g = gpuarray.to_gpu(cudnn_kernel.cpu().numpy().astype(np.float32))                                    
filt_data = ctypes.c_void_p(int(filt_data_g.gpudata))

# Cudnn output descriptor
outdims = libcudnn.cudnnGetConvolutionNdForwardOutputDim(conv_desc, in_desc, filt_desc, tensor_dim, outdimsinit)
out_n, out_c, out_d, out_h, out_w = outdims[0], outdims[1], outdims[2], outdims[3], outdims[4]
print(f'outdims: {outdims}')
out_desc = libcudnn.cudnnCreateTensorDescriptor()
libcudnn.cudnnSetTensorNdDescriptorEx(out_desc,  tensor_format, data_type, tensor_dim, outdims)
cudnn_output_g = gpuarray.to_gpu(cudnn_output.cpu().numpy().astype(np.float32))                             
cudnn_data = ctypes.c_void_p(int(cudnn_output_g.gpudata))


# Compute cudnn workspace size
ws_size = libcudnn.cudnnGetConvolutionForwardWorkspaceSize(cudnn_context, in_desc, filt_desc, conv_desc, out_desc, convolution_algo)
ws_ptr  = drv.mem_alloc(ws_size.value) if ws_size.value > 0 else 0
ws_data = ctypes.c_void_p(int(ws_ptr))

libcudnn.cudnnConvolutionForward(cudnn_context, alpha, in_desc, in_data, filt_desc, filt_data, 
                                conv_desc, convolution_algo, ws_data, ws_size.value, 
                                beta, out_desc, cudnn_data)


# Verification of implicit gemm code
implicit_gemm_conv3d(Input, kernel, Output)
Output = Output.cpu()
imgemm_output = gpuarray.to_gpu(Output.numpy().astype(np.float32))
print("CUDNN output:")
print(cudnn_output_g.get())
# print("implicit gemm output:")
# print(imgemm_output.get())
direct_conv3d(direct_input, direct_kernel, direct_output)
print("Direct convolution:")
print(direct_output)
direct_output_g = gpuarray.to_gpu(direct_output.cpu().numpy().astype(np.float32))

diff = np.linalg.norm((cudnn_output_g - imgemm_output).get()) / (batchsize * outchannels * indepth * inheight * inwidth )
print('Difference between cudnn and implicit gemm values:', diff)
diff = np.linalg.norm(direct_output.cpu() - imgemm_output.get()) / (batchsize * outchannels * indepth * inheight * inwidth )
print('Difference between direct convolution and implicit gemm values:', diff)
diff = np.linalg.norm(cudnn_output_g.get() - direct_output_g.get()) / (batchsize * outchannels * indepth * inheight * inwidth )
print('Difference between direct convolution and cudnn values:', diff)