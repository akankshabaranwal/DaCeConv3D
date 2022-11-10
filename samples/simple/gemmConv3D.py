import numpy as np
import torch
import torch.nn.functional as F
import tensorflow as tf

# Layout NHWC 
indepth = 8
inheight = 8
inwidth = 8
inchannels = 4
batchsize = 2
outchannels = 8
stride = 1
pad = 0
kdim = 3
outdepth = indepth - kdim + 1
outheight = inheight - kdim + 1
outwidth = inwidth - kdim + 1

# TODO: Write gemm formulation of 3D convolution
# TODO: Port cudnn 3D convolution and match the layout
# TODO: Maybe add a feature in the original benchmarking code to make it accept both layouts
# TODO: Replace the gemm formulation with the implicit gemm formulation
# Gemm formulation of 3D convolution
# NHWC, KRSC, NPQK layouts. 

def dace_implicit_gemm_conv2d( Input, kernel, Output ):
    GEMM_M = batchsize*outheight*outwidth
    GEMM_N = outchannels
    GEMM_K = inchannels * kdim * kdim

    for gemm_i in range(0,GEMM_M):
        for gemm_j in range(0,GEMM_N):
            n = int(gemm_i/(outheight*outwidth))
            npq_residual = int(gemm_i % (outheight*outwidth))
            
            p = int(npq_residual/outwidth)
            q = int(npq_residual%outwidth)
           
            accum = np.zeros([1])
            for gemm_k in range(0, GEMM_K):
                k = gemm_j
                
                c = int(gemm_k/(kdim*kdim))
                crs_residual = int(gemm_k%(kdim*kdim))
                
                r = int(crs_residual/kdim)
                s = int(crs_residual%kdim)
                
                h = p + kdim - r -1
                w = q + kdim - s - 1

                a = Input[n, h, w, c]
                b = kernel[k, r, s, c]
                accum[0] = accum[0] + a*b

            Output[ n, p, q, k] = accum[0]


def dace_implicit_gemm_conv3d( Input, kernel, Output ):
    GEMM_M = batchsize*outdepth*outheight*outwidth
    GEMM_N = outchannels
    GEMM_K = inchannels * kdim * kdim * kdim

    for gemm_i in range(0,GEMM_M):
        for gemm_j in range(0,GEMM_N):
            print(f'Computing: {gemm_i}, {gemm_j}')
            n = int(gemm_i/(outdepth*outheight*outwidth))
            nopq_residual = int(gemm_i % (outdepth*outheight*outwidth))
            
            o = int(nopq_residual/(outheight*outwidth))
            opq_residual = int(nopq_residual%(outheight*outwidth))
            
            p = int(opq_residual/outwidth)
            q = int(opq_residual%outwidth)
           
            accum = np.zeros([1])
            for gemm_k in range(0, GEMM_K):
                k = gemm_j
                
                c = int(gemm_k/(kdim*kdim*kdim))
                ctrs_residual = int(gemm_k%(kdim*kdim*kdim))

                t = int(ctrs_residual/(kdim*kdim))
                trs_residual = int(ctrs_residual%(kdim*kdim))

                r = int(trs_residual/kdim)
                s = int(trs_residual%kdim)
                
                d = o + kdim - t - 1 
                h = p + kdim - r - 1
                w = q + kdim - s - 1

                a = Input[n, d, h, w, c]
                b = kernel[k, t, r, s, c]
                accum[0] = accum[0] + a*b

            Output[ n, o, p, q, k] = accum[0]


#Input = torch.rand(batchsize, inheight, inwidth, inchannels).cuda()
#kernel = torch.rand(outchannels, kdim, kdim, inchannels).cuda()
#Output = torch.zeros(batchsize, outheight, outwidth, outchannels).cuda()
#dace_implicit_gemm_conv2d(Input, kernel, Output)

Input = torch.rand(batchsize, indepth, inheight, inwidth, inchannels).cuda()
kernel = torch.rand(outchannels, kdim, kdim, kdim, inchannels).cuda()
Output = torch.zeros(batchsize, outdepth, outheight, outwidth, outchannels).cuda()
# dace_implicit_gemm_conv3d(Input, kernel, Output)
# print(Output)

Input = torch.rand(batchsize, indepth, inheight, inwidth, inchannels).cuda()
kernel = torch.rand(outchannels, kdim, kdim, kdim, inchannels).cuda()
Output = torch.zeros(batchsize, outdepth, outheight, outwidth, outchannels).cuda()

# 3D convolution code for verification
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
import libcudnn, ctypes

## cudnn fixed parameters init
cudnn_context = libcudnn.cudnnCreate()
tensor_format = libcudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW']
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

dims = [batchsize, inchannels, indepth, inheight, inwidth]
strides = [inchannels*indepth*inheight*inwidth, indepth*inheight*inwidth, inheight*inwidth, inwidth, 1]
in_desc = libcudnn.cudnnCreateTensorDescriptor()
libcudnn.cudnnSetTensorNdDescriptor(in_desc, data_type, tensor_dim, dims, strides)
# TODO: Maybe simplify this conversion to gpuarray ??
cudnn_input = Input.detach().clone()
cudnn_kernel = kernel.detach().clone()
cudnn_output = Output.detach().clone()
in_data_g = gpuarray.to_gpu(cudnn_input.cpu().numpy().astype(np.float32))
in_data = ctypes.c_void_p(int(in_data_g.gpudata))

# cudnn filter input
filt_desc = libcudnn.cudnnCreateFilterDescriptor()
filt_dims = [outchannels, inchannels, kdim, kdim, kdim]
libcudnn.cudnnSetFilterNdDescriptor(filt_desc, data_type, tensor_format, tensor_dim, filt_dims)
filt_data_g = gpuarray.to_gpu(cudnn_kernel.cpu().numpy().astype(np.float32))                                    
filt_data = ctypes.c_void_p(int(filt_data_g.gpudata))

# cudnn output
outdims = libcudnn.cudnnGetConvolutionNdForwardOutputDim(conv_desc, in_desc, filt_desc, tensor_dim, outdimsinit)
out_n, out_c, out_d, out_h, out_w = outdims[0], outdims[1], outdims[2], outdims[3], outdims[4]
outstrides = [ out_c*out_d*out_h*out_w, out_d*out_h*out_w, out_h*out_w, out_w, 1]
out_desc = libcudnn.cudnnCreateTensorDescriptor()
libcudnn.cudnnSetTensorNdDescriptor(out_desc, data_type, tensor_dim, outdims, outstrides)
out_data_g = gpuarray.to_gpu(cudnn_output.cpu().numpy().astype(np.float32))                               
out_data = ctypes.c_void_p(int(out_data_g.gpudata))
# Compute cudnn workspace size
ws_size = libcudnn.cudnnGetConvolutionForwardWorkspaceSize(cudnn_context, in_desc, filt_desc, conv_desc, out_desc, convolution_algo)
ws_ptr  = drv.mem_alloc(ws_size.value) if ws_size.value > 0 else 0
ws_data = ctypes.c_void_p(int(ws_ptr))

libcudnn.cudnnConvolutionForward(cudnn_context, alpha, in_desc, in_data, filt_desc, filt_data, 
                                conv_desc, convolution_algo, ws_data, ws_size.value, 
                                beta, out_desc, out_data)
