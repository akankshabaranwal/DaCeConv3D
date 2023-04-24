import numpy as np
import torch
import torch.nn.functional as F
import tensorflow as tf

# 3D convolution code for verification
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
import libcudnn, ctypes

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


def implicit_gemm_conv3d_v1( Input, kernel, Output ):
    GEMM_M = batchsize * outdepth * outheight * outwidth
    GEMM_N = outchannels
    GEMM_K = inchannels * kdim * kdim * kdim
    n=0
    o=0
    p=0
    q=0

    for gemm_i in range(0,GEMM_M):
        if (q==outwidth):
            q = 0
            p = p+1

        if (p == outheight):
            p = 0
            o = o+1
        
        if(o==outdepth):
            o = 0
            n = n+1

        for gemm_j in range(0,GEMM_N):
            print(f'Implicit gemm v1 computing: {gemm_i}, {gemm_j}')

            accum = np.zeros([1])
            k = gemm_j

            c = 0
            t = 0 
            r = 0
            s = 0

            for gemm_k in range(0, GEMM_K):
                
                if(s==kdim):
                    s=0
                    r=r+1
                if(r==kdim):
                    r=0
                    t=t+1
                if(t==kdim):
                    t=0
                    c=c+1
                
                d = o + t
                h = p + r
                w = q + s

                a = Input[n, d, h, w, c]
                b = kernel[k, t, r, s, c]

                accum[0] = accum[0] + a*b
                
                s=s+1
           
            Output[ n, o, p, q, k] = accum[0]

        q = q+1

Input = torch.rand(batchsize, indepth, inheight, inwidth, inchannels).cuda()
kernel = torch.rand(outchannels, kdim, kdim, kdim, inchannels).cuda()
Output = torch.zeros(batchsize, outdepth, outheight, outwidth, outchannels).cuda()

direct_input = Input.detach().clone()
direct_kernel = kernel.detach().clone()
direct_output = Output.detach().clone()

# Verification of implicit gemm code
implicit_gemm_conv3d_v1(Input, kernel, Output)
Output = Output.cpu()
imgemm_output = gpuarray.to_gpu(Output.numpy().astype(np.float32))

direct_conv3d(direct_input, direct_kernel, direct_output)
direct_output_g = gpuarray.to_gpu(direct_output.cpu().numpy().astype(np.float32))
diff = np.linalg.norm(direct_output.cpu() - imgemm_output.get()) / (batchsize * outchannels * indepth * inheight * inwidth )
print('Difference between direct convolution and implicit gemm values:', diff)