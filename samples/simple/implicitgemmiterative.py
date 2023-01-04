import dace
import numpy as np
from dace import dtypes
import torch
from cudnnConv import cudnn_init, cudnnsetlayerdesc, destroydescinoutfilt
import libcudnn
import pycuda.autoinit
from pycuda import gpuarray

# inchannels = 4
# indepth = 10
# inheight = 10
# inwidth = 10
# outchannels = 4
# kdim = 3
# # Iteratively tiling the implicit gemm formulation
# CTAtileM = 4
# CTAtileN = 4
# CTAtileK = 2

# WARPtileM = 2
# WARPtileN = 2
# WARPtileK = 1

# nsplitK = 2

inchannels = 256
indepth = 4
inheight = 4
inwidth = 4
outchannels = 256
kdim = 3
# Iteratively tiling the implicit gemm formulation
CTAtileM = 1
CTAtileN = 32
CTAtileK = 4

WARPtileM = 1
WARPtileN = 16
WARPtileK = 1

nsplitK = 2

outdepth = indepth - kdim + 1
outheight = inheight - kdim + 1
outwidth = inheight - kdim + 1
batchsize = 1


pad = 0
stride = 1
dil = 1
layout = 'NCDHW'

inchannels = np.int32(inchannels)
indepth = np.int32(indepth)
inheight = np.int32(inheight)
inwidth = np.int32(inwidth)
outchannels = np.int32(outchannels)
kdim = np.int32(kdim)
outdepth = np.int32(outdepth)
outheight = np.int32(outheight)
outwidth = np.int32(outwidth)

# Initializing cudnn
conv_desc, cudnn_context, tensor_format, convolution_mode, convolution_algo, alpha, beta, c_int_p, outdimsinit, data_type, tensor_dim, conv_dim = cudnn_init(pad, stride, dil, layout)



def dace_conv3d(Input, kernel, Output):
    d_batchsize = batchsize
    d_outdepth = outdepth
    d_outheight = outheight
    d_outwidth = outwidth
    d_outchannels = outchannels
    d_inchannels = inchannels
    d_kdim = kdim
    d_GEMM_M = (d_batchsize*d_outdepth*d_outheight*d_outwidth)
    d_GEMM_N = d_outchannels
    d_GEMM_K = (d_inchannels * d_kdim * d_kdim * d_kdim)
    d_DHW = d_outdepth*d_outheight*d_outwidth
    d_HW = d_outheight*d_outwidth
    d_kdim3 = d_kdim*d_kdim*d_kdim
    d_kdim2 = d_kdim*d_kdim
    nCTAtileK = dace.int32(d_GEMM_K/CTAtileK)

    splitOutput = torch.zeros(nsplitK, d_batchsize, d_outchannels, d_outdepth, d_outheight, d_outwidth).cuda()
    splitPartK = dace.int32(nCTAtileK/nsplitK)

    for cta_n, cta_m, isplit_k in dace.map[0:d_GEMM_N:CTAtileN, 0:d_GEMM_M:CTAtileM, 0:nsplitK]:
        cta_reducedk = torch.ones(CTAtileN, CTAtileM).cuda()

        for warp_n, warp_m in dace.map[0:CTAtileN:WARPtileN, 0:CTAtileM:WARPtileM]:
            for gemm_n, gemm_m in dace.map[0:WARPtileN, 0:WARPtileM]:
                cta_reducedk[warp_n+gemm_n, warp_m+gemm_m] = 0

        for icta_k in dace.map[isplit_k*splitPartK:(isplit_k+1)*splitPartK]:
            
            cta_input = torch.ones(CTAtileK, CTAtileM).cuda()
            cta_kernel = torch.ones(CTAtileK, CTAtileN).cuda()
            for warp_n, warp_m in dace.map[0:CTAtileN:WARPtileN, 0:CTAtileM:WARPtileM]:
                for warp_k in dace.map[0:CTAtileK:WARPtileK]:
                    for gemm_n, gemm_m, gemm_k in dace.map[0:WARPtileN, 0:WARPtileM, 0:WARPtileK]:
                        n, nopq_residual =  dace.int32((gemm_m+cta_m+warp_m)/d_DHW), dace.int32((gemm_m+cta_m+warp_m) % d_DHW)
                        o, opq_residual = dace.int32(nopq_residual/d_HW), dace.int32(nopq_residual%d_HW)
                        p, q = dace.int32(opq_residual/d_outwidth), dace.int32(opq_residual%d_outwidth)
                        
                        c, ctrs_residual  = dace.int32((gemm_k+(icta_k*CTAtileK)+warp_k)/d_kdim3), dace.int32((gemm_k+(icta_k*CTAtileK)+warp_k)%d_kdim3)
                        t, trs_residual = dace.int32(ctrs_residual/d_kdim2), dace.int32(ctrs_residual%d_kdim2)
                        r, s = dace.int32(trs_residual/d_kdim), dace.int32(trs_residual%d_kdim)
                        d, h, w = o + t, p + r, q + s

                        cta_input[warp_k + gemm_k, warp_m + gemm_m] = Input[n, c, d, h, w]
                        cta_kernel[warp_k + gemm_k, warp_n + gemm_n] = kernel[gemm_n+cta_n+warp_n, c, t, r, s]

            for warp_n, warp_m in dace.map[0:CTAtileN:WARPtileN, 0:CTAtileM:WARPtileM]:
                    warp_reducedk = torch.ones(WARPtileN, WARPtileM).cuda()
                    for gemm_n, gemm_m in dace.map[0:WARPtileN, 0:WARPtileM]:
                            warp_reducedk[gemm_n, gemm_m] = 0
                    for warp_k in dace.map[0:CTAtileK:WARPtileK]:
                        warp_input = torch.ones(WARPtileM, WARPtileK).cuda()
                        warp_kernel = torch.ones(WARPtileK, WARPtileN).cuda()
                        for gemm_k, gemm_n, gemm_m in dace.map[0:WARPtileK, 0:WARPtileN, 0:WARPtileM]:
                            warp_input[gemm_m, gemm_k] = cta_input[warp_k + gemm_k, warp_m + gemm_m]
                            warp_kernel[gemm_k, gemm_n] = cta_kernel[warp_k + gemm_k, warp_n + gemm_n]

                        for gemm_k, gemm_n, gemm_m in dace.map[0:WARPtileK, 0:WARPtileN, 0:WARPtileM]:
                            warp_reducedk[gemm_n, gemm_m] = warp_reducedk[gemm_n, gemm_m] + warp_input[gemm_m, gemm_k]*warp_kernel[gemm_k, gemm_n]
                                    
                    for gemm_n, gemm_m in dace.map[0:WARPtileN, 0:WARPtileM]:
                        cta_reducedk[ gemm_n+warp_n, warp_m+gemm_m] = cta_reducedk[gemm_n+warp_n, warp_m+gemm_m] + warp_reducedk[gemm_n, gemm_m]

            for warp_n, warp_m in dace.map[0: CTAtileN:WARPtileN, 0: CTAtileM:WARPtileM]:
                for gemm_n, gemm_m in dace.map[0:WARPtileN, 0:WARPtileM]:
                    n, nopq_residual = dace.int32((cta_m+gemm_m+warp_m)/d_DHW), dace.int32((cta_m+gemm_m+warp_m) % d_DHW)
                    o, opq_residual = dace.int32(nopq_residual/d_HW), dace.int32(nopq_residual%d_HW)        
                    p, q = dace.int32(opq_residual/d_outwidth), dace.int32(opq_residual%d_outwidth)
                    splitOutput[ isplit_k, n, cta_n+gemm_n+warp_n, o, p, q ] = cta_reducedk[gemm_n+warp_n, gemm_m+warp_m]

    for cta_n, cta_m in dace.map[0:d_GEMM_N:CTAtileN, 0:d_GEMM_M:CTAtileM]:
        for warp_n, warp_m in dace.map[0: CTAtileN:WARPtileN, 0: CTAtileM:WARPtileM]:
            for gemm_n, gemm_m in dace.map[0:WARPtileN, 0:WARPtileM]:
                tmp = 0
                n, nopq_residual = dace.int32((cta_m+gemm_m+warp_m)/d_DHW), dace.int32((cta_m+gemm_m+warp_m) % d_DHW)
                o, opq_residual = dace.int32(nopq_residual/d_HW), dace.int32(nopq_residual%d_HW)        
                p, q = dace.int32(opq_residual/d_outwidth), dace.int32(opq_residual%d_outwidth)
                for isplit_k in dace.map[0:nsplitK]:
                    tmp = tmp + splitOutput[ isplit_k, n, cta_n+gemm_n+warp_n, o, p, q ]
                Output[ n, cta_n+gemm_n+warp_n, o, p, q ] = tmp 

imgemm_input = torch.rand(batchsize, inchannels, indepth, inheight, inwidth).cuda()
imgemm_kernel = torch.rand(outchannels, inchannels, kdim, kdim, kdim).cuda()
imgemm_output = torch.zeros(batchsize, outchannels, outdepth, outheight, outwidth).cuda()

cudnn_input, cudnn_kernel, cudnn_output, in_desc, in_data, in_data_g, out_desc, out_data, out_data_g, outdims,  filt_desc, filt_data, filt_data_g, ws_ptr, ws_data, ws_size = cudnnsetlayerdesc(cudnn_context, outdimsinit, conv_desc, convolution_algo, imgemm_input,  imgemm_kernel, imgemm_output, batchsize, kdim, inchannels, indepth, inheight, inwidth, outchannels, data_type, tensor_dim, tensor_format)
libcudnn.cudnnConvolutionForward(cudnn_context, alpha, in_desc, in_data, filt_desc, filt_data, 
                            conv_desc, convolution_algo, ws_data, ws_size.value, 
                            beta, out_desc, out_data)
dace_conv3d(imgemm_input, imgemm_kernel, imgemm_output)

imgemm_output_g = gpuarray.to_gpu(imgemm_output.cpu().numpy().astype(np.float32))
diff = np.linalg.norm((imgemm_output_g - out_data_g).get()) / (batchsize * outchannels * outdepth * outheight * outwidth )
print('Difference between cudnn and direct conv values:', diff)
in_desc, out_desc, filt_desc, ws_ptr = destroydescinoutfilt(in_desc, out_desc, filt_desc, ws_ptr)
