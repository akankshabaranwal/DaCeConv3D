import dace
import numpy as np
from dace import dtypes
import torch
from cudnnConv import cudnn_init, cudnnsetlayerdesc, destroydescinoutfilt
import libcudnn
import pycuda.autoinit
from pycuda import gpuarray

inchannels = 4
indepth = 8
inheight = 8
inwidth = 8
outchannels = 16
kdim = 3
outdepth = indepth - kdim + 1
outheight = inheight - kdim + 1
outwidth = inheight - kdim + 1
batchsize = 1
pad = 0
stride = 1
dil = 1
layout = 'NDHWC'

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

# Iteratively tiling the implicit gemm formulation
CTAtileM = 4
CTAtileN = 4
CTAtileK = 4

WARPtileM = 2
WARPtileN = 2
WARPtileK = 2

def onlytiled_implicit_gemm_conv3d(imgemm_input, imgemm_kernel, imgemm_output):
    tmp_dhw = outdepth*outheight*outwidth
    tmp_hw = outheight*outwidth
    tmp_kdim3 = kdim*kdim*kdim
    tmp_kdim2 = kdim*kdim
    GEMM_M = batchsize * outdepth * outheight * outwidth
    GEMM_N = outchannels
    GEMM_K = inchannels * kdim * kdim * kdim

    # 2 levels of tiling
    for cta_n in range(0, GEMM_N, CTAtileN): # Work division between blocks
        for cta_m in range(0, GEMM_M, CTAtileM):            
            for cta_k in range(0, GEMM_K, CTAtileK):

                for warp_n in range(cta_n, CTAtileN+cta_n, WARPtileN): # Work division between threads, per block computation
                    for warp_m in range(cta_m, CTAtileM+cta_m, WARPtileM): 
                        for warp_k in range(cta_k, CTAtileK+cta_k, WARPtileK):

                            for gemm_k in range(warp_k, WARPtileK+warp_k): # Work per thread
                                for gemm_i in range(warp_m, WARPtileM+warp_m):
                                    for gemm_j in range(warp_n, WARPtileN+warp_n):
                                        
                                        n, nopq_residual = divmod(gemm_i+warp_m, tmp_dhw)
                                        o, opq_residual = divmod(nopq_residual, tmp_hw)
                                        p, q = divmod(opq_residual, outwidth)
                                        
                                        c, ctrs_residual = divmod(gemm_k+warp_k, tmp_kdim3)
                                        t, trs_residual = divmod(ctrs_residual, tmp_kdim2)
                                        r, s = divmod(trs_residual, kdim)
                                        d = o + t
                                        h = p + r
                                        w = q + s
                                        imgemm_output[ n, o, p, q, gemm_j]  = imgemm_output[ n, o, p, q, gemm_j] + imgemm_input[n, d, h, w, c]*imgemm_kernel[gemm_j, t, r, s, c]


CTAtileM = 4
CTAtileN = 4
CTAtileK = 4

WARPtileM = 2
WARPtileN = 2
WARPtileK = 2

def tiled_implicit_gemm_conv3d(Input, kernel, Output):
    DHW = outdepth*outheight*outwidth
    HW = outheight*outwidth
    kdim3 = kdim*kdim*kdim
    kdim2 = kdim*kdim

    GEMM_M = batchsize * outdepth * outheight * outwidth
    GEMM_N = outchannels
    GEMM_K = inchannels * kdim * kdim * kdim
    
    for cta_n in range(0, GEMM_N, CTAtileN):
        for cta_m in range(0, GEMM_M, CTAtileM):
            cta_reducedk = torch.zeros(CTAtileM, CTAtileN).cuda()
            for cta_k in range(0, GEMM_K, CTAtileK):
                cta_splitk = torch.zeros(CTAtileM, CTAtileN).cuda()

                for warp_n in range(0, CTAtileN, WARPtileN):
                    for warp_m in range(0, CTAtileM, WARPtileM):
                        warp_reducedk = torch.zeros(WARPtileM, WARPtileN).cuda()
                        for warp_k in range(0, CTAtileK, WARPtileK):
                            warp_splitk = torch.zeros(WARPtileM, WARPtileN).cuda()                          
                            
                            for gemm_k in range(0, WARPtileK):
                                for gemm_m in range(0, WARPtileM):
                                    for gemm_n in range(0, WARPtileN):

                                            n = dace.int32((gemm_m+cta_m+warp_m)/DHW)
                                            nopq_residual = dace.int32((gemm_m+cta_m+warp_m) % DHW)
                                            o = dace.int32(nopq_residual/HW)
                                            opq_residual = dace.int32(nopq_residual%HW)        
                                            p = dace.int32(opq_residual/outwidth)
                                            q = dace.int32(opq_residual%outwidth)

                                            c = dace.int32((gemm_k+cta_k+warp_k)/kdim3)
                                            ctrs_residual = dace.int32((gemm_k+cta_k+warp_k)%kdim3)
                                            t = dace.int32(ctrs_residual/kdim2)
                                            trs_residual = dace.int32(ctrs_residual%kdim2)
                                            r = dace.int32(trs_residual/kdim)
                                            s = dace.int32(trs_residual%kdim)

                                            d = o + t
                                            h = p + r
                                            w = q + s

                                            warp_splitk[gemm_m, gemm_n] = Input[n, d, h, w, c]*kernel[gemm_n+cta_n+warp_n, t, r, s, c]
                                warp_reducedk += warp_splitk
                        cta_splitk[warp_m:warp_m+WARPtileM, warp_n:warp_n+WARPtileN] = warp_reducedk
                cta_reducedk += cta_splitk

            for assignM in range(0, CTAtileM):
                for assignN in range(0, CTAtileN):
                    n = dace.int32((cta_m+assignM)/DHW)
                    nopq_residual = dace.int32((cta_m+assignM) % DHW)
                    o = dace.int32(nopq_residual/HW)
                    opq_residual = dace.int32(nopq_residual%HW)        
                    p = dace.int32(opq_residual/outwidth)
                    q = dace.int32(opq_residual%outwidth)
                    Output[ n, o, p, q, cta_n+assignN] = cta_reducedk[assignM, assignN]
                
    

'''  
    for cta_n in range(0, GEMM_N, CTAtileN): # Work division between blocks
        for cta_m in range(0, GEMM_M, CTAtileM):            
            for cta_k in range(0, GEMM_K, CTAtileK):

                cta_output_buffer = torch.zeros(CTAtileN, CTAtileM, CTAtileK).cuda()
                for warp_n in range(cta_n, CTAtileN+cta_n, WARPtileN): # Work division between threads, per block computation
                    for warp_m in range(cta_m, CTAtileM+cta_m, WARPtileM): 
                        for warp_k in range(cta_k, CTAtileK+cta_k, WARPtileK):

                            for gemm_k in range(warp_k, WARPtileK+warp_k): # Work per thread
                                for gemm_i in range(warp_m, WARPtileM+warp_m):
                                    for gemm_j in range(warp_n, WARPtileN+warp_n):
                                        
                                        n, nopq_residual = divmod(gemm_i+warp_m, tmp_dhw)
                                        o, opq_residual = divmod(nopq_residual, tmp_hw)
                                        p, q = divmod(opq_residual, outwidth)
                                        
                                        c, ctrs_residual = divmod(gemm_k+warp_k, tmp_kdim3)
                                        t, trs_residual = divmod(ctrs_residual, tmp_kdim2)
                                        r, s = divmod(trs_residual, kdim)
                                        d = o + t
                                        h = p + r
                                        w = q + s
                                        #cta_output_buffer[warp_n-cta_n, warp_m-cta_m, warp_k-cta_k] = cta_output_buffer[warp_n-cta_n, warp_m-cta_m, warp_k-cta_k] + imgemm_input[n, d, h, w, c]*imgemm_kernel[gemm_j, t, r, s, c]
                                        imgemm_output[ n, o, p, q, gemm_j]  = imgemm_output[ n, o, p, q, gemm_j] + imgemm_input[n, d, h, w, c]*imgemm_kernel[gemm_j, t, r, s, c]
                
                # The Epilogue part
                # Reduce the cta_output_buffer
                # Assign the appropriate imgemm output to the cta_output_buffer
'''

layout = 'NDHWC'
imgemm_input = torch.rand(batchsize, indepth, inheight, inwidth, inchannels).cuda()
imgemm_kernel = torch.rand(outchannels, kdim, kdim, kdim, inchannels).cuda()
imgemm_output = torch.zeros(batchsize, outdepth, outheight, outwidth, outchannels).cuda()

cudnn_input, cudnn_kernel, cudnn_output, in_desc, in_data, in_data_g, out_desc, out_data, out_data_g, outdims,  filt_desc, filt_data, filt_data_g, ws_ptr, ws_data, ws_size = cudnnsetlayerdesc(cudnn_context, outdimsinit, conv_desc, convolution_algo, imgemm_input,  imgemm_kernel, imgemm_output, batchsize, kdim, inchannels, indepth, inheight, inwidth, outchannels, data_type, tensor_dim, tensor_format)
libcudnn.cudnnConvolutionForward(cudnn_context, alpha, in_desc, in_data, filt_desc, filt_data, 
                            conv_desc, convolution_algo, ws_data, ws_size.value, 
                            beta, out_desc, out_data)
tiled_implicit_gemm_conv3d(imgemm_input, imgemm_kernel, imgemm_output)

imgemm_output_g = gpuarray.to_gpu(imgemm_output.cpu().numpy().astype(np.float32))
diff = np.linalg.norm((imgemm_output_g - out_data_g).get()) / (batchsize * outchannels * outdepth * outheight * outwidth )
print('Difference between cudnn and direct conv values:', diff)
in_desc, out_desc, filt_desc, ws_ptr = destroydescinoutfilt(in_desc, out_desc, filt_desc, ws_ptr)