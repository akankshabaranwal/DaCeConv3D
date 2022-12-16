import dace
import numpy as np
from dace import dtypes
import torch
from cudnnConv import cudnn_init, cudnnsetlayerdesc, destroydescinoutfilt
import libcudnn
import pycuda.autoinit
from pycuda import gpuarray

inchannels = 4
indepth = 10
inheight = 10
inwidth = 10
outchannels = 4
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
                    
def dace_conv3d_splitk(Input, kernel, Output):
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

    ncta_n = dace.int32(d_GEMM_N/CTAtileN)
    ncta_m = dace.int32(d_GEMM_M/CTAtileM)
    ncta_k = dace.int32(d_GEMM_K/CTAtileK)
    print(ncta_n, ncta_m, ncta_k)
    commonCTA = torch.zeros(ncta_n, ncta_m, ncta_k, CTAtileM, CTAtileN).cuda()

    for cta_n, cta_m, cta_k in dace.map[0:d_GEMM_N:CTAtileN, 0:d_GEMM_M:CTAtileM, 0:d_GEMM_K:CTAtileK]:
        cta_reducedk = torch.zeros(CTAtileM, CTAtileN).cuda()
        cta_reducedk[:] = 0

        for warp_n, warp_m in dace.map[0: CTAtileN:WARPtileN, 0: CTAtileM:WARPtileM]:
            warp_reducedk = torch.zeros(WARPtileM, WARPtileN).cuda()
            warp_reducedk[:] = 0
            for warp_k in dace.map[0:CTAtileK:WARPtileK]:
                for gemm_k in dace.map[0: WARPtileK]:
                    for gemm_m, gemm_n in dace.map[0:WARPtileM, 0:WARPtileN]:
                        n, nopq_residual =  dace.int32((gemm_m+cta_m+warp_m)/d_DHW), dace.int32((gemm_m+cta_m+warp_m) % d_DHW)
                        o, opq_residual = dace.int32(nopq_residual/d_HW), dace.int32(nopq_residual%d_HW)
                        p, q = dace.int32(opq_residual/d_outwidth), dace.int32(opq_residual%d_outwidth)

                        c, ctrs_residual = dace.int32((gemm_k+cta_k+warp_k)/d_kdim3), dace.int32((gemm_k+cta_k+warp_k)%d_kdim3)
                        t, trs_residual = dace.int32(ctrs_residual/d_kdim2), dace.int32(ctrs_residual%d_kdim2)
                        r, s = dace.int32(trs_residual/d_kdim), dace.int32(trs_residual%d_kdim)
                        d, h, w = o + t, p + r, q + s

                        warp_reducedk[gemm_m, gemm_n] = warp_reducedk[gemm_m, gemm_n] + Input[n, d, h, w, c]*kernel[gemm_n+cta_n+warp_n, t, r, s, c]
            for tmp_m, tmp_n in dace.map[0: WARPtileM, 0:WARPtileN]:
                cta_reducedk[tmp_m+warp_m, warp_n+tmp_n] = cta_reducedk[tmp_m+warp_m, warp_n+tmp_n] + warp_reducedk[tmp_m, tmp_n]
        for warp_n, warp_m in dace.map[0: CTAtileN:WARPtileN, 0: CTAtileM:WARPtileM]:
            for assign_n, assign_m in dace.map[warp_n:WARPtileN+warp_n, warp_m:WARPtileM+warp_m]:
                icta_n = dace.int32(cta_n/CTAtileN)
                icta_m = dace.int32(cta_m/CTAtileM)
                icta_k = dace.int32(cta_k/CTAtileK)
                commonCTA[icta_n, icta_m, icta_k, assign_m, assign_n] = commonCTA[icta_n, icta_m, icta_k, assign_m, assign_n] + cta_reducedk[assign_m, assign_n]

    # Epilogue
    for cta_n, cta_m in dace.map[0:d_GEMM_N:CTAtileN, 0:d_GEMM_M:CTAtileM]:
        cta_newreducedk = torch.zeros(CTAtileM, CTAtileN).cuda()
        cta_newreducedk[:] = 0

        for warp_n, warp_m in dace.map[0: CTAtileN:WARPtileN, 0: CTAtileM:WARPtileM]:
            for cta_k in dace.map[0:d_GEMM_K:CTAtileK]:                        
                for assign_n, assign_m in dace.map[warp_n:WARPtileN+warp_n, warp_m:WARPtileM+warp_m]:
                    icta_n = dace.int32(cta_n/CTAtileN)
                    icta_m = dace.int32(cta_m/CTAtileM)
                    icta_k = dace.int32(cta_k/CTAtileK)
                    cta_newreducedk[assign_m, assign_n] = cta_newreducedk[assign_m, assign_n] + commonCTA[icta_n, icta_m, icta_k, assign_m, assign_n]
        for warp_n, warp_m in dace.map[0: CTAtileN:WARPtileN, 0: CTAtileM:WARPtileM]:
                for assign_n, assign_m in dace.map[warp_n:WARPtileN+warp_n, warp_m:WARPtileM+warp_m]:
                    n, nopq_residual = dace.int32((cta_m+assign_m)/d_DHW), dace.int32((cta_m+assign_m) % d_DHW)
                    o, opq_residual = dace.int32(nopq_residual/d_HW), dace.int32(nopq_residual%d_HW)        
                    p, q = dace.int32(opq_residual/d_outwidth), dace.int32(opq_residual%d_outwidth)
                    Output[ n, o, p, q, cta_n+assign_n] = cta_newreducedk[assign_m, assign_n]



def dace_conv3d_old(Input, kernel, Output):
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
    for cta_n, cta_m in dace.map[0:d_GEMM_N:CTAtileN, 0:d_GEMM_M:CTAtileM]:
            cta_reducedk = torch.zeros(CTAtileM, CTAtileN).cuda()
            for cta_k in range(0, d_GEMM_K, CTAtileK):
                cta_splitk = torch.zeros(CTAtileM, CTAtileN).cuda()
                # Load the required input and filter to shared memory.
                cta_input = torch.zeros(CTAtileM, CTAtileK).cuda()
                cta_kernel = torch.zeros(CTAtileK, CTAtileN).cuda()
                
                # Load input, output to shared memory.
                for warp_n, warp_m, warp_k in dace.map[0: CTAtileN, 0: CTAtileM, 0:CTAtileK]:
                    n =  dace.int32((cta_m+warp_m)/d_DHW)
                    nopq_residual = dace.int32((cta_m+warp_m) % d_DHW)
                    o = dace.int32(nopq_residual/d_HW)
                    opq_residual = dace.int32(nopq_residual%d_HW)
                    p = dace.int32(opq_residual/d_outwidth)
                    q = dace.int32(opq_residual%d_outwidth)

                    c = dace.int32((cta_k+warp_k)/d_kdim3)
                    ctrs_residual = dace.int32((cta_k+warp_k)%d_kdim3)
                    t = dace.int32(ctrs_residual/d_kdim2)
                    trs_residual = dace.int32(ctrs_residual%d_kdim2)
                    r = dace.int32(trs_residual/d_kdim)
                    s = dace.int32(trs_residual%d_kdim)
                    d, h, w = o + t, p + r, q + s
                    cta_input[warp_m, warp_k] = Input[n, d, h, w, c]
                    cta_kernel[warp_k, warp_n] = kernel[cta_n+warp_n, t, r, s, c]

                for warp_n, warp_m in dace.map[0: CTAtileN:WARPtileN, 0: CTAtileM:WARPtileM]:
                    warp_reducedk = torch.zeros(WARPtileM, WARPtileN).cuda()
                    warp_reducedk[:] = 0
                    for warp_k in range(0, CTAtileK, WARPtileK):
                        warp_splitk = torch.zeros(WARPtileM, WARPtileN).cuda()
                        for gemm_k in dace.map[0: WARPtileK]:
                            for gemm_m in range(0, WARPtileM):
                                for gemm_n in range(0, WARPtileN):
                                        warp_splitk[gemm_m, gemm_n] = cta_input[warp_m+gemm_m, warp_k+gemm_k]*cta_kernel[warp_k+gemm_k, warp_n+gemm_n]  
                            warp_reducedk = warp_reducedk + warp_splitk
                    for tmp_m in range(0, WARPtileM):
                        for tmp_n in range(0, WARPtileN):
                            cta_splitk[tmp_m+warp_m, warp_n+tmp_n] = cta_splitk[tmp_m+warp_m, warp_n+tmp_n] + warp_reducedk[tmp_m, tmp_n]
                for warp_n, warp_m in dace.map[0: CTAtileN:WARPtileN, 0: CTAtileM:WARPtileM]:
                    for assign_n in range(warp_n, WARPtileN+warp_n):
                        for assign_m in range(warp_m, WARPtileM+warp_m):
                            cta_reducedk[assign_m, assign_n] = cta_reducedk[assign_m, assign_n] + cta_splitk[assign_m, assign_n]

            for warp_n, warp_m in dace.map[0: CTAtileN:WARPtileN, 0: CTAtileM:WARPtileM]:
                for assign_n in range(warp_n, WARPtileN+warp_n):
                    for assign_m in range(warp_m, WARPtileM+warp_m):
                        
                        n = dace.int32((cta_m+assign_m)/d_DHW)
                        nopq_residual = dace.int32((cta_m+assign_m) % d_DHW)
                        o = dace.int32(nopq_residual/d_HW)
                        opq_residual = dace.int32(nopq_residual%d_HW)        
                        p = dace.int32(opq_residual/d_outwidth)
                        q =  dace.int32(opq_residual%d_outwidth)
                        Output[ n, o, p, q, cta_n+assign_n] = cta_reducedk[assign_m, assign_n]


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

    for cta_n, cta_m in dace.map[0:d_GEMM_N:CTAtileN, 0:d_GEMM_M:CTAtileM]: # block parallel
        cta_reducedk = torch.zeros(CTAtileM, CTAtileN).cuda()
        for cta_k in range(0, d_GEMM_K, CTAtileK):
            # Load the required input and filter to shared memory.
            cta_input = torch.zeros(CTAtileM, CTAtileK).cuda()
            cta_kernel = torch.zeros(CTAtileK, CTAtileN).cuda()
            
            # Load input, kernel to shared memory.
            for thread_n, thread_m, thread_k in dace.map[0: CTAtileN, 0: CTAtileM, 0:CTAtileK]: # thread parallel
                n =  dace.int32((cta_m+thread_m)/d_DHW)
                nopq_residual = dace.int32((cta_m+thread_m) % d_DHW)
                o = dace.int32(nopq_residual/d_HW)
                opq_residual = dace.int32(nopq_residual%d_HW)
                p = dace.int32(opq_residual/d_outwidth)
                q = dace.int32(opq_residual%d_outwidth)

                c = dace.int32((cta_k+thread_k)/d_kdim3)
                ctrs_residual = dace.int32((cta_k+thread_k)%d_kdim3)
                t = dace.int32(ctrs_residual/d_kdim2)
                trs_residual = dace.int32(ctrs_residual%d_kdim2)
                r = dace.int32(trs_residual/d_kdim)
                s = dace.int32(trs_residual%d_kdim)
                d, h, w = o + t, p + r, q + s
                cta_input[thread_m, thread_k] = Input[n, d, h, w, c]
                cta_kernel[thread_k, thread_n] = kernel[cta_n+thread_n, t, r, s, c]

            nthread_n = dace.int32(CTAtileN/WARPtileN)
            nthread_m = dace.int32(CTAtileM/WARPtileM)
            tmpCTA = torch.ones(nthread_n, nthread_m, WARPtileM, WARPtileN).cuda()
            for thread_x, thread_y in dace.map[0:nthread_n, 0:nthread_m]:
                for x in range(0, WARPtileM):
                    for y in range(0, WARPtileN):
                        tmpCTA[thread_x, thread_y, x, y] = 0

            for thread_n, thread_m in dace.map[0: CTAtileN:WARPtileN, 0:CTAtileM:WARPtileM]:
                warp_reducedk = torch.zeros(WARPtileM, WARPtileN).cuda()

                for thread_k in range(0, CTAtileK, WARPtileK):
                    for gemm_k in range(0, WARPtileK):
                        for gemm_n in range(0, WARPtileN):
                            for gemm_m in range(0, WARPtileM):
                                warp_reducedk[gemm_m, gemm_n] = warp_reducedk[gemm_m, gemm_n] + cta_input[thread_m+gemm_m, thread_k+gemm_k]*cta_kernel[thread_k+gemm_k, thread_n+gemm_n]
                ithread_n = dace.int32(thread_n/WARPtileN)
                ithread_m = dace.int32(thread_m/WARPtileM)
                for tmp_m, tmp_n in dace.map[0: WARPtileM, 0:WARPtileN]:
                    tmpCTA[ithread_n, ithread_m, tmp_m, tmp_n] = tmpCTA[ithread_n, ithread_m, tmp_m, tmp_n] + warp_reducedk[tmp_m, tmp_n]
            for thread_n, thread_m in dace.map[0:CTAtileN, 0:CTAtileM]:
                ithread_n = dace.int32(thread_n/WARPtileN)
                tmp_n = dace.int32(thread_n%WARPtileN)
                ithread_m = dace.int32(thread_m/WARPtileM)
                tmp_m = dace.int32(thread_m%WARPtileM)
                
                cta_reducedk[thread_m, thread_n] = cta_reducedk[thread_m, thread_n] + tmpCTA[ithread_n, ithread_m, tmp_m, tmp_n]
        
        for thread_n, thread_m in dace.map[0: CTAtileN, 0: CTAtileM]: # Write to output array
            n = dace.int32((cta_m+thread_m)/d_DHW)
            nopq_residual = dace.int32((cta_m+thread_m) % d_DHW)
            o = dace.int32(nopq_residual/d_HW)
            opq_residual = dace.int32(nopq_residual%d_HW)        
            p = dace.int32(opq_residual/d_outwidth)
            q =  dace.int32(opq_residual%d_outwidth)
            Output[ n, o, p, q, cta_n+thread_n] = cta_reducedk[thread_m, thread_n]
                        
layout = 'NDHWC'
imgemm_input = torch.rand(batchsize, indepth, inheight, inwidth, inchannels).cuda()
imgemm_kernel = torch.rand(outchannels, kdim, kdim, kdim, inchannels).cuda()
imgemm_output = torch.zeros(batchsize, outdepth, outheight, outwidth, outchannels).cuda()

cudnn_input, cudnn_kernel, cudnn_output, in_desc, in_data, in_data_g, out_desc, out_data, out_data_g, outdims,  filt_desc, filt_data, filt_data_g, ws_ptr, ws_data, ws_size = cudnnsetlayerdesc(cudnn_context, outdimsinit, conv_desc, convolution_algo, imgemm_input,  imgemm_kernel, imgemm_output, batchsize, kdim, inchannels, indepth, inheight, inwidth, outchannels, data_type, tensor_dim, tensor_format)
libcudnn.cudnnConvolutionForward(cudnn_context, alpha, in_desc, in_data, filt_desc, filt_data, 
                            conv_desc, convolution_algo, ws_data, ws_size.value, 
                            beta, out_desc, out_data)
dace_conv3d(imgemm_input, imgemm_kernel, imgemm_output)

imgemm_output_g = gpuarray.to_gpu(imgemm_output.cpu().numpy().astype(np.float32))
diff = np.linalg.norm((imgemm_output_g - out_data_g).get()) / (batchsize * outchannels * outdepth * outheight * outwidth )
print('Difference between cudnn and direct conv values:', diff)
in_desc, out_desc, filt_desc, ws_ptr = destroydescinoutfilt(in_desc, out_desc, filt_desc, ws_ptr)