# For iteratively tiling direct convolution implementation.
# Maybe outchannels, inchannels should be in the outermost and you tile along these dimensions ? 
# Take care of the layout. 

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
batchsize = 4
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
# Tiles of size NTile by OutTile by OCTile
# Merge d, h, w indexing to just 1 loop ? 
CTAtileN = 4
CTAtileDHW = 4 # This should divide outdepth, outheight, outwidth individually otherwise the indexing gets tricky.
CTAtileOC = 8

WARPtileN = 2
WARPtileOC = 4
WARPtileDHWIC = 8
DHW = outdepth*outheight*outwidth
HW = outheight*outwidth
CTAtileDHWIC = CTAtileDHW*inchannels

# Tiling direct convolution
def direct_conv3d_onlytiled(direct_input, direct_kernel, direct_output):
    # Hint:
    # dace.ndarray([], dace.float32, storage=dace.StorageType.)
    # for cta_n, dhw, oc in dace.map[0:batchsize:CTAtile_N, ...] @ dace.ScheduleType.GPU_Device:
    #  Whatever you allocate here goes to the shared memory
    #     for warp_n, woc, ... in dace.map[0:CTAtile_N] @ dace.ScheduleType.GPU_ThreadBlock:
    #       Whatever you allocate here goes to the register memory

    for cta_n in range(0, batchsize, CTAtileN): # Can be distributed
        for cta_dhw in range(0, DHW, CTAtileDHW): # Can be distributed
            for cta_oc in range(0, outchannels, CTAtileOC): # Can be distributed
                cta_reducedk = torch.zeros(CTAtileN, CTAtileDHW, CTAtileOC).cuda()
                # Work allocated for a block
                for warp_n in range(0, CTAtileN, WARPtileN):
                    for warp_oc in range(0, CTAtileOC, WARPtileOC):
                        warp_reducedk = torch.zeros(WARPtileN, WARPtileOC).cuda()
                        for warp_dhwic in range(0, CTAtileDHWIC, WARPtileDHWIC):                            
                            for n in range(0, WARPtileN):
                                for oc in range(0, WARPtileOC):
                                        for dhw_ic in range(0, WARPtileDHWIC): 
                                            
                                            dhw, ic = divmod(dhw_ic+warp_dhwic, inchannels)
                                            d, dhw_residual = divmod(dhw+cta_dhw, HW)
                                            h, w = divmod(dhw_residual, outheight)
                                            for kd in range(0, kdim):
                                                for kh in range(0, kdim):
                                                    for kw in range(0, kdim):
                                                        direct_output[n+cta_n+warp_n, d, h, w, oc+cta_oc+warp_oc] = direct_output[n+cta_n+warp_n, d, h, w, oc+cta_oc+warp_oc] + direct_input[ n+cta_n+warp_n, d+kd, h+kh, w+kw, ic]*direct_kernel[oc+cta_oc+warp_oc, kd, kh, kw, ic]


CTAtileN = 1
CTAtileDHW = 4 # This should divide outdepth, outheight, outwidth individually otherwise the indexing gets tricky.
CTAtileOC = 1
def dace_conv3d( Input, kernel, Output):
    
    d_batchsize = batchsize
    d_outdepth = outdepth
    d_outheight = outheight
    d_outwidth = outwidth
    d_outchannels = outchannels
    d_inchannels = inchannels
    d_kdim = kdim
    d_DHW = d_outdepth*d_outheight*d_outwidth
    d_HW = d_outheight*d_outwidth

    for cta_n, cta_dhw, cta_oc in dace.map[0:d_batchsize:CTAtileN, 0:d_DHW:CTAtileDHW, 0:d_outchannels:CTAtileOC]:
        cta_shared = torch.zeros(CTAtileN, CTAtileDHW, CTAtileOC).cuda()
        cta_shared[:] = 0
        for n, dhw, oc in dace.map[0:CTAtileN, 0:CTAtileDHW, 0:CTAtileOC]:
            d, dhw_residual = dace.int32((dhw+cta_dhw)/d_HW), dace.int32((dhw+cta_dhw)%d_HW)
            h, w = dace.int32(dhw_residual/d_outheight), dace.int32(dhw_residual%d_outheight)
            for ic, kd, kh, kw in dace.map[0:d_inchannels, 0:d_kdim, 0:d_kdim, 0:d_kdim]:
                cta_shared[n, dhw, oc] = cta_shared[n, dhw, oc] + Input[ n+cta_n, d+kd, h+kh, w+kw, ic]*kernel[oc+cta_oc, kd, kh, kw, ic]
        for n, dhw, oc in dace.map[0:CTAtileN, 0:CTAtileDHW, 0:CTAtileOC]:
            d, dhw_residual = dace.int32((dhw+cta_dhw)/d_HW), dace.int32((dhw+cta_dhw)%d_HW)
            h, w = dace.int32(dhw_residual/d_outheight), dace.int32(dhw_residual%d_outheight)                         
            Output[n+cta_n, d, h, w, oc+cta_oc] = cta_shared[n, dhw, oc]

layout = 'NDHWC'
direct_input = torch.rand(batchsize, indepth, inheight, inwidth, inchannels).cuda()
direct_kernel = torch.rand(outchannels, kdim, kdim, kdim, inchannels).cuda()
direct_output = torch.zeros(batchsize, outdepth, outheight, outwidth, outchannels).cuda()

cudnn_input, cudnn_kernel, cudnn_output, in_desc, in_data, in_data_g, out_desc, out_data, out_data_g, outdims,  filt_desc, filt_data, filt_data_g, ws_ptr, ws_data, ws_size = cudnnsetlayerdesc(cudnn_context, outdimsinit, conv_desc, convolution_algo, direct_input,  direct_kernel, direct_output, batchsize, kdim, inchannels, indepth, inheight, inwidth, outchannels, data_type, tensor_dim, tensor_format)
libcudnn.cudnnConvolutionForward(cudnn_context, alpha, in_desc, in_data, filt_desc, filt_data, 
                            conv_desc, convolution_algo, ws_data, ws_size.value, 
                            beta, out_desc, out_data)

#direct_output_g = gpuarray.to_gpu(direct_output.cpu().numpy().astype(np.float32))
#diff = np.linalg.norm((direct_output_g - out_data_g).get()) / (batchsize * outchannels * outdepth * outheight * outwidth )
#print('Earlier difference between cudnn and direct conv values:', diff)

dace_conv3d(direct_input, direct_kernel, direct_output)
direct_output_g = gpuarray.to_gpu(direct_output.cpu().numpy().astype(np.float32))
diff = np.linalg.norm((direct_output_g - out_data_g).get()) / (batchsize * outchannels * outdepth * outheight * outwidth )
print('Difference between cudnn and direct conv values:', diff)
in_desc, out_desc, filt_desc, ws_ptr = destroydescinoutfilt(in_desc, out_desc, filt_desc, ws_ptr)