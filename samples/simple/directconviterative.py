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
CTAtile_N = 4
CTAtile_DHW = 4 # This should divide outdepth, outheight, outwidth individually otherwise the indexing gets tricky.
CTAtile_OC = 8

WARPtile_N = 2
WARPtile_OC = 4
WARPtile_DHWIC = 8
DHW = outdepth*outheight*outwidth
HW = outheight*outwidth
CTAtile_DHW_IC = CTAtile_DHW*inchannels

# Tiling direct convolution
def direct_conv3d(direct_input, direct_kernel, direct_output):

    for cta_n in range(0, batchsize, CTAtile_N):
        for cta_dhw in range(0, DHW, CTAtile_DHW):
            for cta_oc in range(0, outchannels, CTAtile_OC):
                
                # Work allocated for a block
                for warp_n in range(0, CTAtile_N, WARPtile_N):
                    for warp_oc in range(0, CTAtile_OC, WARPtile_OC):
                        for warp_dhwic in range(0, CTAtile_DHW_IC, WARPtile_DHWIC):
                            
                            for n in range(0, WARPtile_N):
                                for oc in range(0, WARPtile_OC):
                                        for dhw_ic in range(0, WARPtile_DHWIC): # all must go to the same block, but too much work for a single thread
                                            
                                            dhw, ic = divmod(dhw_ic+warp_dhwic, inchannels)
                                            d, dhw_residual = divmod(dhw+cta_dhw, HW)
                                            h, w = divmod(dhw_residual, outheight)
                                            for kd in range(0, kdim):
                                                for kh in range(0, kdim):
                                                    for kw in range(0, kdim):
                                                            direct_output[n+cta_n+warp_n, d, h, w, oc+cta_oc+warp_oc] = direct_output[n+cta_n+warp_n, d, h, w, oc+cta_oc+warp_oc] + direct_input[ n+cta_n+warp_n, d+kd, h+kh, w+kw, ic]*direct_kernel[oc+cta_oc+warp_oc, kd, kh, kw, ic]

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

direct_conv3d(direct_input, direct_kernel, direct_output)
direct_output_g = gpuarray.to_gpu(direct_output.cpu().numpy().astype(np.float32))
diff = np.linalg.norm((direct_output_g - out_data_g).get()) / (batchsize * outchannels * outdepth * outheight * outwidth )
print('Difference between cudnn and direct conv values:', diff)
in_desc, out_desc, filt_desc, ws_ptr = destroydescinoutfilt(in_desc, out_desc, filt_desc, ws_ptr)