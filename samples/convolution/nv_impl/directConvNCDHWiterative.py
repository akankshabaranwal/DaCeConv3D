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
# # Iteratively tiling the implicit gemm formulation
# CTAtileM = 4
# CTAtileN = 4
# CTAtileK = 2

CTAtileDHW = 2 # This should be min [sqrt(Ss), dhw]
CTAtileOC = 2 # This should be min [sqrt(Ss), OC]
CTAtileIC = 2 # The formula for this from soap analysis is 81*d_batchsize*d_inchannels*d_outchannels*d_outdepth*d_outheight*d_outwidth/(Ss*p)
# This is also limited by IC, especially for the first layer.

batchsize = 2
kdim = 3

outdepth = indepth - kdim + 1
outheight = inheight - kdim + 1
outwidth = inheight - kdim + 1

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
    d_DHW = d_outdepth*d_outheight*d_outwidth
    d_HW = d_outheight*d_outwidth

    for  cta_n, cta_oc, cta_dhw in dace.map[0:d_batchsize, 0:d_outchannels:CTAtileOC, 0:d_DHW:CTAtileDHW]:
        cta_output = torch.ones(CTAtileOC, CTAtileDHW).cuda()

        for dhw, oc in dace.map[0:CTAtileDHW, 0:CTAtileOC]:
            cta_output[oc, dhw] = 0

        for cta_ic in dace.map[0:d_inchannels:CTAtileIC]:
            
            cta_input = torch.zeros(CTAtileIC, CTAtileDHW, kdim, kdim, kdim).cuda()
            cta_kernel = torch.zeros(CTAtileOC, CTAtileIC, kdim, kdim, kdim).cuda()         
            
            for dhw, oc in dace.map[0:CTAtileDHW, 0:CTAtileOC]:
                d, dhw_residual = dace.int32((cta_dhw+dhw)/d_HW), dace.int32((cta_dhw+dhw)%d_HW)
                h, w = dace.int32(dhw_residual/d_outheight), dace.int32(dhw_residual%d_outheight)
                for ic, kd, kh, kw in dace.map[0:CTAtileIC, 0:d_kdim, 0:d_kdim, 0:d_kdim]:
                    cta_input[ic, dhw, kd, kh, kw] = Input[ cta_n, ic+cta_ic, d+kd, h+kh, w+kw]
                    cta_kernel[oc, ic, kd, kh, kw] = kernel[cta_oc+oc, ic+cta_ic, kd, kh, kw]

            for dhw, oc in dace.map[0:CTAtileDHW, 0:CTAtileOC]:
                 for ic, kd, kh, kw in dace.map[0:CTAtileIC, 0:d_kdim, 0:d_kdim, 0:d_kdim]:
                     cta_output[oc, dhw] = cta_output[oc, dhw] + cta_input[ic, dhw, kd, kh, kw]*cta_kernel[oc, ic, kd, kh, kw]

        for dhw, oc in dace.map[0:CTAtileDHW, 0:CTAtileOC]:
            d, dhw_residual = dace.int32((cta_dhw+dhw)/d_HW), dace.int32((cta_dhw+dhw)%d_HW)
            h, w = dace.int32(dhw_residual/d_outheight), dace.int32(dhw_residual%d_outheight)
            for ic, kd, kh, kw in dace.map[0:CTAtileIC, 0:d_kdim, 0:d_kdim, 0:d_kdim]:
                Output[cta_n, cta_oc+oc, d, h, w] = cta_output[oc, dhw]

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
