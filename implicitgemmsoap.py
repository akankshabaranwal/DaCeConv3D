from soaptest import soap_analysis
import dace
import numpy as np
from dace import dtypes
import torch

d_outdepth = dace.symbol('d_outdepth')
d_outheight = dace.symbol('d_outheight')
d_outwidth = dace.symbol('d_outwidth')
d_inchannels = dace.symbol('d_inchannels')
d_outchannels = dace.symbol('d_outchannels')
d_batchsize = dace.symbol('d_batchsize')
d_kdim = dace.symbol('d_kdim')

dtype = dace.float32
np_dtype = np.float32

inchannels = 4
indepth = 128
inheight = 128
inwidth = 128
outchannels = 16
kdim = 3
outdepth = indepth - kdim + 1
outheight = inheight - kdim + 1
outwidth = inheight - kdim + 1
batchsize = 4
dace.Config.set('compiler', 'default_data_types', value='C')

@dace.program(device=dtypes.DeviceType.GPU)
def implicit_gemm_conv3d(Input: dtype[d_batchsize, d_inchannels, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1] @dace.StorageType.GPU_Global ,
                kernel: dtype[d_outchannels, d_inchannels, d_kdim, d_kdim, d_kdim] @dace.StorageType.GPU_Global,
                Output: dtype[d_batchsize, d_outchannels, d_outdepth, d_outheight, d_outwidth] @dace.StorageType.GPU_Global):
   
    GEMM_M = d_batchsize * d_outdepth * d_outheight * d_outwidth
    GEMM_N = d_outchannels
    GEMM_K = d_inchannels * d_kdim * d_kdim * d_kdim

    for gemm_i, gemm_j in dace.map[0:GEMM_M, 0:GEMM_N]:
        n = int(gemm_i/(d_outdepth*d_outheight*d_outwidth))
        nopq_residual = int(gemm_i % (d_outdepth*d_outheight*d_outwidth))
        
        o = int(nopq_residual/(d_outheight*d_outwidth))
        opq_residual = int(nopq_residual%(d_outheight*d_outwidth))
        
        p = int(opq_residual/d_outwidth)
        q = int(opq_residual%d_outwidth)
        
        accum = np.zeros([1], dtype=Input.dtype)
        k = gemm_j
        
        for gemm_k in dace.map[0: GEMM_K]:
            c = int(gemm_k/(d_kdim*d_kdim*d_kdim))
            ctrs_residual = int(gemm_k%(d_kdim*d_kdim*d_kdim))

            t = int(ctrs_residual/(d_kdim*d_kdim))
            trs_residual = int(ctrs_residual%(d_kdim*d_kdim))

            r = int(trs_residual/d_kdim)
            s = int(trs_residual%d_kdim)
            
            d = o + t
            h = p + r
            w = q + s

            a = Input[n, d, h, w, c]
            b = kernel[k, t, r, s, c]
            accum[0] = accum[0] + a*b

        Output[ n, o, p, q, k] = accum[0]

d_input_conv3d = torch.rand(batchsize, inchannels, indepth, inheight, inwidth).cuda()
d_kernel_conv3d = torch.rand(outchannels, inchannels, kdim, kdim, kdim).cuda()
d_output_conv3d = torch.zeros(batchsize, outchannels, outdepth, outheight, outwidth).cuda()
t_input = d_input_conv3d.clone()
t_kernel = d_kernel_conv3d.clone()
sdfg_fun_conv3d: dace.SDFG = implicit_gemm_conv3d.to_sdfg(d_input_conv3d, d_kernel_conv3d, d_output_conv3d)
sdfg_fun_conv3d.apply_gpu_transformations()
sdfg_fun_conv3d(Input=d_input_conv3d, kernel=d_kernel_conv3d, Output=d_output_conv3d, 
             d_inchannels = inchannels, d_batchsize = batchsize, d_outchannels = outchannels,
             d_outdepth = outdepth, d_outheight = outheight, d_outwidth = outwidth, 
             d_kdim = kdim)

