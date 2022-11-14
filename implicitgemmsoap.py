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
indepth = 5
inheight = 5
inwidth = 5
outchannels = 2
kdim = 3
outdepth = indepth - kdim + 1
outheight = inheight - kdim + 1
outwidth = inheight - kdim + 1
batchsize = 2
dace.Config.set('compiler', 'default_data_types', value='C')

# Direct convolution for verification
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


# GEMM_M = d_batchsize * d_outdepth * d_outheight * d_outwidth
# GEMM_N = d_outchannels
# GEMM_K = d_inchannels * d_kdim * d_kdim * d_kdim
def implicit_gemm_conv3d(Input, kernel, Output):

    for gemm_i, gemm_j in dace.map[0:batchsize*outdepth*outheight*outwidth, 0:outchannels]:
        n = gemm_i//(outdepth*outheight*outwidth)
        nopq_residual = gemm_i % (outdepth*outheight*outwidth)
        
        o = nopq_residual//(outheight*outwidth)
        opq_residual = nopq_residual%(outheight*outwidth)
        
        p = opq_residual//outwidth
        q = opq_residual%outwidth
        
        accum = np.zeros([1])
        k = gemm_j
        
        for gemm_k in dace.map[0: inchannels * kdim * kdim * kdim]:
            c = gemm_k//(kdim*kdim*kdim)
            ctrs_residual = gemm_k%(kdim*kdim*kdim)

            t = ctrs_residual//(kdim*kdim)
            trs_residual = ctrs_residual%(kdim*kdim)

            r = trs_residual//kdim
            s = trs_residual%kdim
            
            d = o + t
            h = p + r
            w = q + s

            a = Input[n, d, h, w, c]
            b = kernel[k, t, r, s, c]
            accum[0] = accum[0] + a*b

        Output[ n, o, p, q, k] = accum[0]


# GEMM_M = d_batchsize * d_outdepth * d_outheight * d_outwidth
# GEMM_N = d_outchannels
# GEMM_K = d_inchannels * d_kdim * d_kdim * d_kdim
@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=True)
def dace_implicit_gemm_conv3d(Input: dtype[d_batchsize, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1, d_inchannels] @dace.StorageType.GPU_Global ,
                kernel: dtype[d_outchannels, d_kdim, d_kdim, d_kdim, d_inchannels] @dace.StorageType.GPU_Global,
                Output: dtype[d_batchsize, d_outdepth, d_outheight, d_outwidth, d_outchannels] @dace.StorageType.GPU_Global):
   
    for gemm_i, gemm_j in dace.map[0:d_batchsize*d_outdepth*d_outheight*d_outwidth, 0:d_outchannels]:
        n = gemm_i//(d_outdepth*d_outheight*d_outwidth)
        nopq_residual = gemm_i % (d_outdepth*d_outheight*d_outwidth)
        
        o = nopq_residual//(d_outheight*d_outwidth)
        opq_residual = nopq_residual%(d_outheight*d_outwidth)
        
        p = opq_residual//d_outwidth
        q = opq_residual%d_outwidth
        
        accum = np.zeros([1], dtype=Input.dtype)
        k = gemm_j
        
        for gemm_k in dace.map[0: d_inchannels * d_kdim * d_kdim * d_kdim]:
            c = gemm_k//(d_kdim*d_kdim*d_kdim)
            ctrs_residual = gemm_k%(d_kdim*d_kdim*d_kdim)

            t = ctrs_residual//(d_kdim*d_kdim)
            trs_residual = ctrs_residual%(d_kdim*d_kdim)

            r = trs_residual//d_kdim
            s = trs_residual%d_kdim
            
            d = o + t
            h = p + r
            w = q + s

            a = Input[n, d, h, w, c]
            b = kernel[k, t, r, s, c]
            accum[0] = accum[0] + a*b

        Output[ n, o, p, q, k] = accum[0]

dace_input = torch.rand(batchsize, indepth, inheight, inwidth, inchannels).cuda()
dace_kernel = torch.rand(outchannels, kdim, kdim, kdim, inchannels).cuda()
dace_output = torch.zeros(batchsize, outdepth, outheight, outwidth, outchannels).cuda()
imgemm_input = dace_input.detach().clone()
imgemm_kernel = dace_kernel.detach().clone()
imgemm_output = torch.zeros(batchsize, outdepth, outheight, outwidth, outchannels).cuda()
direct_input = dace_input.detach().clone()
direct_kernel = dace_kernel.detach().clone()
direct_output = torch.zeros(batchsize, outdepth, outheight, outwidth, outchannels).cuda()

# #TODO: Figure out why this doesn't work. Like without converting to sdfg, the call to dace implicit gemm convolution fails.
#dace_implicit_gemm_conv3d(Input=dace_input, kernel=dace_kernel, Output=dace_output)
sdfg_fun_conv3d: dace.SDFG = dace_implicit_gemm_conv3d.to_sdfg(dace_input, dace_kernel, dace_output)
sdfg_fun_conv3d.apply_gpu_transformations()

sdfg_fun_conv3d(Input=dace_input, kernel=dace_kernel, Output=dace_output, 
             d_inchannels = inchannels, d_batchsize = batchsize, d_outchannels = outchannels,
             d_outdepth = outdepth, d_outheight = outheight, d_outwidth = outwidth, 
             d_kdim = kdim)

# diff = np.linalg.norm(imgemm_output.cpu() - dace_output.cpu()) / (batchsize * outchannels * indepth * inheight * inwidth )
# print('Difference between implicit gemm and dace converted values:', diff)

implicit_gemm_conv3d(imgemm_input, imgemm_kernel, imgemm_output)
direct_conv3d(direct_input, direct_kernel, direct_output)
diff = np.linalg.norm(imgemm_output.cpu() - direct_output.cpu()) / (batchsize * outchannels * outdepth * outheight * outwidth )
print('Difference between implicit gemm and direct conv values:', diff)