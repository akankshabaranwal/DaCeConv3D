from soaptest import soap_analysis
import dace
import numpy as np
from dace import dtypes
import torch


def implicit_gemm_conv3d( Input, kernel, Output ):
    GEMM_M = batchsize * outdepth * outheight * outwidth
    GEMM_N = outchannels
    GEMM_K = inchannels * kdim * kdim * kdim

    for gemm_i in range(0,GEMM_M):
        for gemm_j in range(0,GEMM_N):
            #print(f'Implicit gemm computing: {gemm_i}, {gemm_j}')
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
                
                d = o + t
                h = p + r
                w = q + s

                a = Input[n, d, h, w, c]
                b = kernel[k, t, r, s, c]
                accum[0] = accum[0] + a*b

            Output[ n, o, p, q, k] = accum[0]


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
outchannels = 8
#TODO: Fix the error when the input dimensions are large
kdim = 3
outdepth = indepth - kdim + 1
outheight = inheight - kdim + 1
outwidth = inheight - kdim + 1
batchsize = 4
dace.Config.set('compiler', 'default_data_types', value='C')

@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=True)
def dace_implicit_gemm_conv3d(Input: dtype[d_batchsize, d_inchannels, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1] @dace.StorageType.GPU_Global ,
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
            accum = accum + a*b

        Output[ n, o, p, q, k] = accum

dace_input = torch.rand(batchsize, indepth, inheight, inwidth, inchannels).cuda()
dace_kernel = torch.rand(outchannels, kdim, kdim, kdim, inchannels).cuda()
dace_output = torch.zeros(batchsize, outdepth, outheight, outwidth, outchannels).cuda()
imgemm_input = dace_input.clone()
imgemm_kernel = dace_kernel.clone()
imgemm_output = dace_output.clone()

#TODO: Figure out why this doesn't work. Like without converting to sdfg, the call to dace implicit gemm convolution fails.
#dace_implicit_gemm_conv3d(Input=dace_input, kernel=dace_kernel, Output=dace_output)
sdfg_fun_conv3d: dace.SDFG = dace_implicit_gemm_conv3d.to_sdfg(dace_input, dace_kernel, dace_output)
sdfg_fun_conv3d.apply_gpu_transformations()
sdfg_fun_conv3d(Input=dace_input, kernel=dace_kernel, Output=dace_output, 
             d_inchannels = inchannels, d_batchsize = batchsize, d_outchannels = outchannels,
             d_outdepth = outdepth, d_outheight = outheight, d_outwidth = outwidth, 
             d_kdim = kdim)

implicit_gemm_conv3d(imgemm_input, imgemm_kernel, imgemm_output)
diff = np.linalg.norm(imgemm_output.cpu() - dace_output.cpu()) / (batchsize * outchannels * indepth * inheight * inwidth )
print('Difference between implicit gemm and dace converted values:', diff)