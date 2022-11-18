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
indepth = 16
inheight = 16
inwidth = 16
outchannels = 5
kdim = 3
outdepth = indepth - kdim + 1
outheight = inheight - kdim + 1
outwidth = inheight - kdim + 1
batchsize = 3
dace.Config.set('compiler', 'default_data_types', value='C')
inchannels = np.int32(inchannels)
indepth = np.int32(indepth)
inheight = np.int32(inheight)
inwidth = np.int32(inwidth)
outchannels = np.int32(outchannels)
kdim = np.int32(kdim)
outdepth = np.int32(outdepth)
outheight = np.int32(outheight)
outwidth = np.int32(outwidth)

# Direct convolution for verification
def direct_conv3d(direct_input, direct_kernel, direct_output):
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
                                        tmp += direct_input[ n, d+kd, h+kh, w+kw, ic]*direct_kernel[oc, kd, kh, kw, ic]
                        direct_output[n, d, h, w, oc] = tmp


# GEMM_M = d_batchsize * d_outdepth * d_outheight * d_outwidth
# GEMM_N = d_outchannels
# GEMM_K = d_inchannels * d_kdim * d_kdim * d_kdim
def implicit_gemm_conv3d(imgemm_input, imgemm_kernel, imgemm_output):
    tmp_dhw = outdepth*outheight*outwidth
    tmp_hw = outheight*outwidth
    tmp_kdim3 = kdim*kdim*kdim
    tmp_kdim2 = kdim*kdim
    for gemm_i, gemm_j in dace.map[0:batchsize*outdepth*outheight*outwidth, 0:outchannels]:
        n = gemm_i//tmp_dhw
        nopq_residual = gemm_i % tmp_dhw
        
        o = nopq_residual//tmp_hw
        opq_residual = nopq_residual%tmp_hw
        
        p = opq_residual//outwidth
        q = opq_residual%outwidth
        
        accum = np.zeros([1])
        
        for gemm_k in dace.map[0: inchannels * kdim * kdim * kdim]:
            c = gemm_k//tmp_kdim3
            ctrs_residual = gemm_k%tmp_kdim3

            t = ctrs_residual//tmp_kdim2
            trs_residual = ctrs_residual%tmp_kdim2

            r = trs_residual//kdim
            s = trs_residual%kdim
            
            d = o + t
            h = p + r
            w = q + s

            accum[0] = accum[0] + imgemm_input[n, d, h, w, c]*imgemm_kernel[gemm_j, t, r, s, c]

        imgemm_output[ n, o, p, q, gemm_j] = accum[0]


# GEMM_M = d_batchsize * d_outdepth * d_outheight * d_outwidth
# GEMM_N = d_outchannels
# GEMM_K = d_inchannels * d_kdim * d_kdim * d_kdim
@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=False)
def dace_implicit_gemm_conv3d(dace_input: dtype[d_batchsize, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1, d_inchannels] @dace.StorageType.GPU_Global ,
                dace_kernel: dtype[d_outchannels, d_kdim, d_kdim, d_kdim, d_inchannels] @dace.StorageType.GPU_Global,
                dace_output: dtype[d_batchsize, d_outdepth, d_outheight, d_outwidth, d_outchannels] @dace.StorageType.GPU_Global):
    
    for gemm_i, gemm_j in dace.map[0:(d_batchsize*d_outdepth*d_outheight*d_outwidth), 0:d_outchannels]:
        n = dace.int32(gemm_i/( d_outdepth*d_outheight*d_outwidth))
        nopq_residual = dace.int32(gemm_i % (d_outdepth*d_outheight*d_outwidth))
        
        o = dace.int32(nopq_residual/(d_outheight*d_outwidth))
        opq_residual = dace.int32(nopq_residual%(d_outheight*d_outwidth))
        
        p = dace.int32(opq_residual/d_outwidth)
        q = dace.int32(opq_residual%d_outwidth)
        
        accum = np.zeros([1], dtype=dace_input.dtype)
        
        for gemm_k in dace.map[0: (d_inchannels * d_kdim * d_kdim * d_kdim)]:
            c = dace.int32(gemm_k/(d_kdim*d_kdim*d_kdim))
            ctrs_residual = dace.int32(gemm_k%(d_kdim*d_kdim*d_kdim))

            t = dace.int32(ctrs_residual/(d_kdim*d_kdim))
            trs_residual = dace.int32(ctrs_residual%(d_kdim*d_kdim))

            r = dace.int32(trs_residual/d_kdim)
            s = dace.int32(trs_residual%d_kdim)
            
            d = o + t
            h = p + r
            w = q + s

            accum[0] = accum[0] + dace_input[n, d, h, w, c]*dace_kernel[gemm_j, t, r, s, c]

        dace_output[ n, o, p, q, gemm_j] = accum[0]

dace_input = torch.rand(batchsize, indepth, inheight, inwidth, inchannels).cuda()
dace_kernel = torch.rand(outchannels, kdim, kdim, kdim, inchannels).cuda()
dace_output = torch.zeros(batchsize, outdepth, outheight, outwidth, outchannels).cuda()
imgemm_input = dace_input.detach().cpu().numpy()
imgemm_kernel = dace_kernel.detach().cpu().numpy()
imgemm_output = torch.zeros(batchsize, outdepth, outheight, outwidth, outchannels).cuda()
direct_input = dace_input.detach().cpu().numpy()
direct_kernel = dace_kernel.detach().cpu().numpy()
direct_output = torch.zeros(batchsize, outdepth, outheight, outwidth, outchannels).cuda()

# #TODO: Figure out why this doesn't work. Like without converting to sdfg, the call to dace implicit gemm convolution fails.
#dace_implicit_gemm_conv3d(Input=dace_input, kernel=dace_kernel, Output=dace_output)
from dace.sdfg.utils import load_precompiled_sdfg

sdfg_fun_conv3d: dace.SDFG = dace_implicit_gemm_conv3d.to_sdfg(dace_input, dace_kernel, dace_output)
#sdfg_fun_conv3d = load_precompiled_sdfg('./.dacecache/dace_implicit_gemm_conv3d/')
#            # if __CUDA_ARCH__>=200
#                printf("AB: %d \n", __in2);
#            #endif     
#sdfg_fun_conv3d.expand_library_nodes()
#sdfg_fun_conv3d.openmp_sections = False
sdfg_fun_conv3d.apply_gpu_transformations()
sdfg_fun_conv3d(dace_input=dace_input, dace_kernel=dace_kernel, dace_output=dace_output, 
             d_inchannels = inchannels, d_batchsize = batchsize, d_outchannels = outchannels,
             d_outdepth = outdepth, d_outheight = outheight, d_outwidth = outwidth, 
             d_kdim = kdim)
implicit_gemm_conv3d(imgemm_input, imgemm_kernel, imgemm_output)
#print(f'dace output is: {dace_output.cpu()}')
#print(f'imgemm output is: {imgemm_output.cpu()}')
diff = np.linalg.norm(imgemm_output.cpu() - dace_output.cpu()) / (batchsize * outchannels * indepth * inheight * inwidth )
print('Difference between implicit gemm and dace converted values:', diff)

# direct_conv3d(direct_input, direct_kernel, direct_output)
# diff = np.linalg.norm(imgemm_output.cpu() - direct_output.cpu()) / (batchsize * outchannels * outdepth * outheight * outwidth )
# print('Difference between implicit gemm and direct conv values:', diff)
# return_Q_conv3D = soap_analysis(sdfg_fun_conv3d, inchannels, batchsize, kdim, outheight, outwidth, outdepth, outchannels)
# print(f'For implicit gemm 3D convolution leading order terms are: {return_Q_conv3D}')