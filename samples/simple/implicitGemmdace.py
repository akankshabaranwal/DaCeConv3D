import dace
import numpy as np
from dace import dtypes
from dace.transformation.interstate import StateFusion
from dace.transformation.dataflow import TaskletFusion

# Define symbolic sizes for arbitrary inputs
d_outdepth = dace.symbol('d_outdepth')
d_outheight = dace.symbol('d_outheight')
d_outwidth = dace.symbol('d_outwidth')
d_kdim = dace.symbol('d_kdim')
d_inchannels = dace.symbol('d_inchannels')
d_outchannels = dace.symbol('d_outchannels')
d_batchsize = dace.symbol('d_batchsize')

# Define data type to use
dtype = dace.float32
np_dtype = np.float32

def optimize_for_gpu(sdfg: dace.SDFG):
    """ Optimize 3D convolution example for GPUs. """
    dace.Config.set('compiler', 'default_data_types', value='C')
    # Fuse the map and reduce nodes
    # Apply GPU transformation
    sdfg.apply_transformations_repeated(StateFusion)
    sdfg.simplify()
    sdfg.apply_gpu_transformations()
    #sdfg.apply_transformations_repeated(TaskletFusion)
    return

@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=True)
def dace_conv3d_baseline(Input: dtype[d_batchsize, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1, d_inchannels] @dace.StorageType.GPU_Global ,
                kernel: dtype[d_outchannels, d_kdim, d_kdim, d_kdim, d_inchannels] @dace.StorageType.GPU_Global,
                Output: dtype[d_batchsize, d_outdepth, d_outheight, d_outwidth, d_outchannels] @dace.StorageType.GPU_Global):
    
    GEMM_M = (d_batchsize*d_outdepth*d_outheight*d_outwidth)
    GEMM_N = d_outchannels
    GEMM_K = (d_inchannels * d_kdim * d_kdim * d_kdim)
    for gemm_i, gemm_j in dace.map[0:GEMM_M, 0:GEMM_N]:
        n = dace.int32(gemm_i/( d_outdepth*d_outheight*d_outwidth))
        nopq_residual = dace.int32(gemm_i % (d_outdepth*d_outheight*d_outwidth))
        
        o = dace.int32(nopq_residual/(d_outheight*d_outwidth))
        opq_residual = dace.int32(nopq_residual%(d_outheight*d_outwidth))
        
        p = dace.int32(opq_residual/d_outwidth)
        q = dace.int32(opq_residual%d_outwidth)
        
        accum = np.zeros([1], dtype=Input.dtype)
        
        for gemm_k in dace.map[0: GEMM_K]:
            c = dace.int32(gemm_k/(d_kdim*d_kdim*d_kdim))
            ctrs_residual = dace.int32(gemm_k%(d_kdim*d_kdim*d_kdim))

            t = dace.int32(ctrs_residual/(d_kdim*d_kdim))
            trs_residual = dace.int32(ctrs_residual%(d_kdim*d_kdim))

            r = dace.int32(trs_residual/d_kdim)
            s = dace.int32(trs_residual%d_kdim)
            
            d = o + t
            h = p + r
            w = q + s

            accum = accum + Input[n, d, h, w, c]*kernel[gemm_j, t, r, s, c]

        Output[ n, o, p, q, gemm_j] = accum

@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=True)
def dace_conv3d(Input: dtype[d_batchsize, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1, d_inchannels] @dace.StorageType.GPU_Global ,
                kernel: dtype[d_outchannels, d_kdim, d_kdim, d_kdim, d_inchannels] @dace.StorageType.GPU_Global,
                Output: dtype[d_batchsize, d_outdepth, d_outheight, d_outwidth, d_outchannels] @dace.StorageType.GPU_Global):
    
    GEMM_M = (d_batchsize*d_outdepth*d_outheight*d_outwidth)
    GEMM_N = d_outchannels
    GEMM_K = (d_inchannels * d_kdim * d_kdim * d_kdim)

    for gemm_i, gemm_j in dace.map[0:GEMM_M, 0:GEMM_N]:
        accum = np.zeros([1], dtype=Input.dtype)
        n = dace.int32(gemm_i/( d_outdepth*d_outheight*d_outwidth))
        nopq_residual = gemm_i % (d_outdepth*d_outheight*d_outwidth)
        
        o = dace.int32(nopq_residual/(d_outheight*d_outwidth))
        opq_residual = nopq_residual%(d_outheight*d_outwidth)
        
        p = dace.int32(opq_residual/d_outwidth)
        q = opq_residual%d_outwidth

        for gemm_k in dace.map[0: GEMM_K]:
            
            c = dace.int32(gemm_k/(d_kdim*d_kdim*d_kdim))
            ctrs_residual = gemm_k%(d_kdim*d_kdim*d_kdim)

            t = dace.int32(ctrs_residual/(d_kdim*d_kdim))
            trs_residual = ctrs_residual%(d_kdim*d_kdim)

            r = dace.int32(trs_residual/d_kdim)
            s = trs_residual%d_kdim
            
            d = o + t
            h = p + r
            w = q + s

            accum = accum + Input[n, d, h, w, c]*kernel[gemm_j, t, r, s, c]

        Output[ n, o, p, q, gemm_j] = accum


CTATileM = 16
CTATileN = 16
CTATileK = 16

WARPtileM = 4
WARPtileN = 4
WARPtileK = 4

# # Tiling impl;icit gemm implementation
# def dace_tiled_conv3d(Input, kernel, Output, outdepth, outheight, outwidth, outchannels, inchannels, batchsize, kdim):
#     GEMM_M = (batchsize*outdepth*outheight*outwidth)
#     GEMM_N = outchannels
#     GEMM_K = (inchannels * kdim * kdim * kdim)
#     for cta_n in range(0, GEMM_N, CTATileN):
#         for cta_m in range(0, GEMM_M, CTATileM):
#             for cta_k in range(0, GEMM_K, CTATileK):
#                 for warp_n in range(0, CTATileN, WARPtileN):
#                     for warp_m in range(0, CTATileM, WARPtileM):
#                         for warp_k in range(0, CTATileK, WARPtileK):
#                             for mma_k in range(0, WARPtileK):
#                                 for mma_n in range(0, WARPtileN):
#                                     for mma_m in range(0, WARPtileM):

                                