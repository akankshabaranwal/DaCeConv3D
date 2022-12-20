import dace
import numpy as np
from dace import dtypes
from dace.transformation.interstate import StateFusion
from dace.transformation.dataflow import TaskletFusion, MapReduceFusion, InLocalStorage
from convutils import find_map_by_param

# Define symbolic sizes for arbitrary inputs
d_outdepth = dace.symbol('d_outdepth')
d_outheight = dace.symbol('d_outheight')
d_outwidth = dace.symbol('d_outwidth')
d_kdim = dace.symbol('d_kdim')
d_inchannels = dace.symbol('d_inchannels')
d_outchannels = dace.symbol('d_outchannels')
d_batchsize = dace.symbol('d_batchsize')

d_GEMM_M = dace.symbol('d_GEMM_M')
d_GEMM_N = dace.symbol('d_GEMM_N')
d_GEMM_K = dace.symbol('d_GEMM_K')

d_DHW = dace.symbol('d_DHW')
d_HW = dace.symbol('d_HW')

d_kdim3 = dace.symbol('d_kdim3')
d_kdim2 = dace.symbol('d_kdim2')

d_ncta_n = dace.symbol('d_ncta_n')
d_ncta_m = dace.symbol('d_ncta_m')
d_ncta_k = dace.symbol('d_ncta_k')

# Define data type to use
dtype = dace.float32
np_dtype = np.float32

def optimize_for_gpu(sdfg: dace.SDFG):
    """ Optimize 3D convolution example for GPUs. """
    print("INFO: Calling optimize for GPU function")
    dace.Config.set('compiler', 'default_data_types', value='C')
    # Fuse the map and reduce nodes
    # Apply GPU transformation
    # tmp_i = find_map_by_param(sdfg, '__i0')
    # tmp_i.map.schedule = dace.ScheduleType.Sequential
    
    return


# Distribute computation along GEMM_M, GEMM_N 
CTAtileM =  32
CTAtileN = 16
CTAtileK = 4

WARPtileM = 4
WARPtileN = 2
WARPtileK = 1

# Best perf is with below for the first layer
'''
CTAtileM =  32
CTAtileN = 16
CTAtileK = 4

WARPtileM = 4
WARPtileN = 2
WARPtileK = 1
'''

# Best perf is with below for the second layer onwards
'''
CTAtileM =  16
CTAtileN = 32
CTAtileK = 4

WARPtileM = 2
WARPtileN = 4
WARPtileK = 1
'''

# Best perf is with below for the third layer onwards
'''
CTAtileM =  8
CTAtileN = 64
CTAtileK = 4

WARPtileM = 2
WARPtileN = 4
WARPtileK = 1
'''

# Assertion for shared memory 
assert((CTAtileM * CTAtileN + CTAtileM * CTAtileK + CTAtileN*CTAtileK)*4 < 81920)
# Assertion for thread size
assert(np.int32((CTAtileM/WARPtileM)*(CTAtileN/WARPtileN)*(CTAtileK/WARPtileK))<1024)

@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=False)
def dace_conv3d(Input: dtype[d_batchsize, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1, d_inchannels] @dace.StorageType.GPU_Global,
                kernel: dtype[d_outchannels, d_kdim, d_kdim, d_kdim, d_inchannels] @dace.StorageType.GPU_Global,
                Output: dtype[d_batchsize, d_outdepth, d_outheight, d_outwidth, d_outchannels] @dace.StorageType.GPU_Global):
    
    d_GEMM_M = (d_batchsize*d_outdepth*d_outheight*d_outwidth)
    d_GEMM_N = d_outchannels
    d_GEMM_K = (d_inchannels * d_kdim * d_kdim * d_kdim)
    d_DHW = d_outdepth*d_outheight*d_outwidth
    d_HW = d_outheight*d_outwidth
    d_kdim3 = d_kdim*d_kdim*d_kdim
    d_kdim2 = d_kdim*d_kdim

    for cta_n, cta_m in dace.map[0:d_GEMM_N:CTAtileN, 0:d_GEMM_M:CTAtileM] @dace.ScheduleType.GPU_Device:
            cta_reducedk = dace.ndarray([CTAtileM, CTAtileN], dtype=Input.dtype, storage=dace.StorageType.GPU_Shared)
            for warp_n, warp_m in dace.map[0:CTAtileN:WARPtileN, 0:CTAtileM:WARPtileM]@dace.ScheduleType.GPU_ThreadBlock:
                for gemm_m, gemm_n in dace.map[0:WARPtileM, 0:WARPtileN]@dace.ScheduleType.Sequential:
                        cta_reducedk[warp_m+gemm_m, warp_n+gemm_n] = 0                               

            for cta_k in dace.map[0:d_GEMM_K:CTAtileK]@dace.ScheduleType.Sequential:
                cta_input = dace.ndarray([CTAtileM, CTAtileK], dtype=Input.dtype, storage=dace.StorageType.GPU_Shared)
                cta_kernel = dace.ndarray([CTAtileK, CTAtileN], dtype=Input.dtype, storage=dace.StorageType.GPU_Shared)
                for warp_n, warp_m in dace.map[0:CTAtileN:WARPtileN, 0:CTAtileM:WARPtileM]@dace.ScheduleType.GPU_ThreadBlock:
                    for warp_k in dace.map[0:CTAtileK:WARPtileK]@dace.ScheduleType.Sequential:
                        for gemm_k, gemm_m, gemm_n in dace.map[0:WARPtileK, 0:WARPtileM, 0:WARPtileN]@dace.ScheduleType.Sequential:
                            n =  dace.int32((gemm_m+cta_m+warp_m)/d_DHW)
                            #n =  dace.int32(dace.float32(gemm_m+cta_m+warp_m)/dace.float32(d_DHW))
                            nopq_residual =  dace.int32((gemm_m+cta_m+warp_m) % d_DHW)

                            o = dace.int32(nopq_residual/d_HW)
                            #o = dace.int32(dace.float32(nopq_residual)/dace.float32(d_HW))
                            opq_residual = dace.int32(nopq_residual%d_HW)
                            
                            p = dace.int32(opq_residual/d_outwidth)
                            #p = dace.int32(dace.float32(opq_residual)/dace.float32(d_outwidth))
                            q = dace.int32(opq_residual%d_outwidth)

                            c  = dace.int32((gemm_k+cta_k+warp_k)/d_kdim3)
                            #c  = dace.int32(dace.float32(gemm_k+cta_k+warp_k)/dace.float32(d_kdim3))
                            ctrs_residual  = dace.int32((gemm_k+cta_k+warp_k)%d_kdim3)
                            
                            t = dace.int32(ctrs_residual/d_kdim2)
                            #t = dace.int32(dace.float32(ctrs_residual)/dace.float32(d_kdim2))
                            trs_residual = dace.int32(ctrs_residual%d_kdim2)
                            
                            r = dace.int32(trs_residual/d_kdim)
                            #r = dace.int32(dace.float32(trs_residual)/dace.float32(d_kdim))
                            s = dace.int32(trs_residual%d_kdim)
                            
                            d = o + t
                            h = p + r
                            w = q + s

                            cta_input[warp_m + gemm_m, warp_k + gemm_k] = Input[n, d, h, w, c]
                            cta_kernel[warp_k + gemm_k, warp_n + gemm_n] = kernel[gemm_n+cta_n+warp_n, t, r, s, c]

                for warp_n, warp_m in dace.map[0:CTAtileN:WARPtileN, 0:CTAtileM:WARPtileM]@dace.ScheduleType.GPU_ThreadBlock:
                        warp_reducedk = dace.ndarray([WARPtileM, WARPtileN], dtype=Input.dtype, storage=dace.StorageType.Register)
                        for gemm_m, gemm_n in dace.map[0:WARPtileM, 0:WARPtileN]@dace.ScheduleType.Sequential:
                                warp_reducedk[gemm_m, gemm_n] = 0
                        for warp_k in dace.map[0:CTAtileK:WARPtileK]@dace.ScheduleType.Sequential:
                            warp_input = dace.ndarray([WARPtileM, WARPtileK], dtype=Input.dtype, storage=dace.StorageType.Register)
                            warp_kernel = dace.ndarray([WARPtileK, WARPtileN], dtype=Input.dtype, storage=dace.StorageType.Register)
                            for gemm_k, gemm_m, gemm_n in dace.map[0:WARPtileK, 0:WARPtileM, 0:WARPtileN]@dace.ScheduleType.Sequential:
                                warp_input[gemm_m, gemm_k] = cta_input[warp_m + gemm_m, warp_k + gemm_k]
                                warp_kernel[gemm_k, gemm_n] = cta_kernel[warp_k + gemm_k, warp_n + gemm_n]

                            for gemm_k, gemm_m, gemm_n in dace.map[0:WARPtileK, 0:WARPtileM, 0:WARPtileN]@dace.ScheduleType.Sequential:
                                warp_reducedk[gemm_m, gemm_n] = warp_reducedk[gemm_m, gemm_n] + warp_input[gemm_m, gemm_k]*warp_kernel[gemm_k, gemm_n]
                                                
                        for tmp_m, tmp_n in dace.map[0:WARPtileM, 0:WARPtileN]@dace.ScheduleType.Sequential:
                            cta_reducedk[tmp_m+warp_m, warp_n+tmp_n] = cta_reducedk[tmp_m+warp_m, warp_n+tmp_n] + warp_reducedk[tmp_m, tmp_n]

            for warp_n, warp_m in dace.map[0: CTAtileN:WARPtileN, 0: CTAtileM:WARPtileM]@dace.ScheduleType.GPU_ThreadBlock:
                for gemm_m, gemm_n in dace.map[0:WARPtileM, 0:WARPtileN]@dace.ScheduleType.Sequential:

                    n =  dace.int32((gemm_m+cta_m+warp_m)/d_DHW)
                    #n = dace.int32(dace.float32((cta_m+gemm_m+warp_m)/dace.float32(d_DHW)))
                    nopq_residual = dace.int32((cta_m+gemm_m+warp_m) % d_DHW)
                    
                    o = dace.int32(nopq_residual/d_HW)
                    #o = dace.int32(dace.float32(nopq_residual)/dace.float32(d_HW))        
                    opq_residual = dace.int32(nopq_residual%d_HW)        
                    
                    p = dace.int32(opq_residual/d_outwidth)
                    #p = dace.int32(dace.float32(opq_residual)/dace.float32(d_outwidth))
                    q = dace.int32(opq_residual%d_outwidth)

                    Output[ n, o, p, q, cta_n+gemm_n+warp_n] = cta_reducedk[gemm_m+warp_m, gemm_n+warp_n]
