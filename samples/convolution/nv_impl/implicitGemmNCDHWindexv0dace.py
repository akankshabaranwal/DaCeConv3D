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


# Distribute computation along GEMM_M, GEMM_N. For all other
# CTAtileM = 64
# CTAtileN = 64
# CTAtileK = 8
# WARPtileM = 4
# WARPtileN = 8

CTAtileM = 64
CTAtileN = 2

CTAtileK = 1

WARPtileM = 2
WARPtileN = 1

# Best perf is with below for the first layer
'''
CTAtileM =  256
CTAtileN = 2
CTAtileK = 1

WARPtileM = 4
WARPtileN = 1
WARPtileK = 1
'''

# Best perf is with below for the second layer onwards
'''
CTAtileM =  128
CTAtileN = 32
CTAtileK = 4

WARPtileM = 2
WARPtileN = 16
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
#assert((CTAtileM * CTAtileN + CTAtileM * CTAtileK + CTAtileN*CTAtileK)*4 < 81920)
# Assertion for thread size
#assert(np.int32((CTAtileM/WARPtileM)*(CTAtileN/WARPtileN)*(CTAtileK/WARPtileK))<1024)

# Improve the index computation stuff. Change division to multiplication.
# Change the modulo to something else.
# Change the indices so that the smallest ones like r, s etc. depend on threadIdx.
@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=False)
def dace_conv3d(Input: dtype[d_batchsize, d_inchannels, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1] @dace.StorageType.GPU_Global,
                kernel: dtype[d_outchannels, d_inchannels, d_kdim, d_kdim, d_kdim] @dace.StorageType.GPU_Global,
                Output: dtype[d_batchsize, d_outchannels, d_outdepth, d_outheight, d_outwidth] @dace.StorageType.GPU_Global):
    
    d_GEMM_M = (d_batchsize*d_outdepth*d_outheight*d_outwidth)
    d_GEMM_N = d_outchannels
    d_DHW = d_outdepth*d_outheight*d_outwidth
    d_HW = d_outheight*d_outwidth

    for cta_n, cta_m in dace.map[0:d_GEMM_N:CTAtileN, 0:d_GEMM_M:CTAtileM] @dace.ScheduleType.GPU_Device:
            cta_reducedk = dace.ndarray([CTAtileN, CTAtileM], dtype=Input.dtype, storage=dace.StorageType.GPU_Shared)
            for warp_n, warp_m in dace.map[0:CTAtileN:WARPtileN, 0:CTAtileM:WARPtileM]@dace.ScheduleType.GPU_ThreadBlock:
                for gemm_n, gemm_m in dace.map[0:WARPtileN, 0:WARPtileM]@dace.ScheduleType.Sequential:
                        cta_reducedk[warp_n+gemm_n, warp_m+gemm_m] = 0
            cta_input = dace.ndarray([CTAtileM, 27], dtype=Input.dtype, storage=dace.StorageType.GPU_Shared)
            cta_kernel = dace.ndarray([CTAtileN, 27], dtype=Input.dtype, storage=dace.StorageType.GPU_Shared)
            for cta_c in dace.map[0:d_inchannels]@dace.ScheduleType.Sequential:
                for warp_n, warp_m in dace.map[0:CTAtileN:WARPtileN, 0:CTAtileM:WARPtileM]@dace.ScheduleType.GPU_ThreadBlock:
                        for gemm_n, gemm_m in dace.map[0:WARPtileN, 0:WARPtileM]@dace.ScheduleType.Sequential:
                            n =  dace.int32((gemm_m+cta_m+warp_m)/d_DHW)
                            nopq_residual =  dace.int32((gemm_m+cta_m+warp_m) % d_DHW)

                            o = dace.int32(nopq_residual/d_HW)
                            opq_residual = dace.int32(nopq_residual%d_HW)
                            
                            p = dace.int32(opq_residual/d_outwidth)
                            q = dace.int32(opq_residual%d_outwidth)

                            for t in range(0, 3):
                                for r in range(0, 3):
                                    for s in range(0, 3):
                                        d = o + t
                                        h = p + r
                                        w = q + s

                                        cta_input[ warp_m + gemm_m, s*9+3*r+t] = Input[n, cta_c, d, h, w]
                                        cta_kernel[warp_n + gemm_n, s*9+3*r+t] = kernel[gemm_n+cta_n+warp_n, cta_c, t, r, s]

                for warp_n, warp_m in dace.map[0:CTAtileN:WARPtileN, 0:CTAtileM:WARPtileM]@dace.ScheduleType.GPU_ThreadBlock:
                        warp_input = dace.ndarray([WARPtileM, 27], dtype=Input.dtype, storage=dace.StorageType.Register)
                        warp_kernel = dace.ndarray([WARPtileN, 27], dtype=Input.dtype, storage=dace.StorageType.Register)
                        for gemm_n, gemm_m in dace.map[0:WARPtileN, 0:WARPtileM]@dace.ScheduleType.Sequential:
                            for  t in range(0, 3):
                                 for r in range(0, 3):
                                      for s in range(0, 3):
                                        warp_input[gemm_m, s*9+3*r+t] = cta_input[warp_m + gemm_m, s*9+3*r+t]
                                        warp_kernel[gemm_n, s*9+3*r+t] = cta_kernel[warp_n + gemm_n, s*9+3*r+t]

                        for gemm_n, gemm_m in dace.map[0:WARPtileN, 0:WARPtileM]@dace.ScheduleType.Sequential:
                                for t in range(0, 3):
                                     for r in range(0, 3):
                                          for s in range(0, 3):
                                            cta_reducedk[gemm_n+warp_n, warp_m+gemm_m] = cta_reducedk[gemm_n+warp_n, warp_m+gemm_m] + warp_input[gemm_m, s*9+3*r+t]*warp_kernel[gemm_n, s*9+3*r+t]

            for warp_n, warp_m in dace.map[0: CTAtileN:WARPtileN, 0: CTAtileM:WARPtileM]@dace.ScheduleType.GPU_ThreadBlock:
                for gemm_n, gemm_m in dace.map[0:WARPtileN, 0:WARPtileM]@dace.ScheduleType.Sequential:

                    n =  dace.int32((gemm_m+cta_m+warp_m)/d_DHW)
                    nopq_residual = dace.int32((cta_m+gemm_m+warp_m) % d_DHW)
                    
                    o = dace.int32(nopq_residual/d_HW)
                    opq_residual = dace.int32(nopq_residual%d_HW)        
                    
                    p = dace.int32(opq_residual/d_outwidth)
                    q = dace.int32(opq_residual%d_outwidth)

                    Output[ n, cta_n+gemm_n+warp_n, o, p, q ] = cta_reducedk[gemm_n+warp_n, gemm_m+warp_m]