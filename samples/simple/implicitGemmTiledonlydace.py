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
    # # Fuse the map and reduce nodes
    # # Apply GPU transformation
    # tmp_i = find_map_by_param(sdfg, '__i0')
    # tmp_i.map.schedule = dace.ScheduleType.Sequential
    
    return


# Distribute computation along GEMM_M, GEMM_N 
CTAtileM =  32
CTAtileN = 16
CTAtileK = 1 # Does not effect the parallel part. Keep it 1.

WARPtileM = 2
WARPtileN = 4
WARPtileK = 1 # Does not effect the parallel part. Keep it 1.

nthread_n = np.int32(CTAtileN/WARPtileN)
nthread_m = np.int32(CTAtileM/WARPtileM)
# Assertion for shared memory 
assert((CTAtileM * CTAtileN + CTAtileM * CTAtileK + CTAtileN*CTAtileK)*4 < 81920)
# Assertion for thread size
assert(np.int32((CTAtileM/WARPtileM)*(CTAtileN/WARPtileN)*(CTAtileK/WARPtileK))<1024)


@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=False)
def dace_conv3d(Input: dtype[d_batchsize, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1, d_inchannels] @dace.StorageType.GPU_Global ,
                kernel: dtype[d_outchannels, d_kdim, d_kdim, d_kdim, d_inchannels] @dace.StorageType.GPU_Global,
                Output: dtype[d_batchsize, d_outdepth, d_outheight, d_outwidth, d_outchannels] @dace.StorageType.GPU_Global):
    
    d_GEMM_M = (d_batchsize*d_outdepth*d_outheight*d_outwidth)
    d_GEMM_N = d_outchannels
    d_GEMM_K = (d_inchannels * d_kdim * d_kdim * d_kdim)
    d_DHW = d_outdepth*d_outheight*d_outwidth
    d_HW = d_outheight*d_outwidth
    d_kdim3 = d_kdim*d_kdim*d_kdim
    d_kdim2 = d_kdim*d_kdim

    for cta_n, cta_m in dace.map[0:d_GEMM_N:CTAtileN, 0:d_GEMM_M:CTAtileM]@dace.ScheduleType.GPU_Device: # block parallel
        for cta_k in dace.map[0:d_GEMM_K:CTAtileK]@dace.ScheduleType.Sequential:
            for thread_n, thread_m in dace.map[0: CTAtileN, 0: CTAtileM]@dace.ScheduleType.GPU_ThreadBlock: # thread parallel
                for thread_k in dace.map[0:CTAtileK]@dace.ScheduleType.Sequential:
                    n =  dace.int32((cta_m+thread_m)/d_DHW)
                    nopq_residual = dace.int32((cta_m+thread_m) % d_DHW)
                    o = dace.int32(nopq_residual/d_HW)
                    opq_residual = dace.int32(nopq_residual%d_HW)
                    p = dace.int32(opq_residual/d_outwidth)
                    q = dace.int32(opq_residual%d_outwidth)

                    c = dace.int32((cta_k+thread_k)/d_kdim3)
                    ctrs_residual = dace.int32((cta_k+thread_k)%d_kdim3)
                    t = dace.int32(ctrs_residual/d_kdim2)
                    trs_residual = dace.int32(ctrs_residual%d_kdim2)
                    r = dace.int32(trs_residual/d_kdim)
                    s = dace.int32(trs_residual%d_kdim)
                    d, h, w = o + t, p + r, q + s

                    Output[ n, o, p, q, cta_n+thread_n]  =  Output[ n, o, p, q, cta_n+thread_n]  + Input[n, d, h, w, c]*kernel[cta_n+thread_n, t, r, s, c]