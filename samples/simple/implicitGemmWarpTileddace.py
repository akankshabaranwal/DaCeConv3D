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
    tmp_i = find_map_by_param(sdfg, '__i0')
    tmp_i.map.schedule = dace.ScheduleType.Sequential
    
    return


# Distribute computation along GEMM_M, GEMM_N 
CTAtileM =  16
CTAtileN = 4
CTAtileK = 2 # Does not effect the parallel part. Keep it 1.

WARPtileM = 2
WARPtileN = 2
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

    for cta_n, cta_m in dace.map[0:d_GEMM_N:CTAtileN, 0:d_GEMM_M:CTAtileM] @dace.ScheduleType.GPU_Device:
            cta_reducedk = dace.ndarray([CTAtileM, CTAtileN], dtype=Input.dtype, storage=dace.StorageType.GPU_Shared)
            cta_reducedk[:] = 0
            for cta_k in range(0, d_GEMM_K, CTAtileK):
                # Load the desired memory to shared memory.
                cta_input = dace.ndarray([CTAtileM, CTAtileK], dtype=Input.dtype, storage=dace.StorageType.GPU_Shared)
                cta_kernel = dace.ndarray([CTAtileK, CTAtileN], dtype=Input.dtype, storage=dace.StorageType.GPU_Shared)
                
                # Load input, kernel to shared memory.
                for thread_n, thread_m, thread_k in dace.map[0: CTAtileN, 0: CTAtileM, 0:CTAtileK]@dace.ScheduleType.GPU_ThreadBlock: # thread parallel
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
                    cta_input[thread_m, thread_k] = Input[n, d, h, w, c]
                    cta_kernel[thread_k, thread_n] = kernel[cta_n+thread_n, t, r, s, c]

                for warp_n, warp_m in dace.map[0: CTAtileN:WARPtileN, 0: CTAtileM:WARPtileM]@dace.ScheduleType.GPU_ThreadBlock:
                        warp_reducedk = dace.ndarray([WARPtileM, WARPtileN], dtype=Input.dtype, storage=dace.StorageType.Register)
                        warp_reducedk[:] = 0
                        for warp_k in range(0, CTAtileK, WARPtileK):
                            for gemm_k in dace.map[0: WARPtileK]@dace.ScheduleType.Sequential:
                                for gemm_m in range(0, WARPtileM):
                                    for gemm_n in range(0, WARPtileN):
                                            warp_reducedk[gemm_m, gemm_n] = warp_reducedk[gemm_m, gemm_n] + cta_input[warp_m+gemm_m, warp_k+gemm_k]*cta_kernel[warp_k+gemm_k, warp_n+gemm_n]
                        for tmp_m in range(0, WARPtileM):
                            for tmp_n in range(0, WARPtileN):
                                cta_reducedk[tmp_m+warp_m, warp_n+tmp_n] = cta_reducedk[tmp_m+warp_m, warp_n+tmp_n] + warp_reducedk[tmp_m, tmp_n]

            for warp_n, warp_m in dace.map[0: CTAtileN:WARPtileN, 0: CTAtileM:WARPtileM]@dace.ScheduleType.GPU_ThreadBlock:
                for assign_n in range(warp_n, WARPtileN+warp_n):
                    for assign_m in range(warp_m, WARPtileM+warp_m):
                        
                        n = dace.int32((cta_m+assign_m)/d_DHW)
                        nopq_residual = dace.int32((cta_m+assign_m) % d_DHW)
                        o = dace.int32(nopq_residual/d_HW)
                        opq_residual = dace.int32(nopq_residual%d_HW)        
                        p = dace.int32(opq_residual/d_outwidth)
                        q =  dace.int32(opq_residual%d_outwidth)
                        Output[ n, o, p, q, cta_n+assign_n] = cta_reducedk[assign_m, assign_n]

                        
# # Use code from here to revert to last working version: https://gitlab.ethz.ch/abaranwal/dacelocal/-/blob/7c78decc3d7dbd3da47576e24bc0c5d051601f65/samples/simple/implicitGemmTileddace.py
# Tiling and buffering along GEMM_M, GEMM_N
#TODO: Check what happens if you change the kdim variables to constants instead of dace map.
@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=False)
def dace_conv3d_new(Input: dtype[d_batchsize, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1, d_inchannels] @dace.StorageType.GPU_Global ,
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
        cta_reducedk = dace.ndarray([CTAtileM, CTAtileN], dtype=Input.dtype, storage=dace.StorageType.GPU_Shared)
        cta_reducedk[:] = 0

        for cta_k in range(0, d_GEMM_K, CTAtileK):
            # Load the required input and filter to shared memory.
            cta_input = dace.ndarray([CTAtileM, CTAtileK], dtype=Input.dtype, storage=dace.StorageType.GPU_Shared)
            cta_kernel = dace.ndarray([CTAtileK, CTAtileN], dtype=Input.dtype, storage=dace.StorageType.GPU_Shared)
            
            # Load input, kernel to shared memory.
            for thread_n, thread_m, thread_k in dace.map[0: CTAtileN, 0: CTAtileM, 0:CTAtileK]@dace.ScheduleType.GPU_ThreadBlock: # thread parallel
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
                cta_input[thread_m, thread_k] = Input[n, d, h, w, c]
                cta_kernel[thread_k, thread_n] = kernel[cta_n+thread_n, t, r, s, c]

            tmpCTA = dace.ndarray([nthread_n, nthread_m, WARPtileM, WARPtileN], dtype=Input.dtype, storage=dace.StorageType.GPU_Shared)
            for thread_x, thread_y in dace.map[0:nthread_n, 0:nthread_m]@dace.ScheduleType.GPU_ThreadBlock:
                for x in range(0, WARPtileM):
                    for y in range(0, WARPtileN):
                        tmpCTA[thread_x, thread_y, x, y] = 0
            for thread_n, thread_m in dace.map[0: CTAtileN: WARPtileN, 0: CTAtileM: WARPtileM]@dace.ScheduleType.GPU_ThreadBlock: # thread parallel
                warp_reducedk = dace.ndarray([WARPtileM, WARPtileN], dtype=Input.dtype, storage=dace.StorageType.Register)
                warp_reducedk[:] = 0               
                for thread_k in range(0, CTAtileK, WARPtileK):
                    for gemm_k in range(0, WARPtileK):
                        for gemm_n in range(0, WARPtileN):
                            for gemm_m in range(0, WARPtileM):
                                warp_reducedk[gemm_m, gemm_n] = warp_reducedk[gemm_m, gemm_n] + cta_input[thread_m+gemm_m, thread_k+gemm_k]*cta_kernel[thread_k+gemm_k, thread_n+gemm_n]
                ithread_n = dace.int32(thread_n/WARPtileN)
                ithread_m = dace.int32(thread_m/WARPtileM)
                for tmp_m in range(0, WARPtileM):
                    for tmp_n in range(0, WARPtileN):
                        tmpCTA[ithread_n, ithread_m, tmp_m, tmp_n] = tmpCTA[ithread_n, ithread_m, tmp_m, tmp_n] + warp_reducedk[tmp_m, tmp_n]
            
            for thread_n, thread_m in dace.map[0:CTAtileN, 0:CTAtileM]@dace.ScheduleType.GPU_ThreadBlock:
                ithread_n = dace.int32(thread_n/WARPtileN)
                tmp_n = dace.int32(thread_n%WARPtileN)
                ithread_m = dace.int32(thread_m/WARPtileM)
                tmp_m = dace.int32(thread_m%WARPtileM)
                cta_reducedk[thread_m, thread_n] = cta_reducedk[thread_m, thread_n] + tmpCTA[ithread_n, ithread_m, tmp_m, tmp_n]
        
        for thread_n, thread_m in dace.map[0: CTAtileN, 0: CTAtileM]@dace.ScheduleType.GPU_ThreadBlock:
            n = dace.int32((cta_m+thread_m)/d_DHW)
            nopq_residual = dace.int32((cta_m+thread_m) % d_DHW)
            o = dace.int32(nopq_residual/d_HW)
            opq_residual = dace.int32(nopq_residual%d_HW)        
            p = dace.int32(opq_residual/d_outwidth)
            q =  dace.int32(opq_residual%d_outwidth)
            Output[ n, o, p, q, cta_n+thread_n] = cta_reducedk[thread_m, thread_n]