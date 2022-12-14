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
CTAtileM =  64
CTAtileN = 2
CTAtileK = 1 # Does not effect the parallel part. Keep it 1.

WARPtileM = 2
WARPtileN = 1
WARPtileK = 1 # Does not effect the parallel part. Keep it 1.

# Assertion for shared memory 
assert((CTAtileM * CTAtileN + CTAtileM * 256 + CTAtileN*256)*4 < 81920)
# Assertion for thread size
assert(np.int32((CTAtileM/WARPtileM)*(CTAtileN/WARPtileN))<1024)


# Tiling and buffering along GEMM_M, GEMM_N
#TODO: Check what happens if you change the kdim variables to constants instead of dace map.
@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=False)
def dace_conv3d_mn(Input: dtype[d_batchsize, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1, d_inchannels] @dace.StorageType.GPU_Global ,
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
            
            for warp_n, warp_m in dace.map[0: CTAtileN:WARPtileN, 0: CTAtileM:WARPtileM]@dace.ScheduleType.GPU_ThreadBlock:
                for cta_k in dace.map[0:d_GEMM_K:CTAtileK]@dace.ScheduleType.Sequential:
                    warp_reducedk = dace.ndarray([WARPtileM, WARPtileN], dtype=Input.dtype, storage=dace.StorageType.Register)
                    warp_reducedk[:] = 0
                    for warp_k in dace.map[0:CTAtileK:WARPtileK]@dace.ScheduleType.Sequential:
                        for gemm_k in dace.map[0: WARPtileK]@dace.ScheduleType.Sequential:
                            for gemm_m, gemm_n in dace.map[0:WARPtileM, 0:WARPtileN]@dace.ScheduleType.Sequential:
                                        n, nopq_residual =  dace.int32((gemm_m+cta_m+warp_m)/d_DHW), dace.int32((gemm_m+cta_m+warp_m) % d_DHW)
                                        o, opq_residual = dace.int32(nopq_residual/d_HW), dace.int32(nopq_residual%d_HW)
                                        p, q = dace.int32(opq_residual/d_outwidth), dace.int32(opq_residual%d_outwidth)

                                        c, ctrs_residual = dace.int32((gemm_k+cta_k+warp_k)/d_kdim3), dace.int32((gemm_k+cta_k+warp_k)%d_kdim3)
                                        t, trs_residual = dace.int32(ctrs_residual/d_kdim2), dace.int32(ctrs_residual%d_kdim2)
                                        r, s = dace.int32(trs_residual/d_kdim), dace.int32(trs_residual%d_kdim)
                                        d, h, w = o + t, p + r, q + s

                                        warp_reducedk[gemm_m, gemm_n] = warp_reducedk[gemm_m, gemm_n] + Input[n, d, h, w, c]*kernel[gemm_n+cta_n+warp_n, t, r, s, c]
                    for tmp_m, tmp_n in dace.map[0: WARPtileM, 0:WARPtileN]:
                            cta_reducedk[tmp_m+warp_m, warp_n+tmp_n] = cta_reducedk[tmp_m+warp_m, warp_n+tmp_n] + warp_reducedk[tmp_m, tmp_n]

            for warp_n, warp_m in dace.map[0: CTAtileN:WARPtileN, 0: CTAtileM:WARPtileM]@dace.ScheduleType.GPU_ThreadBlock:
                    for assign_n, assign_m in dace.map[warp_n:WARPtileN+warp_n, warp_m:WARPtileM+warp_m]@dace.ScheduleType.Sequential:
                        n, nopq_residual = dace.int32((cta_m+assign_m)/d_DHW), dace.int32((cta_m+assign_m) % d_DHW)
                        o, opq_residual = dace.int32(nopq_residual/d_HW), dace.int32(nopq_residual%d_HW)        
                        p, q = dace.int32(opq_residual/d_outwidth), dace.int32(opq_residual%d_outwidth)
                        Output[ n, o, p, q, cta_n+assign_n] = cta_reducedk[assign_m, assign_n]



# CTAtileM/WARPtileM * CTAtileN/WARPtileN * CTAtileK/WARPtileK <= 1024
# For it to fit in the shared memory, CTAtileM and CTAtileN <=64
# Number of blocks launched is GEMM_M/CTA_M * GEMM_N/CTA_N * GEMM_K/CTA_K

CTAtileM =  64
CTAtileN = 64
CTAtileK = 1

WARPtileM = 4
WARPtileN = 4
WARPtileK = 1
# Assertion for shared memory 
assert((CTAtileM * CTAtileN + CTAtileM * CTAtileK + CTAtileN*CTAtileK)*4 < 81920)
# Assertion for thread size
assert(np.int32((CTAtileM/WARPtileM)*(CTAtileN/WARPtileN)*(CTAtileK/WARPtileK))<1024)

# Tiling and buffering
#TODO: Check what happens if you change the kdim variables to constants instead of dace map.
@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=False)
def dace_conv3d_atomic(Input: dtype[d_batchsize, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1, d_inchannels] @dace.StorageType.GPU_Global ,
                kernel: dtype[d_outchannels, d_kdim, d_kdim, d_kdim, d_inchannels] @dace.StorageType.GPU_Global,
                Output: dtype[d_batchsize, d_outdepth, d_outheight, d_outwidth, d_outchannels] @dace.StorageType.GPU_Global):
    
    d_GEMM_M = (d_batchsize*d_outdepth*d_outheight*d_outwidth)
    d_GEMM_N = d_outchannels
    d_GEMM_K = (d_inchannels * d_kdim * d_kdim * d_kdim)
    d_DHW = d_outdepth*d_outheight*d_outwidth
    d_HW = d_outheight*d_outwidth
    d_kdim3 = d_kdim*d_kdim*d_kdim
    d_kdim2 = d_kdim*d_kdim
    
    for cta_n, cta_m, cta_k in dace.map[0:d_GEMM_N:CTAtileN, 0:d_GEMM_M:CTAtileM, 0:d_GEMM_K:CTAtileK] @dace.ScheduleType.GPU_Device:
        cta_reducedk = dace.ndarray([CTAtileM, CTAtileN], dtype=Input.dtype, storage=dace.StorageType.GPU_Shared)
        cta_reducedk[:] = 0
        for warp_n, warp_m in dace.map[0: CTAtileN:WARPtileN, 0: CTAtileM:WARPtileM]@dace.ScheduleType.GPU_ThreadBlock:
            for assign_n, assign_m in dace.map[warp_n:WARPtileN+warp_n, warp_m:WARPtileM+warp_m]@dace.ScheduleType.Sequential:
                n, nopq_residual = dace.int32((cta_m+assign_m)/d_DHW), dace.int32((cta_m+assign_m) % d_DHW)
                o, opq_residual = dace.int32(nopq_residual/d_HW), dace.int32(nopq_residual%d_HW)        
                p, q = dace.int32(opq_residual/d_outwidth), dace.int32(opq_residual%d_outwidth)
                cta_reducedk[assign_m, assign_n] = Output[ n, o, p, q, cta_n+assign_n]

        for warp_n, warp_m in dace.map[0: CTAtileN:WARPtileN, 0: CTAtileM:WARPtileM]@dace.ScheduleType.GPU_ThreadBlock:
            warp_reducedk = dace.ndarray([WARPtileM, WARPtileN], dtype=Input.dtype, storage=dace.StorageType.Register)
            warp_reducedk[:] = 0
            for warp_k in dace.map[0:CTAtileK:WARPtileK]@dace.ScheduleType.Sequential:
                for gemm_k in dace.map[0: WARPtileK]@dace.ScheduleType.Sequential:
                    for gemm_m, gemm_n in dace.map[0:WARPtileM, 0:WARPtileN]@dace.ScheduleType.Sequential:
                        n, nopq_residual =  dace.int32((gemm_m+cta_m+warp_m)/d_DHW), dace.int32((gemm_m+cta_m+warp_m) % d_DHW)
                        o, opq_residual = dace.int32(nopq_residual/d_HW), dace.int32(nopq_residual%d_HW)
                        p, q = dace.int32(opq_residual/d_outwidth), dace.int32(opq_residual%d_outwidth)

                        c, ctrs_residual = dace.int32((gemm_k+cta_k+warp_k)/d_kdim3), dace.int32((gemm_k+cta_k+warp_k)%d_kdim3)
                        t, trs_residual = dace.int32(ctrs_residual/d_kdim2), dace.int32(ctrs_residual%d_kdim2)
                        r, s = dace.int32(trs_residual/d_kdim), dace.int32(trs_residual%d_kdim)
                        d, h, w = o + t, p + r, q + s

                        warp_reducedk[gemm_m, gemm_n] = warp_reducedk[gemm_m, gemm_n] + Input[n, d, h, w, c]*kernel[gemm_n+cta_n+warp_n, t, r, s, c]
            for tmp_m, tmp_n in dace.map[0: WARPtileM, 0:WARPtileN]@dace.ScheduleType.Sequential:
                    cta_reducedk[tmp_m+warp_m, warp_n+tmp_n] = cta_reducedk[tmp_m+warp_m, warp_n+tmp_n] + warp_reducedk[tmp_m, tmp_n]

        for warp_n, warp_m in dace.map[0: CTAtileN:WARPtileN, 0: CTAtileM:WARPtileM]@dace.ScheduleType.GPU_ThreadBlock:
            for assign_n, assign_m in dace.map[warp_n:WARPtileN+warp_n, warp_m:WARPtileM+warp_m]@dace.ScheduleType.Sequential:
                n, nopq_residual = dace.int32((cta_m+assign_m)/d_DHW), dace.int32((cta_m+assign_m) % d_DHW)
                o, opq_residual = dace.int32(nopq_residual/d_HW), dace.int32(nopq_residual%d_HW)        
                p, q = dace.int32(opq_residual/d_outwidth), dace.int32(opq_residual%d_outwidth)
                Output[ n, o, p, q, cta_n+assign_n] = cta_reducedk[assign_m, assign_n]


# Also paritioning the CTAtileK (Sliced K reduction) Useful when GEMM_K is large.
CTAtileM =  32
CTAtileN = 4
CTAtileK = 4

WARPtileM = 2
WARPtileN = 2
WARPtileK = 1

ncta_m = np.int32((128*128*128*16)/CTAtileM)
ncta_n = np.int32(256/CTAtileN)
ncta_k = np.int32(256/CTAtileK)

# CTAtileM/WARPtileM * CTAtileN/WARPtileN * CTAtileK/WARPtileK <= 1024
# Assertion for shared memory 
assert((CTAtileM * CTAtileN + CTAtileM * CTAtileK + CTAtileN*CTAtileK)*4 < 81920)
# Assertion for thread size
assert(np.int32((CTAtileM/WARPtileM)*(CTAtileN/WARPtileN)*(CTAtileK/WARPtileK))<1024)


# Use code from here to revert to last working version: https://gitlab.ethz.ch/abaranwal/dacelocal/-/blob/7c78decc3d7dbd3da47576e24bc0c5d051601f65/samples/simple/implicitGemmTileddace.py
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
    commonCTA = dace.ndarray([ncta_n, ncta_m, ncta_k, CTAtileM, CTAtileN], dtype=Input.dtype)
    commonCTA[:] = 0
    for cta_n, cta_m, cta_k in dace.map[0:d_GEMM_N:CTAtileN, 0:d_GEMM_M:CTAtileM, 0:d_GEMM_K:CTAtileK] @dace.ScheduleType.GPU_Device:
        cta_reducedk = dace.ndarray([CTAtileM, CTAtileN], dtype=Input.dtype, storage=dace.StorageType.GPU_Shared)
        cta_reducedk[:] = 0
        for warp_n, warp_m in dace.map[0: CTAtileN:WARPtileN, 0: CTAtileM:WARPtileM]@dace.ScheduleType.GPU_ThreadBlock:
            warp_reducedk = dace.ndarray([WARPtileM, WARPtileN], dtype=Input.dtype, storage=dace.StorageType.Register)
            warp_reducedk[:] = 0
            for warp_k in dace.map[0:CTAtileK:WARPtileK]@dace.ScheduleType.Sequential:
                for gemm_k in dace.map[0: WARPtileK]@dace.ScheduleType.Sequential:
                    for gemm_m, gemm_n in dace.map[0:WARPtileM, 0:WARPtileN]@dace.ScheduleType.Sequential:
                        n, nopq_residual =  dace.int32((gemm_m+cta_m+warp_m)/d_DHW), dace.int32((gemm_m+cta_m+warp_m) % d_DHW)
                        o, opq_residual = dace.int32(nopq_residual/d_HW), dace.int32(nopq_residual%d_HW)
                        p, q = dace.int32(opq_residual/d_outwidth), dace.int32(opq_residual%d_outwidth)

                        c, ctrs_residual = dace.int32((gemm_k+cta_k+warp_k)/d_kdim3), dace.int32((gemm_k+cta_k+warp_k)%d_kdim3)
                        t, trs_residual = dace.int32(ctrs_residual/d_kdim2), dace.int32(ctrs_residual%d_kdim2)
                        r, s = dace.int32(trs_residual/d_kdim), dace.int32(trs_residual%d_kdim)
                        d, h, w = o + t, p + r, q + s
                        warp_reducedk[gemm_m, gemm_n] = warp_reducedk[gemm_m, gemm_n] + Input[n, d, h, w, c]*kernel[gemm_n+cta_n+warp_n, t, r, s, c]
            for tmp_m, tmp_n in dace.map[0: WARPtileM, 0:WARPtileN]@dace.ScheduleType.Sequential:
                cta_reducedk[tmp_m+warp_m, warp_n+tmp_n] = cta_reducedk[tmp_m+warp_m, warp_n+tmp_n] + warp_reducedk[tmp_m, tmp_n]

        for warp_n, warp_m in dace.map[0: CTAtileN:WARPtileN, 0: CTAtileM:WARPtileM]@dace.ScheduleType.GPU_ThreadBlock:
                for assign_n, assign_m in dace.map[warp_n:WARPtileN+warp_n, warp_m:WARPtileM+warp_m]@dace.ScheduleType.Sequential:
                    n, nopq_residual = dace.int32((cta_m+assign_m)/d_DHW), dace.int32((cta_m+assign_m) % d_DHW)
                    o, opq_residual = dace.int32(nopq_residual/d_HW), dace.int32(nopq_residual%d_HW)        
                    p, q = dace.int32(opq_residual/d_outwidth), dace.int32(opq_residual%d_outwidth)
                    icta_n = dace.int32(cta_n/CTAtileN)
                    icta_m = dace.int32(cta_m/CTAtileM)
                    icta_k = dace.int32(cta_k/CTAtileK)
                    commonCTA[icta_n, icta_m, 0, assign_m, assign_n] = commonCTA[icta_n, icta_m, icta_k, assign_m, assign_n] + cta_reducedk[assign_m, assign_n]

    # Epilogue
    for cta_n, cta_m in dace.map[0:d_GEMM_N:CTAtileN, 0:d_GEMM_M:CTAtileM]@dace.ScheduleType.GPU_Device:
        cta_newreducedk = dace.ndarray([CTAtileM, CTAtileN], dtype=Input.dtype, storage=dace.StorageType.GPU_Shared)
        cta_newreducedk[:] = 0

        for warp_n, warp_m in dace.map[0: CTAtileN:WARPtileN, 0: CTAtileM:WARPtileM]@dace.ScheduleType.GPU_ThreadBlock:
            for cta_k in dace.map[0:d_GEMM_K:CTAtileK]@dace.ScheduleType.Sequential:
                for assign_n, assign_m in dace.map[warp_n:WARPtileN+warp_n, warp_m:WARPtileM+warp_m]@dace.ScheduleType.Sequential:
                    icta_n = dace.int32(cta_n/CTAtileN)
                    icta_m = dace.int32(cta_m/CTAtileM)
                    icta_k = dace.int32(cta_k/CTAtileK)
                    cta_newreducedk[assign_m, assign_n] = cta_newreducedk[assign_m, assign_n] + commonCTA[icta_n, icta_m, icta_k, assign_m, assign_n]
        for warp_n, warp_m in dace.map[0: CTAtileN:WARPtileN, 0: CTAtileM:WARPtileM]@dace.ScheduleType.GPU_ThreadBlock:
                for assign_n, assign_m in dace.map[warp_n:WARPtileN+warp_n, warp_m:WARPtileM+warp_m]@dace.ScheduleType.Sequential:
                    n, nopq_residual = dace.int32((cta_m+assign_m)/d_DHW), dace.int32((cta_m+assign_m) % d_DHW)
                    o, opq_residual = dace.int32(nopq_residual/d_HW), dace.int32(nopq_residual%d_HW)        
                    p, q = dace.int32(opq_residual/d_outwidth), dace.int32(opq_residual%d_outwidth)
                    Output[ n, o, p, q, cta_n+assign_n] = cta_newreducedk[assign_m, assign_n]