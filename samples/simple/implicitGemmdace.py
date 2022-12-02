import dace
import numpy as np
from dace import dtypes
from dace.transformation.interstate import StateFusion
from dace.transformation.dataflow import TaskletFusion, MapReduceFusion

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
# Define data type to use
dtype = dace.float32
np_dtype = np.float32

def optimize_for_gpu(sdfg: dace.SDFG):
    """ Optimize 3D convolution example for GPUs. """
    dace.Config.set('compiler', 'default_data_types', value='C')
    # Fuse the map and reduce nodes
    # Apply GPU transformation
    #sdfg.apply_transformations_repeated(MapReduceFusion)
    sdfg.apply_gpu_transformations()
    #sdfg.apply_transformations_repeated(TaskletFusion)
    return

@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=True)
def dace_conv3d_baseline(Input: dtype[d_batchsize, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1, d_inchannels] @dace.StorageType.GPU_Global ,
                kernel: dtype[d_outchannels, d_kdim, d_kdim, d_kdim, d_inchannels] @dace.StorageType.GPU_Global,
                Output: dtype[d_batchsize, d_outdepth, d_outheight, d_outwidth, d_outchannels] @dace.StorageType.GPU_Global):
    
    d_GEMM_M = (d_batchsize*d_outdepth*d_outheight*d_outwidth)
    d_GEMM_N = d_outchannels
    d_GEMM_K = (d_inchannels * d_kdim * d_kdim * d_kdim)
    DHW = d_outdepth*d_outheight*d_outwidth
    HW = d_outheight*d_outwidth
    kdim3 = d_kdim*d_kdim*d_kdim
    kdim2 = d_kdim*d_kdim

    for gemm_i, gemm_j in dace.map[0:d_GEMM_M, 0:d_GEMM_N]:
        n = dace.int32(gemm_i/DHW)
        nopq_residual = dace.int32(gemm_i % DHW)

        o = dace.int32(nopq_residual/HW)
        opq_residual = dace.int32(nopq_residual%HW)        
        p = dace.int32(opq_residual/d_outwidth)
        q = dace.int32(opq_residual%d_outwidth)
        
        accum = np.zeros([1], dtype=Input.dtype)
        
        for gemm_k in dace.map[0: d_GEMM_K]:
            c = dace.int32(gemm_k/kdim3)
            ctrs_residual = dace.int32(gemm_k%kdim3)

            t = dace.int32(ctrs_residual/kdim2)
            trs_residual = dace.int32(ctrs_residual%kdim2)

            r = dace.int32(trs_residual/d_kdim)
            s = dace.int32(trs_residual%d_kdim)
            
            d = o + t
            h = p + r
            w = q + s

            accum = accum + Input[n, d, h, w, c]*kernel[gemm_j, t, r, s, c]

        Output[ n, o, p, q, gemm_j] = accum
    


CTAtileM = 4
CTAtileN = 4
CTAtileK = 4

WARPtileM = 2
WARPtileN = 2
WARPtileK = 2

# Tile based on matrix multiplication code
@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=True)
def dace_conv3d_bad(Input: dtype[d_batchsize, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1, d_inchannels] @dace.StorageType.GPU_Global ,
                kernel: dtype[d_outchannels, d_kdim, d_kdim, d_kdim, d_inchannels] @dace.StorageType.GPU_Global,
                Output: dtype[d_batchsize, d_outdepth, d_outheight, d_outwidth, d_outchannels] @dace.StorageType.GPU_Global):
    
    d_GEMM_M = (d_batchsize*d_outdepth*d_outheight*d_outwidth)
    d_GEMM_N = d_outchannels
    d_GEMM_K = (d_inchannels * d_kdim * d_kdim * d_kdim)
    DHW = d_outdepth*d_outheight*d_outwidth
    HW = d_outheight*d_outwidth
    kdim3 = d_kdim*d_kdim*d_kdim
    kdim2 = d_kdim*d_kdim

    tmp_splitk = np.ndarray([d_batchsize*d_outdepth*d_outheight*d_outwidth, d_outchannels, d_inchannels * d_kdim * d_kdim * d_kdim], dtype=Input.dtype)

    for gemm_i, gemm_j, gemm_k in dace.map[0:d_GEMM_M, 0:d_GEMM_N, 0:d_GEMM_K]:
        n, nopq_residual = dace.int32(gemm_i/DHW), dace.int32(gemm_i % DHW)
        o, opq_residual = dace.int32(nopq_residual/HW), dace.int32(nopq_residual%HW)
        p, q = dace.int32(opq_residual/d_outwidth), dace.int32(opq_residual%d_outwidth)

        c, ctrs_residual = dace.int32(gemm_k/kdim3), dace.int32(gemm_k%kdim3)
        t, trs_residual = dace.int32(ctrs_residual/kdim2), dace.int32(ctrs_residual%kdim2)
        r, s = dace.int32(trs_residual/d_kdim), dace.int32(trs_residual%d_kdim)

        d, h, w = (o + t), (p + r), (q + s)

        tmp_splitk[gemm_i, gemm_j, gemm_k] = Input[n, d, h, w, c]*kernel[gemm_j, t, r, s, c]

    tmp_reducedk = np.ndarray([d_batchsize*d_outdepth*d_outheight*d_outwidth, d_outchannels], dtype=Input.dtype)
    dace.reduce(lambda a, b: a + b, tmp_splitk, tmp_reducedk, axis=2, identity=0)
    for gemm_i, gemm_j in dace.map[0:d_GEMM_M, 0:d_GEMM_N]:
        n, nopq_residual = dace.int32(gemm_i/DHW), dace.int32(gemm_i % DHW)
        o, opq_residual = dace.int32(nopq_residual/HW), dace.int32(nopq_residual%HW)
        p, q = dace.int32(opq_residual/d_outwidth), dace.int32(opq_residual%d_outwidth)
        Output[ n, o, p, q, gemm_j] = tmp_reducedk[gemm_i, gemm_j]

CTAtileM = 4
CTAtileN = 4
CTAtileK = 4

WARPtileM = 2
WARPtileN = 2
WARPtileK = 2
# Tiling and buffering
@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=True)
def dace_conv3d(Input: dtype[d_batchsize, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1, d_inchannels] @dace.StorageType.GPU_Global ,
                kernel: dtype[d_outchannels, d_kdim, d_kdim, d_kdim, d_inchannels] @dace.StorageType.GPU_Global,
                Output: dtype[d_batchsize, d_outdepth, d_outheight, d_outwidth, d_outchannels] @dace.StorageType.GPU_Global):
    
    d_GEMM_M = (d_batchsize*d_outdepth*d_outheight*d_outwidth)
    d_GEMM_N = d_outchannels
    d_GEMM_K = (d_inchannels * d_kdim * d_kdim * d_kdim)
    d_DHW = d_outdepth*d_outheight*d_outwidth
    d_HW = d_outheight*d_outwidth
    kdim3 = d_kdim*d_kdim*d_kdim
    kdim2 = d_kdim*d_kdim

    for cta_n, cta_m in dace.map[0:d_GEMM_N:CTAtileN, 0:d_GEMM_M:CTAtileM] @dace.ScheduleType.GPU_Device:
            cta_reducedk = np.zeros([CTAtileM, CTAtileN], dtype=Input.dtype)
            for cta_k in dace.map[0:d_GEMM_K:CTAtileK]:
                cta_splitk = np.zeros([CTAtileM, CTAtileN], dtype=Input.dtype)

                for warp_n, warp_m in dace.map[0: CTAtileN:WARPtileN, 0: CTAtileM:WARPtileM]:
                        warp_reducedk = np.zeros([WARPtileM, WARPtileN], dtype=Input.dtype)
                        for warp_k in dace.map[0:CTAtileK:WARPtileK]:
                            warp_splitk = np.zeros([WARPtileM, WARPtileN], dtype=Input.dtype)
                            
                            for gemm_k in dace.map[0: WARPtileK]:
                                for gemm_m, gemm_n in dace.map[0:WARPtileM, 0:WARPtileN] @dace.ScheduleType.GPU_ThreadBlock:
                                            n, nopq_residual = dace.int32((gemm_m+cta_m+warp_m)/d_DHW), dace.int32((gemm_m+cta_m+warp_m) % d_DHW)
                                            o, opq_residual = dace.int32(nopq_residual/d_HW), dace.int32(nopq_residual%d_HW)        
                                            p, q = dace.int32(opq_residual/d_outwidth), dace.int32(opq_residual%d_outwidth)

                                            c, ctrs_residual = dace.int32((gemm_k+cta_k+warp_k)/kdim3), dace.int32((gemm_k+cta_k+warp_k)%kdim3)
                                            t, trs_residual = dace.int32(ctrs_residual/kdim2), dace.int32(ctrs_residual%kdim2)
                                            r, s = dace.int32(trs_residual/d_kdim), dace.int32(trs_residual%d_kdim)

                                            d, h, w = o + t, p + r, q + s

                                            warp_splitk[gemm_m, gemm_n] = Input[n, d, h, w, c]*kernel[gemm_n+cta_n+warp_n, t, r, s, c]
                                warp_reducedk = warp_reducedk + warp_splitk
                        for tmp_m, tmp_n in dace.map[0: WARPtileM, 0:WARPtileN]:
                                cta_splitk[tmp_m+warp_m, warp_n+tmp_n] = warp_reducedk[tmp_m, tmp_n]
                cta_reducedk = cta_reducedk+cta_splitk

            for assign_m, assign_n in dace.map[0:CTAtileM, 0:CTAtileN]:
                    n, nopq_residual = dace.int32((cta_m+assign_m)/d_DHW), dace.int32((cta_m+assign_m) % d_DHW)
                    o, opq_residual = dace.int32(nopq_residual/d_HW), dace.int32(nopq_residual%d_HW)        
                    p, q = dace.int32(opq_residual/d_outwidth), dace.int32(opq_residual%d_outwidth)
                    Output[ n, o, p, q, cta_n+assign_n] = cta_reducedk[assign_m, assign_n]