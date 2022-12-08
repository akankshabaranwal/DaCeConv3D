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
# Define data type to use
dtype = dace.float32
np_dtype = np.float32

def optimize_for_gpu(sdfg: dace.SDFG):
    """ Optimize 3D convolution example for GPUs. """
    print("INFO: Calling optimize for GPU function")
    dace.Config.set('compiler', 'default_data_types', value='C')
    # Fuse the map and reduce nodes
    # Apply GPU transformation
    sdfg.apply_transformations(MapReduceFusion)
    sdfg.apply_gpu_transformations()
    
    btile = find_map_by_param(sdfg, 'warp_m')
    btile.map.schedule = dace.ScheduleType.GPU_ThreadBlock
    return
    ktile = find_map_by_param(sdfg, 'cta_k')
    smem_Input = InLocalStorage.apply_to(sdfg, dict(array='Input'), node_a=ktile, node_b=btile)
    smem_kernel = InLocalStorage.apply_to(sdfg, dict(array='kernel'), node_a=ktile, node_b=btile)
    sdfg.arrays[smem_Input.data].storage = dace.StorageType.GPU_Shared
    sdfg.arrays[smem_kernel.data].storage = dace.StorageType.GPU_Shared

    return


CTAtileM =  8
CTAtileN = 4
CTAtileK = 2

WARPtileM = 2
WARPtileN = 2
WARPtileK = 1


# Tiling and buffering
#TODO: Check what happens if you change the kdim variables to constants instead of dace map.
@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=True, regenerate_code=True)
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
            cta_reducedk = np.zeros([CTAtileM, CTAtileN], dtype=Input.dtype)
            for cta_k in dace.map[0:d_GEMM_K:CTAtileK]:
            #    cta_splitk = np.zeros([CTAtileM, CTAtileN], dtype=Input.dtype)

                for warp_n, warp_m in dace.map[0: CTAtileN:WARPtileN, 0: CTAtileM:WARPtileM] @dace.ScheduleType.GPU_ThreadBlock:
                        warp_reducedk = np.zeros([WARPtileM, WARPtileN], dtype=Input.dtype)
                        for warp_k in dace.map[0:CTAtileK:WARPtileK]:
                            #warp_splitk = np.zeros([WARPtileM, WARPtileN], dtype=Input.dtype)
                            
                            for gemm_k in dace.map[0: WARPtileK]:
                                for gemm_m, gemm_n in dace.map[0:WARPtileM, 0:WARPtileN]:
                                            n, nopq_residual =  dace.int32((gemm_m+cta_m+warp_m)/d_DHW), dace.int32((gemm_m+cta_m+warp_m) % d_DHW)
                                            o, opq_residual = dace.int32(nopq_residual/d_HW), dace.int32(nopq_residual%d_HW)
                                            p, q = dace.int32(opq_residual/d_outwidth), dace.int32(opq_residual%d_outwidth)

                                            c, ctrs_residual = dace.int32((gemm_k+cta_k+warp_k)/d_kdim3), dace.int32((gemm_k+cta_k+warp_k)%d_kdim3)
                                            t, trs_residual = dace.int32(ctrs_residual/d_kdim2), dace.int32(ctrs_residual%d_kdim2)
                                            r, s = dace.int32(trs_residual/d_kdim), dace.int32(trs_residual%d_kdim)
                                            d, h, w = o + t, p + r, q + s

                                            #warp_splitk[gemm_m, gemm_n] = Input[n, d, h, w, c]*kernel[gemm_n+cta_n+warp_n, t, r, s, c]
                                            warp_reducedk[gemm_m, gemm_n] = warp_reducedk[gemm_m, gemm_n] + Input[n, d, h, w, c]*kernel[gemm_n+cta_n+warp_n, t, r, s, c]
                                #warp_reducedk = warp_reducedk + warp_splitk
                        for tmp_m, tmp_n in dace.map[0: WARPtileM, 0:WARPtileN]:
                                #cta_splitk[tmp_m+warp_m, warp_n+tmp_n] = warp_reducedk[tmp_m, tmp_n]
                                cta_reducedk[tmp_m+warp_m, warp_n+tmp_n] = cta_reducedk[tmp_m+warp_m, warp_n+tmp_n] + warp_reducedk[tmp_m, tmp_n]
                #cta_reducedk = cta_reducedk+cta_splitk

            for warp_n, warp_m in dace.map[0: CTAtileN:WARPtileN, 0: CTAtileM:WARPtileM] @dace.ScheduleType.GPU_ThreadBlock:
                    for assign_n, assign_m in dace.map[warp_n:WARPtileN+warp_n, warp_m:WARPtileM+warp_m]:
                        n, nopq_residual = dace.int32((cta_m+assign_m)/d_DHW), dace.int32((cta_m+assign_m) % d_DHW)
                        o, opq_residual = dace.int32(nopq_residual/d_HW), dace.int32(nopq_residual%d_HW)        
                        p, q = dace.int32(opq_residual/d_outwidth), dace.int32(opq_residual%d_outwidth)
                        Output[ n, o, p, q, cta_n+assign_n] = cta_reducedk[assign_m, assign_n]

