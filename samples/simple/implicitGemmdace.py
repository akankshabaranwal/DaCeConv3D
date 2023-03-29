import dace
import numpy as np
from dace import dtypes
from dace.transformation.interstate import StateFusion
from dace.transformation.dataflow import TaskletFusion, MapReduceFusion, MapCollapse, MapTiling, StripMining
from convutils import find_map_by_param
from dace.transformation import helpers as xfutil


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
    dace.Config.set('compiler', 'default_data_types', value='C')
    # Fuse the map and reduce nodes
    # Apply GPU transformation
    #sdfg.apply_transformations(MapReduceFusion)
    #sdfg.apply_gpu_transformations()
    
    #entry_i = find_map_by_param(sdfg, 'gemm_i')
    #xfutil.tile(sdfg, entry_i, False, False, gemm_i=64, gemm_j=16) # CTA tiling Old values: 64,16 
    #entry_k = find_map_by_param(sdfg, 'gemm_k')
    #xfutil.tile(sdfg, entry_k, False, False, gemm_k=8) # CTA tiling Old values: 64,16
    #xfutil.tile(sdfg, entry_i, False, False, gemm_i=8, gemm_j=4) # Warp tiling Old values: 8, 4

    # # Collapse tiled gemm_i and gemm_j maps
    # gtile_i = find_map_by_param(sdfg, 'tile_gemm_i')
    # gtile_j = find_map_by_param(sdfg, 'tile_gemm_j')
    # MapCollapse.apply_to(sdfg, outer_map_entry=gtile_i, inner_map_entry=gtile_j, permissive=True)
    
    # # Collapse inner gemm_i and gemm_j maps
    # btile_i = find_map_by_param(sdfg, 'tile1_gemm_i')
    # btile_j = find_map_by_param(sdfg, 'tile1_gemm_j')
    # MapCollapse.apply_to(sdfg, outer_map_entry=btile_i, inner_map_entry=btile_j, permissive=True)
    
    # btile = find_map_by_param(sdfg, 'tile1_gemm_i')
    # btile.map.schedule = dace.ScheduleType.GPU_ThreadBlock

    return

    #xfutil.tile(sdfg, entry_k, False, False, gemm_k=4)
    
    return

@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=True)
def dace_conv3d(Input: dtype[d_batchsize, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1, d_inchannels] @dace.StorageType.GPU_Global ,
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