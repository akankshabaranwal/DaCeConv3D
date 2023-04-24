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
    return


@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=False)
def dace_conv3d(Input: dtype[d_batchsize, d_inchannels, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1] @dace.StorageType.GPU_Global ,
                kernel: dtype[d_outchannels, d_inchannels, d_kdim, d_kdim, d_kdim] @dace.StorageType.GPU_Global,
                Output: dtype[d_batchsize, d_outchannels, d_outdepth, d_outheight, d_outwidth] @dace.StorageType.GPU_Global):
    
    d_GEMM_M = (d_batchsize*d_outdepth*d_outheight*d_outwidth)
    d_GEMM_N = d_outchannels
    d_GEMM_K = (d_inchannels * d_kdim * d_kdim * d_kdim)
    DHW = d_outdepth*d_outheight*d_outwidth
    HW = d_outheight*d_outwidth
    kdim3 = d_kdim*d_kdim*d_kdim
    kdim2 = d_kdim*d_kdim

    for gemm_i, gemm_j in dace.map[0:d_GEMM_M, 0:d_GEMM_N]@dace.ScheduleType.GPU_Device:
        
        for gemm_k in dace.map[0: d_GEMM_K]@dace.ScheduleType.Sequential:
            n = dace.int32(gemm_i/DHW)
            nopq_residual = dace.int32(gemm_i % DHW)

            o = dace.int32(nopq_residual/HW)
            opq_residual = dace.int32(nopq_residual%HW)        
            p = dace.int32(opq_residual/d_outwidth)
            q = dace.int32(opq_residual%d_outwidth)
            
            c = dace.int32(gemm_k/kdim3)
            ctrs_residual = dace.int32(gemm_k%kdim3)

            t = dace.int32(ctrs_residual/kdim2)
            trs_residual = dace.int32(ctrs_residual%kdim2)

            r = dace.int32(trs_residual/d_kdim)
            s = dace.int32(trs_residual%d_kdim)
            
            d = o + t
            h = p + r
            w = q + s

            Output[ n, gemm_j, o, p, q] = Output[ n, gemm_j, o, p, q] + Input[n, c, d, h, w]*kernel[gemm_j, c, t, r, s]