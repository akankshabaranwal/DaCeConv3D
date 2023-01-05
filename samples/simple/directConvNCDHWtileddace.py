import dace
import numpy as np
from dace import dtypes

from dace.transformation.interstate import StateFusion
from convutils import find_map_by_param

# Define symbolic sizes for arbitrary inputs
d_outdepth = dace.symbol('d_outdepth')
d_outheight = dace.symbol('d_outheight')
d_outwidth = dace.symbol('d_outwidth')
d_inchannels = dace.symbol('d_inchannels')
d_outchannels = dace.symbol('d_outchannels')
d_batchsize = dace.symbol('d_batchsize')
d_kdim = dace.symbol('d_kdim')

d_DHW = dace.symbol('d_DHW')
d_HW = dace.symbol('d_HW')
d_CTAtileDHW = dace.symbol('d_CTAtileDHW')


# Define data type to use
dtype = dace.float32
np_dtype = np.float32


# Optimize code on the GPU
def optimize_for_gpu(sdfg: dace.SDFG):
    """ Optimize 3D convolution example for GPUs. """
    print("INFO: Calling optimize for GPU function")
    dace.Config.set('compiler', 'default_data_types', value='C')
    # tmp_i = find_map_by_param(sdfg, '__i0')
    # tmp_i.map.schedule = dace.ScheduleType.Sequential
    return


CTAtileN = 1
CTAtileDHW = 64 # This should divide outdepth, outheight, outwidth individually otherwise the indexing gets tricky.
CTAtileOC = 1

@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=False)
def dace_conv3d( Input: dtype[d_batchsize,  d_inchannels, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1] @dace.StorageType.GPU_Global ,
                kernel: dtype[d_outchannels, d_inchannels, d_kdim, d_kdim, d_kdim] @dace.StorageType.GPU_Global,
                Output: dtype[d_batchsize, d_outchannels, d_outdepth, d_outheight, d_outwidth] @dace.StorageType.GPU_Global):
    
    d_DHW = d_outdepth*d_outheight*d_outwidth
    d_HW = d_outheight*d_outwidth

    for n, dhw, oc in dace.map[0:d_batchsize, 0:d_DHW, 0:d_outchannels]@dace.ScheduleType.GPU_Device:
            d, dhw_residual = dace.int32((dhw)/d_HW), dace.int32((dhw)%d_HW)
            h, w = dace.int32(dhw_residual/d_outheight), dace.int32(dhw_residual%d_outheight)
            for ic, kd, kh, kw in dace.map[0:d_inchannels, 0:d_kdim, 0:d_kdim, 0:d_kdim]@dace.ScheduleType.Sequential:
                Output[n, oc, d, h, w] = Output[n, oc, d, h, w] + Input[ n, ic, d+kd, h+kh, w+kw]*kernel[oc, ic, kd, kh, kw]

# @dace.program(device=dtypes.DeviceType.GPU, auto_optimize=False)
# def dace_conv3d_tiled( Input: dtype[d_batchsize,  d_inchannels, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1] @dace.StorageType.GPU_Global ,
#                 kernel: dtype[d_outchannels, d_inchannels, d_kdim, d_kdim, d_kdim] @dace.StorageType.GPU_Global,
#                 Output: dtype[d_batchsize, d_outchannels, d_outdepth, d_outheight, d_outwidth] @dace.StorageType.GPU_Global):
    
#     d_DHW = d_outdepth*d_outheight*d_outwidth
#     d_HW = d_outheight*d_outwidth

#     for cta_n, cta_dhw, cta_oc in dace.map[0:d_batchsize:CTAtileN, 0:d_DHW:CTAtileDHW, 0:d_outchannels:CTAtileOC]@dace.ScheduleType.GPU_Device:
#         cta_shared = dace.ndarray([CTAtileN, CTAtileDHW, CTAtileOC], dtype=Input.dtype)
#         cta_shared[:] = 0
#         for n, dhw, oc in dace.map[0:CTAtileN, 0:CTAtileDHW, 0:CTAtileOC]@dace.ScheduleType.GPU_ThreadBlock:
#             d, dhw_residual = dace.int32((dhw+cta_dhw)/d_HW), dace.int32((dhw+cta_dhw)%d_HW)
#             h, w = dace.int32(dhw_residual/d_outheight), dace.int32(dhw_residual%d_outheight)
#             for ic, kd, kh, kw in dace.map[0:d_inchannels, 0:d_kdim, 0:d_kdim, 0:d_kdim]@dace.ScheduleType.Sequential:
#                 cta_shared[n, dhw, oc] = cta_shared[n, dhw, oc] + Input[ n+cta_n, ic, d+kd, h+kh, w+kw]*kernel[oc+cta_oc, ic, kd, kh, kw]
#         for n, dhw, oc in dace.map[0:CTAtileN, 0:CTAtileDHW, 0:CTAtileOC]@dace.ScheduleType.GPU_ThreadBlock:
#             d, dhw_residual = dace.int32((dhw+cta_dhw)/d_HW), dace.int32((dhw+cta_dhw)%d_HW)
#             h, w = dace.int32(dhw_residual/d_outheight), dace.int32(dhw_residual%d_outheight)                         
#             Output[n+cta_n, oc+cta_oc, d, h, w] = cta_shared[n, dhw, oc]