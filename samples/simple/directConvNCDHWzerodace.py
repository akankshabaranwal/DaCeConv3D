import dace
import numpy as np
from dace import dtypes

from dace.transformation.interstate import StateFusion

# Define symbolic sizes for arbitrary inputs
d_outdepth = dace.symbol('d_outdepth')
d_outheight = dace.symbol('d_outheight')
d_outwidth = dace.symbol('d_outwidth')
d_inchannels = dace.symbol('d_inchannels')
d_outchannels = dace.symbol('d_outchannels')
d_batchsize = dace.symbol('d_batchsize')
d_kdim = dace.symbol('d_kdim')

# Define data type to use
dtype = dace.float32
np_dtype = np.float32

# Optimize code on the GPU
def optimize_for_gpu(sdfg: dace.SDFG):
    """ Optimize 3D convolution example for GPUs. """
    dace.Config.set('compiler', 'default_data_types', value='C')
    return

# BlockD = 1
# BlockH = 16
# BlockOC = 1

# # Simple parallel 3D convolution. Direct convolution
# @dace.program(device=dtypes.DeviceType.GPU, auto_optimize=False)
# def dace_conv3d( Input: dtype[d_batchsize, d_inchannels, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1] @dace.StorageType.GPU_Global ,
#                 kernel: dtype[d_outchannels, d_inchannels, d_kdim, d_kdim, d_kdim] @dace.StorageType.GPU_Global,
#                 Output: dtype[d_batchsize, d_outchannels, d_outdepth, d_outheight, d_outwidth] @dace.StorageType.GPU_Global):
#     for n, b_d, b_oc in dace.map[0:d_batchsize, 0:d_outdepth:BlockD, 0:d_outchannels:BlockOC]@dace.ScheduleType.GPU_Device:
#         for d, b_h in dace.map[0:BlockD, 0:d_outheight:BlockH]@dace.ScheduleType.GPU_ThreadBlock:
#             for h, w in dace.map[0:BlockH, 0:d_outwidth]:
#                 tmp = dace.ndarray([1], dtype=Input.dtype, storage=dace.StorageType.Register)
#                 tmp = 0
#                 for kd, kh, kw, ic in dace.map[0:d_kdim, 0:d_kdim, 0:d_kdim, 0:d_inchannels]@dace.ScheduleType.Sequential:
#                     tmp = tmp+Input[n, ic, b_d+d+kd, b_h+h+kh, w+kw] * kernel[b_oc+oc, ic, kd, kh, kw]
#                 Output[n, b_oc+oc, b_d+d, b_h+h, w] = tmp
CTAtileDHW = 32
CTAtileOC = 32
CTAtileIC = 4

@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=False)
def dace_conv3d( Input: dtype[d_batchsize,  d_inchannels, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1] @dace.StorageType.GPU_Global ,
                kernel: dtype[d_outchannels, d_inchannels, d_kdim, d_kdim, d_kdim] @dace.StorageType.GPU_Global,
                Output: dtype[d_batchsize, d_outchannels, d_outdepth, d_outheight, d_outwidth] @dace.StorageType.GPU_Global):
        
    d_DHW = d_outdepth*d_outheight*d_outwidth
    d_HW = d_outheight*d_outwidth
    for cta_oc, n, cta_dhw in dace.map[ 0:d_outchannels:CTAtileOC, 0:d_batchsize, 0:d_DHW:CTAtileDHW]@dace.ScheduleType.GPU_Device:
        for oc, dhw in dace.map[ 0:CTAtileOC, 0:CTAtileDHW]@dace.ScheduleType.GPU_ThreadBlock:
            d, dhw_residual = dace.int32((dhw+cta_dhw)/d_HW), dace.int32((dhw+cta_dhw)%d_HW)
            h, w = dace.int32(dhw_residual/d_outheight), dace.int32(dhw_residual%d_outheight)
            tmp = dace.ndarray([1], dtype=Input.dtype, storage=dace.StorageType.Register)
            tmp = 0
            for ic, kd, kh, kw in dace.map[0:d_inchannels, 0:d_kdim, 0:d_kdim, 0:d_kdim]@dace.ScheduleType.Sequential:
                tmp = tmp + Input[ n, ic, d+kd, h+kh, w+kw]*kernel[oc+cta_oc, ic, kd, kh, kw]
            Output[n, oc+cta_oc, d, h, w] = tmp