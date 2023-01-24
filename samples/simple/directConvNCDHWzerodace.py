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

# BlockD = 4
# BlockH = 16
# BlockOC = 1
# placeidx = 1

blockOC = 4
blockIC = 4
blockD = 1

blockH = 4
blockW = 4

# # Simple parallel 3D convolution. Direct convolution
@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=False)
def dace_conv3d( Input: dtype[d_batchsize, d_inchannels, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1] @dace.StorageType.GPU_Global ,
                kernel: dtype[d_outchannels, d_inchannels, d_kdim, d_kdim, d_kdim] @dace.StorageType.GPU_Global,
                Output: dtype[d_batchsize, d_outchannels, d_outdepth, d_outheight, d_outwidth] @dace.StorageType.GPU_Global):
    # for n, b_d, b_oc in dace.map[0:d_batchsize, 0:d_outdepth:BlockD, 0:d_outchannels:BlockOC]@dace.ScheduleType.GPU_Device:
    #     for d, b_h, oc in dace.map[0:BlockD:placeidx, 0:d_outheight:BlockH, 0:BlockOC:placeidx]@dace.ScheduleType.GPU_ThreadBlock:
    #         for h, w in dace.map[0:BlockH, 0:d_outwidth]:
    #             tmp = dace.ndarray([1], dtype=Input.dtype, storage=dace.StorageType.Register)
    #             tmp = 0
    #             for kd, kh, kw, ic in dace.map[0:d_kdim, 0:d_kdim, 0:d_kdim, 0:d_inchannels]@dace.ScheduleType.Sequential:
    #                 tmp = tmp+Input[n, ic, b_d+d+kd, b_h+h+kh, w+kw] * kernel[b_oc+oc, ic, kd, kh, kw]
    #             Output[n, b_oc+oc, b_d+d, b_h+h, w] = tmp
    for n, b_oc, b_d in dace.map[0:d_batchsize, 0:d_outchannels:blockOC, 0:d_outdepth:blockD]@dace.ScheduleType.GPU_Device:
        for b_ic in dace.map[0:d_inchannels:blockIC]@dace.ScheduleType.Sequential:
            for b_w, b_h in dace.map[0:d_outwidth:blockW, 0:d_outheight:blockH]@dace.ScheduleType.GPU_ThreadBlock:
                for w, h in dace.map[ 0:blockW, 0:blockH]@dace.ScheduleType.Sequential:
                    for kd, kh, kw in dace.map[0:d_kdim, 0:d_kdim, 0:d_kdim]@dace.ScheduleType.Sequential:
                        for ic, d, oc in dace.map[0:blockIC, 0:blockD, 0:blockOC]@dace.ScheduleType.Sequential:
                            Output[n, b_oc+oc, b_d+d, b_h+h, b_w+w] = Output[n, b_oc+oc, b_d+d, b_h+h, b_w+w] + Input[n, b_ic+ic, b_d+d+kd, b_h+h+kh, b_w+w+kw] * kernel[b_oc+oc, ic+b_ic, kd, kh, kw]