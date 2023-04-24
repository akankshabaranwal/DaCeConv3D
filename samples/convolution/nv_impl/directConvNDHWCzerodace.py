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

blockOC = 4
blockIC = 4
blockD = 4

blockH = 4
blockW = 4

# Simple parallel 3D convolution. Direct convolution
@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=False)
def dace_conv3d( Input: dtype[d_batchsize, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1, d_inchannels] @dace.StorageType.GPU_Global ,
                kernel: dtype[d_outchannels, d_kdim, d_kdim, d_kdim, d_inchannels] @dace.StorageType.GPU_Global,
                Output: dtype[d_batchsize, d_outdepth, d_outheight, d_outwidth, d_outchannels] @dace.StorageType.GPU_Global):

    for n, b_oc, b_d in dace.map[0:d_batchsize, 0:d_outchannels:blockOC, 0:d_outdepth:blockD]@dace.ScheduleType.GPU_Device:
        for b_ic in dace.map[0:d_inchannels:blockIC]@dace.ScheduleType.Sequential:
            for w in dace.map[0:d_outwidth]@dace.ScheduleType.GPU_ThreadBlock:
                for h in dace.map[ 0:d_outheight]@dace.ScheduleType.Sequential:
                    for kd, kh, kw in dace.map[0:d_kdim, 0:d_kdim, 0:d_kdim]@dace.ScheduleType.Sequential:
                        for ic, d, oc in dace.map[0:blockIC, 0:blockD, 0:blockOC]@dace.ScheduleType.Sequential:
                            Output[n, b_d+d, h, w, b_oc+oc] = Output[n, b_d+d, h, w, b_oc+oc] + Input[n, b_d+d+kd, h+kh, w+kw, b_ic+ic] * kernel[b_oc+oc, kd, kh, kw, ic+b_ic]