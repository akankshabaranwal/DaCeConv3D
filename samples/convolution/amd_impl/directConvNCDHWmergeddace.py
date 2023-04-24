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
    # Fuse the map and reduce nodes
    return

kdim = 3

# The code to repro the shared memory buffer issue from sdfg
@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=False)
def dace_conv3d( Input: dtype[d_batchsize,  d_inchannels, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1] @dace.StorageType.GPU_Global ,
                kernel: dtype[d_outchannels, d_inchannels, d_kdim, d_kdim, d_kdim] @dace.StorageType.GPU_Global,
                Output: dtype[d_batchsize, d_outchannels, d_outdepth, d_outheight, d_outwidth] @dace.StorageType.GPU_Global):
        
    d_DHW = d_outdepth*d_outheight*d_outwidth
    d_HW = d_outheight*d_outwidth
    for n, dhw, oc in dace.map[0:d_batchsize, 0:d_DHW, 0:d_outchannels]@dace.ScheduleType.GPU_Device:
        for ic in dace.map[0:d_inchannels]@dace.ScheduleType.Sequential:
            d, dhw_residual = dace.int32((dhw)/d_HW), dace.int32((dhw)%d_HW)
            h, w = dace.int32(dhw_residual/d_outheight), dace.int32(dhw_residual%d_outheight)
            for kd, kh, kw in dace.map[0:kdim, 0:kdim, 0:kdim]@dace.ScheduleType.Sequential:
                Output[n, oc, d, h, w]  = Output[n, oc, d, h, w]  + Input[ n, ic, d+kd, h+kh, w+kw]*kernel[oc, ic, kd, kh, kw]