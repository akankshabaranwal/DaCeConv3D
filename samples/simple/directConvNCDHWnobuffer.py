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
    return

CTAtileDHW = 64 # This should be min [sqrt(Ss), dhw] 64, 32, 16 works for layers 0 to 5; 8 works for layer 6
#CTAtileOC = 16 # This should be min [sqrt(Ss), OC] 16 works for layers 0 to 7; 32 doesn't work for anything
# Nothing works for layer 0
# 16 works for layer 1 to 6
# 32 doesn't work
 
batchsize = 16
kdim = 3

# # The one with 32 ms runtime.
@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=False)
def dace_conv3d( Input: dtype[d_batchsize,  d_inchannels, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1] @dace.StorageType.GPU_Global ,
                kernel: dtype[d_outchannels, d_inchannels, d_kdim, d_kdim, d_kdim] @dace.StorageType.GPU_Global,
                Output: dtype[d_batchsize, d_outchannels, d_outdepth, d_outheight, d_outwidth] @dace.StorageType.GPU_Global):
        
    d_DHW = d_outdepth*d_outheight*d_outwidth
    d_HW = d_outheight*d_outwidth
    for cta_n, cta_dhw, cta_oc in dace.map[0:d_batchsize, 0:d_DHW:CTAtileDHW, 0:d_outchannels]@dace.ScheduleType.GPU_Device:
        for dhw in dace.map[ 0:CTAtileDHW]@dace.ScheduleType.GPU_ThreadBlock:
            d, dhw_residual = dace.int32((dhw+cta_dhw)/d_HW), dace.int32((dhw+cta_dhw)%d_HW)
            h, w = dace.int32(dhw_residual/d_outheight), dace.int32(dhw_residual%d_outheight)
            tmp = dace.ndarray([1], dtype=Input.dtype, storage=dace.StorageType.Register)
            tmp = 0
            for ic, kd, kh, kw in dace.map[0:d_inchannels, 0:d_kdim, 0:d_kdim, 0:d_kdim]@dace.ScheduleType.Sequential:
                tmp = tmp + Input[ cta_n, ic, d+kd, h+kh, w+kw]*kernel[cta_oc, ic, kd, kh, kw]
            Output[cta_n, cta_oc, d, h, w] = tmp
