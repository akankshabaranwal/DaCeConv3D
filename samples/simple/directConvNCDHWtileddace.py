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

CTAtileDHW = 64 # This should divide outdepth, outheight, outwidth individually otherwise the indexing gets tricky.
CTAtileOC = 1
inchannels = 16
kdim = 3
# Since cta_kernel is a small kernel, it makes sense to keep it in the shared memory instead of loading it multiple times
# Input and Outputs are used only once so you don't need to load it to the shared memory.
@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=False)
def dace_conv3d( Input: dtype[d_batchsize,  d_inchannels, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1] @dace.StorageType.GPU_Global ,
                kernel: dtype[d_outchannels, d_inchannels, d_kdim, d_kdim, d_kdim] @dace.StorageType.GPU_Global,
                Output: dtype[d_batchsize, d_outchannels, d_outdepth, d_outheight, d_outwidth] @dace.StorageType.GPU_Global):
        
    d_DHW = d_outdepth*d_outheight*d_outwidth
    d_HW = d_outheight*d_outwidth
    for cta_n, cta_dhw, cta_oc in dace.map[0:d_batchsize, 0:d_DHW:CTAtileDHW, 0:d_outchannels]@dace.ScheduleType.GPU_Device:
        cta_shared = dace.ndarray([CTAtileDHW], dtype=Input.dtype, storage=dace.StorageType.GPU_Shared)
        cta_input = dace.ndarray([CTAtileDHW, inchannels* kdim* kdim* kdim], dtype=Input.dtype,storage=dace.StorageType.GPU_Shared)
        cta_kernel = dace.ndarray([inchannels*kdim*kdim*kdim], dtype=Input.dtype,storage=dace.StorageType.GPU_Shared)
        
        for dhw in dace.map[ 0:CTAtileDHW]@dace.ScheduleType.GPU_ThreadBlock:
            cta_shared[dhw] = 0
            d, dhw_residual = dace.int32((dhw+cta_dhw)/d_HW), dace.int32((dhw+cta_dhw)%d_HW)
            h, w = dace.int32(dhw_residual/d_outheight), dace.int32(dhw_residual%d_outheight)
            for ic, kd, kh, kw in dace.map[0:d_inchannels, 0:d_kdim, 0:d_kdim, 0:d_kdim]@dace.ScheduleType.Sequential:
                cta_input[dhw, 27*ic+ 9*kd +3*kh + kw] = Input[ cta_n, ic, d+kd, h+kh, w+kw]
                cta_kernel[ 27*ic+ 9*kd+3*kh + kw] = kernel[cta_oc, ic, kd, kh, kw]

        for dhw in dace.map[ 0:CTAtileDHW]@dace.ScheduleType.GPU_ThreadBlock:
            d, dhw_residual = dace.int32((dhw+cta_dhw)/d_HW), dace.int32((dhw+cta_dhw)%d_HW)
            h, w = dace.int32(dhw_residual/d_outheight), dace.int32(dhw_residual%d_outheight)
            tmp = dace.ndarray([1], dtype=Input.dtype, storage=dace.StorageType.Register)
            tmp = 0
            for ic, kd, kh, kw in dace.map[0:d_inchannels, 0:d_kdim, 0:d_kdim, 0:d_kdim]@dace.ScheduleType.Sequential:
                tmp = tmp + cta_input[ dhw, 27*ic+ 9*kd +3*kh + kw]*cta_kernel[ 27*ic+ 9*kd+3*kh + kw]
            cta_shared[dhw] = tmp

        for dhw in dace.map[ 0:CTAtileDHW]@dace.ScheduleType.GPU_ThreadBlock:
            d, dhw_residual = dace.int32((dhw+cta_dhw)/d_HW), dace.int32((dhw+cta_dhw)%d_HW)
            h, w = dace.int32(dhw_residual/d_outheight), dace.int32(dhw_residual%d_outheight)                         
            Output[cta_n, cta_oc, d, h, w] = cta_shared[dhw]


