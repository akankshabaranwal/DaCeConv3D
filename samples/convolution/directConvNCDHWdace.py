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
    # Apply GPU transformation
    sdfg.apply_transformations_repeated(StateFusion)
    sdfg.simplify()
    sdfg.apply_gpu_transformations()
    return

# Simple parallel 3D convolution. Direct convolution
@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=True, regenerate_code=False)
def dace_conv3d( Input: dtype[d_batchsize, d_inchannels, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1] @dace.StorageType.GPU_Global ,
                kernel: dtype[d_outchannels, d_inchannels, d_kdim, d_kdim, d_kdim] @dace.StorageType.GPU_Global,
                Output: dtype[d_batchsize, d_outchannels, d_outdepth, d_outheight, d_outwidth] @dace.StorageType.GPU_Global):
    for oc, n, d, h, w in dace.map[0:d_outchannels, 0:d_batchsize, 0:d_outdepth, 0:d_outheight, 0:d_outwidth]:
        r_tmp = np.zeros([1], dtype=Input.dtype)
        for kd, kh, kw, ic in dace.map[0:d_kdim, 0:d_kdim, 0:d_kdim, 0:d_inchannels]:
            r_tmp = r_tmp + Input[n, ic, d+kd, h+kh, w+kw] * kernel[oc, ic, kd, kh, kw]
        Output[n, oc, d, h, w] = r_tmp

