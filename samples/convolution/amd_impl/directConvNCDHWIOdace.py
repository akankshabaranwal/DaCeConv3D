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

CTAtilex = 4
CTAtileOC = 2
kdim = 3

# Using inspiration from: https://arxiv.org/pdf/2012.15667.pdf
# Simple parallel 3D convolution. Direct convolution

# For layer 0: Original is 52 ms. With unrolling it goes down to 32 ms. 

@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=False)
def dace_conv3d( Input: dtype[d_batchsize, d_inchannels, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1] @dace.StorageType.GPU_Global ,
                kernel: dtype[d_outchannels, d_inchannels, d_kdim, d_kdim, d_kdim] @dace.StorageType.GPU_Global,
                Output: dtype[d_batchsize, d_outchannels, d_outdepth, d_outheight, d_outwidth] @dace.StorageType.GPU_Global):
    for b_d, b_h, b_w in dace.map[0:d_outdepth:CTAtilex, 0:d_outheight:CTAtilex, 0:d_outwidth:CTAtilex]@dace.ScheduleType.GPU_Device:
        for n, b_oc in dace.map[0:d_batchsize, 0:d_outchannels:CTAtileOC]@dace.ScheduleType.Sequential:
            cta_output = dace.ndarray([CTAtileOC, CTAtilex, CTAtilex, CTAtilex], dtype=Input.dtype, storage=dace.StorageType.GPU_Shared)
            for d, h, w in dace.map[0:CTAtilex, 0:CTAtilex, 0:CTAtilex]@dace.ScheduleType.GPU_ThreadBlock:
                for oc in dace.map[0:CTAtileOC]@dace.ScheduleType.Sequential:
                    cta_output[oc, d, h, w] = 0
            for ic in dace.map[0:d_inchannels]@dace.ScheduleType.Sequential:
                cta_input = dace.ndarray([CTAtilex, CTAtilex, CTAtilex], dtype=Input.dtype, storage=dace.StorageType.GPU_Shared)
                cta_kernel = dace.ndarray([CTAtileOC, kdim, kdim, kdim], dtype=Input.dtype, storage=dace.StorageType.GPU_Shared)

                for d, h, w in dace.map[0:CTAtilex, 0:CTAtilex, 0:CTAtilex]@dace.ScheduleType.GPU_ThreadBlock:
                    for oc, kd, kh, kw in dace.map[0:CTAtileOC, 0:d_kdim, 0:d_kdim, 0:d_kdim]@dace.ScheduleType.Sequential:
                        cta_input[d+kd, h+kh, w+kw] = Input[n, ic, b_d+d+kd, b_h+h+kh, b_w+w+kw]
                        cta_kernel[oc, kd, kh, kw] = kernel[oc+b_oc, ic, kd, kh, kw]
                for d, h, w in dace.map[0:CTAtilex, 0:CTAtilex, 0:CTAtilex]@dace.ScheduleType.GPU_ThreadBlock:
                    for oc, kd, kh, kw in dace.map[0:CTAtileOC, 0:d_kdim, 0:d_kdim, 0:d_kdim]@dace.ScheduleType.Sequential:
                        cta_output[oc, d, h, w] = cta_output[oc, d, h, w] + cta_input[d+kd, h+kh, w+kw] * cta_kernel[oc, kd, kh, kw]
            for d, h, w in dace.map[0:CTAtilex, 0:CTAtilex, 0:CTAtilex]@dace.ScheduleType.GPU_ThreadBlock:
                for oc in dace.map[0:CTAtileOC]@dace.ScheduleType.Sequential:
                    Output[n, oc+b_oc, b_d+d, b_h+h, b_w+w] = cta_output[oc, d, h, w]