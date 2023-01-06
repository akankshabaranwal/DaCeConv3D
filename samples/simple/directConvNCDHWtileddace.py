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


# # This interpretation is not very useful imo. Because the outdepth is very less in some cases. And then this limits the tile size we can set to
# # So I am instead interpreting the soap output as CTAtilD to CTAtileDHW and the rest is as it is
# CTAtileD = 4
# CTAtileOC = 4

# @dace.program(device=dtypes.DeviceType.GPU, auto_optimize=False)
# def dace_conv3d( Input: dtype[d_batchsize,  d_inchannels, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1] @dace.StorageType.GPU_Global ,
#                 kernel: dtype[d_outchannels, d_inchannels, d_kdim, d_kdim, d_kdim] @dace.StorageType.GPU_Global,
#                 Output: dtype[d_batchsize, d_outchannels, d_outdepth, d_outheight, d_outwidth] @dace.StorageType.GPU_Global):
    
#     for cta_d, cta_oc in dace.map[0:d_outdepth:CTAtileD, 0:d_outchannels:CTAtileOC]@dace.ScheduleType.GPU_Device:
#         for n, d, h, w, oc in dace.map[0:d_batchsize, 0:CTAtileD, 0:d_outheight, 0:d_outwidth, 0:CTAtileOC]:
#             r_tmp = np.zeros([1], dtype=Input.dtype)
#             for kd, kh, kw, ic in dace.map[0:d_kdim, 0:d_kdim, 0:d_kdim, 0:d_inchannels]@dace.ScheduleType.Sequential:
#                 r_tmp = r_tmp + Input[n, ic, cta_d+d+kd, h+kh, w+kw] * kernel[cta_oc+oc, ic, kd, kh, kw]
#             Output[n, oc+cta_oc, d, h, w] = r_tmp

CTAtileDHW = 64 # This should be min [sqrt(Ss), dhw] 64, 32, 16 works for layers 0 to 5; 8 works for layer 6
CTAtileOC = 16 # This should be min [sqrt(Ss), OC] 16 works for layers 0 to 7; 32 doesn't work for anything
CTAtileIC = 2 # The formula for this from soap analysis is 81*d_batchsize*d_inchannels*d_outchannels*d_outdepth*d_outheight*d_outwidth/(Ss*p)

# This is also limited by IC, especially for the first layer.
#WARPtileDHW = 1
#WARPtileOC = 1
#WARPtileIC = 1

batchsize = 16
kdim = 3

# Maybe the batch size should also have an outermost dace.map . Not sure why the soap analysis script ignores this
@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=False)
def dace_conv3d( Input: dtype[d_batchsize,  d_inchannels, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1] @dace.StorageType.GPU_Global ,
                kernel: dtype[d_outchannels, d_inchannels, d_kdim, d_kdim, d_kdim] @dace.StorageType.GPU_Global,
                Output: dtype[d_batchsize, d_outchannels, d_outdepth, d_outheight, d_outwidth] @dace.StorageType.GPU_Global):
    
    d_DHW = d_outdepth*d_outheight*d_outwidth
    d_HW = d_outheight*d_outwidth

    # for cta_n, cta_oc, cta_dhw in dace.map[0:d_batchsize, 0:d_outchannels:CTAtileOC, 0:d_DHW:CTAtileDHW]@dace.ScheduleType.GPU_Device:
        
    #     cta_output = dace.ndarray([CTAtileOC, CTAtileDHW], dtype=Input.dtype, storage=dace.StorageType.GPU_Shared)
    #     for dhw, oc in dace.map[0:CTAtileDHW, 0:CTAtileOC]@dace.ScheduleType.Sequential:
    #             cta_output[oc, dhw] = 0

    #     for cta_ic in dace.map[0:d_inchannels:CTAtileIC]@dace.ScheduleType.Sequential:            
            
    #         cta_input = dace.ndarray([CTAtileIC, CTAtileDHW, kdim, kdim, kdim], dtype=Input.dtype, storage=dace.StorageType.GPU_Shared)
    #         cta_kernel = dace.ndarray([CTAtileOC, CTAtileIC, kdim, kdim, kdim], dtype=Input.dtype, storage=dace.StorageType.GPU_Shared)

    #         for dhw, oc in dace.map[0:CTAtileDHW, 0:CTAtileOC]@dace.ScheduleType.Sequential:
    #             d, dhw_residual = dace.int32((cta_dhw+dhw)/d_HW), dace.int32((cta_dhw+dhw)%d_HW)
    #             h, w = dace.int32(dhw_residual/d_outheight), dace.int32(dhw_residual%d_outheight)
    #             for ic, kd, kh, kw in dace.map[0:CTAtileIC, 0:d_kdim, 0:d_kdim, 0:d_kdim]@dace.ScheduleType.Sequential:
    #                 cta_input[ic, dhw, kd, kh, kw] = Input[ cta_n, ic+cta_ic, d+kd, h+kh, w+kw]
    #                 cta_kernel[oc, ic, kd, kh, kw] = kernel[cta_oc+oc, ic+cta_ic, kd, kh, kw]

    #         for dhw, oc in dace.map[0:CTAtileDHW, 0:CTAtileOC]@dace.ScheduleType.Sequential:
    #             for ic, kd, kh, kw in dace.map[0:CTAtileIC, 0:d_kdim, 0:d_kdim, 0:d_kdim]@dace.ScheduleType.Sequential:
    #                 cta_output[oc, dhw] = cta_output[oc, dhw] + cta_input[ic, dhw, kd, kh, kw] * cta_kernel[oc, ic, kd, kh, kw]

    #     for dhw, oc in dace.map[0:CTAtileDHW, 0:CTAtileOC]@dace.ScheduleType.Sequential:
    #         d, dhw_residual = dace.int32((cta_dhw+dhw)/d_HW), dace.int32((cta_dhw+dhw)%d_HW)
    #         h, w = dace.int32(dhw_residual/d_outheight), dace.int32(dhw_residual%d_outheight)
    #         for ic, kd, kh, kw in dace.map[0:CTAtileIC, 0:d_kdim, 0:d_kdim, 0:d_kdim]@dace.ScheduleType.Sequential:
    #             Output[cta_n, cta_oc+oc, d, h, w] = cta_output[oc, dhw]

    for n, cta_dhw, cta_oc in dace.map[0:d_batchsize, 0:d_DHW:CTAtileDHW, 0:d_outchannels:CTAtileOC]@dace.ScheduleType.GPU_Device:
        for ic in 
        for dhw, oc in dace.map[0:CTAtileDHW, 0:CTAtileOC]@dace.ScheduleType.GPU_ThreadBlock:
            d, dhw_residual = dace.int32((cta_dhw+dhw)/d_HW), dace.int32((cta_dhw+dhw)%d_HW)
            h, w = dace.int32(dhw_residual/d_outheight), dace.int32(dhw_residual%d_outheight)
            for ic, kd, kh, kw in dace.map[0:d_inchannels, 0:d_kdim, 0:d_kdim, 0:d_kdim]@dace.ScheduleType.Sequential:
                Output[n, cta_oc+oc, d, h, w] = Output[n, cta_oc+oc, d, h, w] + Input[ n, ic, d+kd, h+kh, w+kw]*kernel[cta_oc+oc, ic, kd, kh, kw]