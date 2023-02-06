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

# CTAtileDHW = 64 # This should divide outdepth, outheight, outwidth individually otherwise the indexing gets tricky.
# CTAtileOC = 1
# inchannels = 1
# kdim = 3
# # Since cta_kernel is a small kernel, it makes sense to keep it in the shared memory instead of loading it multiple times
# # Input and Outputs are used only once so you don't need to load it to the shared memory.
# @dace.program(device=dtypes.DeviceType.GPU, auto_optimize=False)
# def dace_conv3d( Input: dtype[d_batchsize,  d_inchannels, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1] @dace.StorageType.GPU_Global ,
#                 kernel: dtype[d_outchannels, d_inchannels, d_kdim, d_kdim, d_kdim] @dace.StorageType.GPU_Global,
#                 Output: dtype[d_batchsize, d_outchannels, d_outdepth, d_outheight, d_outwidth] @dace.StorageType.GPU_Global):
        
#     d_DHW = d_outdepth*d_outheight*d_outwidth
#     d_HW = d_outheight*d_outwidth
#     for cta_n, cta_dhw, cta_oc in dace.map[0:d_batchsize, 0:d_DHW:CTAtileDHW, 0:d_outchannels]@dace.ScheduleType.GPU_Device:
#         cta_shared = dace.ndarray([CTAtileDHW], dtype=Input.dtype, storage=dace.StorageType.GPU_Shared)
#         cta_input = dace.ndarray([CTAtileDHW, inchannels* kdim* kdim* kdim], dtype=Input.dtype,storage=dace.StorageType.GPU_Shared)
#         cta_kernel = dace.ndarray([inchannels*kdim*kdim*kdim], dtype=Input.dtype,storage=dace.StorageType.GPU_Shared)
        
#         for dhw in dace.map[ 0:CTAtileDHW]@dace.ScheduleType.GPU_ThreadBlock:
#             cta_shared[dhw] = 0
#             d, dhw_residual = dace.int32((dhw+cta_dhw)/d_HW), dace.int32((dhw+cta_dhw)%d_HW)
#             h, w = dace.int32(dhw_residual/d_outheight), dace.int32(dhw_residual%d_outheight)
#             for ic, kd, kh, kw in dace.map[0:d_inchannels, 0:d_kdim, 0:d_kdim, 0:d_kdim]@dace.ScheduleType.Sequential:
#                 cta_input[dhw, 27*ic+ 9*kd +3*kh + kw] = Input[ cta_n, ic, d+kd, h+kh, w+kw]
#                 cta_kernel[ 27*ic+ 9*kd+3*kh + kw] = kernel[cta_oc, ic, kd, kh, kw]

#         for dhw in dace.map[ 0:CTAtileDHW]@dace.ScheduleType.GPU_ThreadBlock:
#             d, dhw_residual = dace.int32((dhw+cta_dhw)/d_HW), dace.int32((dhw+cta_dhw)%d_HW)
#             h, w = dace.int32(dhw_residual/d_outheight), dace.int32(dhw_residual%d_outheight)
#             tmp = dace.ndarray([1], dtype=Input.dtype, storage=dace.StorageType.Register)
#             tmp = 0
#             for ic, kd, kh, kw in dace.map[0:d_inchannels, 0:d_kdim, 0:d_kdim, 0:d_kdim]@dace.ScheduleType.Sequential:
#                 tmp = tmp + cta_input[ dhw, 27*ic+ 9*kd +3*kh + kw]*cta_kernel[ 27*ic+ 9*kd+3*kh + kw]
#             cta_shared[dhw] = tmp

#         for dhw in dace.map[ 0:CTAtileDHW]@dace.ScheduleType.GPU_ThreadBlock:
#             d, dhw_residual = dace.int32((dhw+cta_dhw)/d_HW), dace.int32((dhw+cta_dhw)%d_HW)
#             h, w = dace.int32(dhw_residual/d_outheight), dace.int32(dhw_residual%d_outheight)                         
#             Output[cta_n, cta_oc, d, h, w] = cta_shared[dhw]


CTAtileDHW = 64 # This should be min [sqrt(Ss), dhw] 64, 32, 16 works for layers 0 to 5; 8 works for layer 6
WARPtileDHW = 2
# This is also limited by IC, especially for the first layer.
#WARPtileDHW = 4
#WARPtileOC = 2
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

    for n, cta_dhw, cta_oc in dace.map[0:d_batchsize, 0:d_DHW:CTAtileDHW, 0:d_outchannels]@dace.ScheduleType.GPU_Device:

        #cta_output = dace.ndarray([ CTAtileDHW], dtype=Input.dtype, storage=dace.StorageType.GPU_Shared)
        #cta_kernel = dace.ndarray([ 64, kdim, kdim, kdim], dtype=Input.dtype, storage=dace.StorageType.GPU_Shared)
        
        # for warp_dhw in dace.map[0:CTAtileDHW:WARPtileDHW]@dace.ScheduleType.GPU_ThreadBlock:
        #     for dhw in dace.map[0:WARPtileDHW]@dace.ScheduleType.Sequential:
        #         cta_output[dhw+warp_dhw] = 0

        # for cta_ic in dace.map[0:d_inchannels]@dace.ScheduleType.Sequential:
        #     for dhw in dace.map[0:CTAtileDHW]@dace.ScheduleType.GPU_ThreadBlock:
        #         for kd, kh, kw in dace.map[0:d_kdim, 0:d_kdim, 0:d_kdim]@dace.ScheduleType.Sequential:
        #             cta_kernel[cta_ic, kd, kh, kw] = kernel[cta_oc, cta_ic, kd, kh, kw]
        
        for cta_ic in dace.map[0:d_inchannels]@dace.ScheduleType.Sequential:
            #cta_input = dace.ndarray([ CTAtileDHW, kdim, kdim, kdim], dtype=Input.dtype, storage=dace.StorageType.GPU_Shared)
            

            #for dhw in dace.map[0:CTAtileDHW]@dace.ScheduleType.GPU_ThreadBlock:
            #        d, dhw_residual = dace.int32((cta_dhw+dhw)/d_HW), dace.int32((cta_dhw+dhw)%d_HW)
            #        h, w = dace.int32(dhw_residual/d_outheight), dace.int32(dhw_residual%d_outheight)
            #        for kd, kh, kw in dace.map[0:d_kdim, 0:d_kdim, 0:d_kdim]@dace.ScheduleType.Sequential:
                        #cta_input[dhw, kd, kh, kw] = Input[ n, cta_ic, d+kd, h+kh, w+kw]
                        #cta_kernel[kd, kh, kw] = kernel[cta_oc, cta_ic, kd, kh, kw]

            for warp_dhw in dace.map[0:CTAtileDHW:WARPtileDHW]@dace.ScheduleType.GPU_ThreadBlock:
                for dhw in dace.map[0:WARPtileDHW]@dace.ScheduleType.Sequential:
                    d, dhw_residual = dace.int32((cta_dhw+dhw+warp_dhw)/d_HW), dace.int32((cta_dhw+dhw+warp_dhw)%d_HW)
                    h, w = dace.int32(dhw_residual/d_outheight), dace.int32(dhw_residual%d_outheight)
                    for kd, kh, kw in dace.map[0:d_kdim, 0:d_kdim, 0:d_kdim]@dace.ScheduleType.Sequential:
                        Output[n, cta_oc, d, h, w] = Output[n, cta_oc, d, h, w] + Input[ n, cta_ic, d+kd, h+kh, w+kw]*kernel[cta_oc, cta_ic, kd, kh, kw]
                        #cta_output[dhw+warp_dhw] = cta_output[dhw+warp_dhw] + Input[ n, cta_ic, d+kd, h+kh, w+kw]*kernel[cta_oc, cta_ic, kd, kh, kw]
            
        # for warp_dhw in dace.map[0:CTAtileDHW:WARPtileDHW]@dace.ScheduleType.GPU_ThreadBlock:
        #     for dhw in dace.map[0:WARPtileDHW]@dace.ScheduleType.Sequential:
        #         d, dhw_residual = dace.int32((cta_dhw+dhw)/d_HW), dace.int32((cta_dhw+dhw)%d_HW)
        #         h, w = dace.int32(dhw_residual/d_outheight), dace.int32(dhw_residual%d_outheight)
        #         Output[n, cta_oc, d, h, w] = cta_output[dhw+warp_dhw]