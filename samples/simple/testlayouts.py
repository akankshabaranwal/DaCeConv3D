import dace
import numpy as np
from dace import dtypes
from convutils import find_map_by_param
import torch 
from daceml.testing.profiling import time_funcs, print_time_statistics
import torch.nn.functional as F

# Define symbolic sizes for arbitrary inputs
d_outdepth = dace.symbol('d_outdepth')
d_outheight = dace.symbol('d_outheight')
d_outwidth = dace.symbol('d_outwidth')
d_kdim = dace.symbol('d_kdim')
d_inchannels = dace.symbol('d_inchannels')
d_outchannels = dace.symbol('d_outchannels')
d_batchsize = dace.symbol('d_batchsize')

d_GEMM_M = dace.symbol('d_GEMM_M')
d_GEMM_N = dace.symbol('d_GEMM_N')
d_GEMM_K = dace.symbol('d_GEMM_K')

d_DHW = dace.symbol('d_DHW')
d_HW = dace.symbol('d_HW')

d_kdim3 = dace.symbol('d_kdim3')
d_kdim2 = dace.symbol('d_kdim2')

d_ncta_n = dace.symbol('d_ncta_n')
d_ncta_m = dace.symbol('d_ncta_m')
d_ncta_k = dace.symbol('d_ncta_k')

# Define data type to use
dtype = dace.float32
np_dtype = np.float32

def optimize_for_gpu(sdfg: dace.SDFG):
    """ Optimize 3D convolution example for GPUs. """
    print("INFO: Calling optimize for GPU function")
    dace.Config.set('compiler', 'default_data_types', value='C')

    return

CTAtileM = 64
CTAtileN = 16

CTAtileK = 1

WARPtileM = 2
WARPtileN = 16

@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=False)
def dace_conv3d(Input: dtype[d_inchannels, d_batchsize, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1] @dace.StorageType.GPU_Global,
                kernel: dtype[d_inchannels, d_outchannels,  d_kdim, d_kdim, d_kdim] @dace.StorageType.GPU_Global,
                Output: dtype[d_outchannels, d_batchsize,  d_outdepth, d_outheight, d_outwidth] @dace.StorageType.GPU_Global):
    
    d_GEMM_M = (d_batchsize*d_outdepth*d_outheight*d_outwidth)
    d_GEMM_N = d_outchannels
    d_GEMM_K = (d_inchannels * d_kdim * d_kdim * d_kdim)
    d_DHW = d_outdepth*d_outheight*d_outwidth
    d_HW = d_outheight*d_outwidth
    d_kdim3 = d_kdim*d_kdim*d_kdim
    d_kdim2 = d_kdim*d_kdim

    for cta_n, cta_m in dace.map[0:d_GEMM_N:CTAtileN, 0:d_GEMM_M:CTAtileM] @dace.ScheduleType.GPU_Device:
            cta_reducedk = dace.ndarray([CTAtileN, CTAtileM], dtype=Input.dtype, storage=dace.StorageType.GPU_Shared)
            for warp_n, warp_m in dace.map[0:CTAtileN:WARPtileN, 0:CTAtileM:WARPtileM]@dace.ScheduleType.GPU_ThreadBlock:
                for gemm_n, gemm_m in dace.map[0:WARPtileN, 0:WARPtileM]@dace.ScheduleType.Sequential:
                        cta_reducedk[warp_n+gemm_n, warp_m+gemm_m] = 0
            cta_input = dace.ndarray([CTAtileM], dtype=Input.dtype, storage=dace.StorageType.GPU_Shared)
            cta_kernel = dace.ndarray([CTAtileN], dtype=Input.dtype, storage=dace.StorageType.GPU_Shared)
            for cta_k in dace.map[0:d_GEMM_K]@dace.ScheduleType.Sequential:
                for warp_n, warp_m in dace.map[0:CTAtileN:WARPtileN, 0:CTAtileM:WARPtileM]@dace.ScheduleType.GPU_ThreadBlock:
                        for gemm_n, gemm_m in dace.map[0:WARPtileN, 0:WARPtileM]@dace.ScheduleType.Sequential:
                            n =  dace.int32((gemm_m+cta_m+warp_m)/d_DHW)
                            nopq_residual =  dace.int32((gemm_m+cta_m+warp_m) % d_DHW)

                            o = dace.int32(nopq_residual/d_HW)
                            opq_residual = dace.int32(nopq_residual%d_HW)
                            
                            p = dace.int32(opq_residual/d_outwidth)
                            q = dace.int32(opq_residual%d_outwidth)

                            c  = dace.int32((cta_k)/d_kdim3)
                            ctrs_residual  = dace.int32((cta_k)%d_kdim3)
                            
                            t = dace.int32(ctrs_residual/d_kdim2)
                            trs_residual = dace.int32(ctrs_residual%d_kdim2)
                            
                            r = dace.int32(trs_residual/d_kdim)
                            s = dace.int32(trs_residual%d_kdim)
                            
                            d = o + t
                            h = p + r
                            w = q + s

                            cta_input[warp_m + gemm_m] = Input[c, n, d, h, w]
                            cta_kernel[warp_n + gemm_n] = kernel[c, gemm_n+cta_n+warp_n, t, r, s]

                for warp_n, warp_m in dace.map[0:CTAtileN:WARPtileN, 0:CTAtileM:WARPtileM]@dace.ScheduleType.GPU_ThreadBlock:
                        warp_input = dace.ndarray([WARPtileM], dtype=Input.dtype, storage=dace.StorageType.Register)
                        warp_kernel = dace.ndarray([WARPtileN], dtype=Input.dtype, storage=dace.StorageType.Register)
                        for gemm_n, gemm_m in dace.map[0:WARPtileN, 0:WARPtileM]@dace.ScheduleType.Sequential:
                            warp_input[gemm_m] = cta_input[warp_m + gemm_m]
                            warp_kernel[gemm_n] = cta_kernel[warp_n + gemm_n]

                        for gemm_n, gemm_m in dace.map[0:WARPtileN, 0:WARPtileM]@dace.ScheduleType.Sequential:
                            cta_reducedk[gemm_n+warp_n, warp_m+gemm_m] = cta_reducedk[gemm_n+warp_n, warp_m+gemm_m] + warp_input[gemm_m]*warp_kernel[gemm_n]

            for warp_n, warp_m in dace.map[0: CTAtileN:WARPtileN, 0: CTAtileM:WARPtileM]@dace.ScheduleType.GPU_ThreadBlock:
                for gemm_n, gemm_m in dace.map[0:WARPtileN, 0:WARPtileM]@dace.ScheduleType.Sequential:

                    n =  dace.int32((gemm_m+cta_m+warp_m)/d_DHW)
                    nopq_residual = dace.int32((cta_m+gemm_m+warp_m) % d_DHW)
                    
                    o = dace.int32(nopq_residual/d_HW)
                    opq_residual = dace.int32(nopq_residual%d_HW)        
                    
                    p = dace.int32(opq_residual/d_outwidth)
                    q = dace.int32(opq_residual%d_outwidth)

                    Output[ cta_n+gemm_n+warp_n, n, o, p, q ] = cta_reducedk[gemm_n+warp_n, gemm_m+warp_m]


batchsize = 1

in_depth = 514
in_height = 514
in_width = 514

kdim = 3

out_depth = in_depth - kdim + 1
out_height = in_depth - kdim + 1
out_width = in_depth - kdim + 1

inchannels = 1
outchannels = 16

d_input = torch.rand(batchsize, inchannels, in_depth, in_height, in_width).cuda()
d_kernel = torch.rand(outchannels, inchannels, kdim, kdim, kdim).cuda()

t_input = torch.rand(batchsize, inchannels, in_depth, in_height, in_width).cuda()
t_kernel = torch.rand(outchannels, inchannels, kdim, kdim, kdim).cuda()

d_output = torch.rand(batchsize, outchannels, out_depth, out_height, out_width).cuda()

sdfg_fun: dace.SDFG = dace_conv3d.to_sdfg(d_input, d_kernel, d_output)
optimize_for_gpu(sdfg_fun)
optim_dace = sdfg_fun.compile()

def run_optim_dace():
    optim_dace(Input=d_input, kernel=d_kernel, Output=d_output,
            d_inchannels=inchannels, d_outdepth=out_depth, d_outheight=out_height,d_outwidth=out_width, 
            d_outchannels=outchannels, d_batchsize=batchsize, d_kdim=kdim)

def run_torch():
     ref_op = F.conv3d(t_input, t_kernel, stride=1, padding='valid')
     return ref_op


# times = time_funcs([run_optim_dace, run_torch],
#                 func_names=["dace", "torch"],
#                 warmups=10,
#                 num_iters=100,
#                 launch_wait=False)
# print_time_statistics(times, [ "dace", "torch"])

times = time_funcs([run_optim_dace],
                func_names=["dace"],
                warmups=10,
                num_iters=100,
                launch_wait=False)
print_time_statistics(times, [ "dace"])

times = time_funcs([run_torch],
                func_names=["torch"],
                warmups=10,
                num_iters=100,
                launch_wait=False)
print_time_statistics(times, [ "torch"])