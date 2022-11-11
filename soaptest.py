import dace
import numpy as np
from dace import dtypes
import torch

from dace.transformation.estimator.soap.io_analysis import perform_soap_analysis
from dace.transformation.estimator.soap.utils import get_lead_term
import sympy
import torch.nn.functional as F

# Launch matlab using: /home/akanksha/bin/matlab -nosplash -r "cd '/home/akanksha/Downloads/matlab'; BackgroundSolver();exit"
def soap_analysis(sdfg: dace.SDFG):
    result = perform_soap_analysis(sdfg, generate_schedule=False, solver_timeout=60)
    print("Result printing starts")
    print(result)
    print("Result printing ends")
    Q = get_lead_term(result.Q)

    # "Ss": max elements in fast memory
    # Example values! Iterate over arrays in SDFG to determine data type
    bytes_per_element = 4.0
    cache_size = 1024 * 1024
    num_elements = int(cache_size / bytes_per_element)
    
    # SOAP messes with the symbols in the SDFG, e.g., changes the case
    symbol_map = {"Ss": num_elements}
    symbol_map['d_outheight'] = outheight
    symbol_map['d_outwidth'] = outwidth
    symbol_map['d_outdepth'] = outdepth
    symbol_map['d_inchannels'] = inchannels
    symbol_map['d_outchannels'] = outchannels
    symbol_map['d_batchsize'] = batchsize
    symbol_map['d_kdim'] = kdim
    for sym in Q.free_symbols:
        print(sym)
        if str(sym) in sdfg.constants:
            symbol_map[sym] = sdfg.constants[str(sym)]
            continue

        s = str(sym).upper()
        if s in sdfg.constants:
            symbol_map[sym] = sdfg.constants[s]
    
    print(f"AB: symbol map is: {symbol_map}")
    # Now: symbol map contains all known symbol values
    # Try to get the actual value
    print(f"AB: Q is {Q}")
    simplified_Q = sympy.simplify(Q, symbols=symbol_map)
    Q_ = dace.symbolic.evaluate(simplified_Q, symbols=symbol_map)
    return Q_

d_outdepth = dace.symbol('d_outdepth')
d_outheight = dace.symbol('d_outheight')
d_outwidth = dace.symbol('d_outwidth')
d_inchannels = dace.symbol('d_inchannels')
d_outchannels = dace.symbol('d_outchannels')
d_batchsize = dace.symbol('d_batchsize')
d_kdim = dace.symbol('d_kdim')

dtype = dace.float32
np_dtype = np.float32

inchannels = 4
indepth = 128
inheight = 128
inwidth = 128
outchannels = 16
kdim = 3
outdepth = indepth - kdim + 1
outheight = inheight - kdim + 1
outwidth = inheight - kdim + 1
batchsize = 4
dace.Config.set('compiler', 'default_data_types', value='C')


@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=True)
def dace_conv3d( Input: dtype[d_batchsize, d_inchannels, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1] @dace.StorageType.GPU_Global ,
                kernel: dtype[d_outchannels, d_inchannels, d_kdim, d_kdim, d_kdim] @dace.StorageType.GPU_Global,
                Output: dtype[d_batchsize, d_outchannels, d_outdepth, d_outheight, d_outwidth] @dace.StorageType.GPU_Global):
    for n, d, h, w, oc in dace.map[0:d_batchsize, 0:d_outdepth, 0:d_outheight, 0:d_outwidth, 0:d_outchannels]:
        r_tmp = np.zeros([1], dtype=Input.dtype)
        for kd, kh, kw, ic in dace.map[0:d_kdim, 0:d_kdim, 0:d_kdim, 0:d_inchannels]:
            r_tmp = r_tmp + Input[n, ic, d+kd, h+kh, w+kw] * kernel[oc, ic, kd, kh, kw]
        Output[n, oc, d, h, w] = r_tmp

@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=True)
def dace_conv3d_naive( Input: dtype[d_batchsize, d_inchannels, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1] @dace.StorageType.GPU_Global ,
                kernel: dtype[d_outchannels, d_inchannels, d_kdim, d_kdim, d_kdim] @dace.StorageType.GPU_Global,
                Output: dtype[d_batchsize, d_outchannels, d_outdepth, d_outheight, d_outwidth] @dace.StorageType.GPU_Global):
    for n, d, h, w, oc, kd, kh, kw, ic in dace.map[0:d_batchsize, 0:d_outdepth, 0:d_outheight, 0:d_outwidth, 0:d_outchannels, 0:d_kdim, 0:d_kdim, 0:d_kdim, 0:d_inchannels]:
        Output[n, oc, d, h, w] = Output[n, oc, d, h, w] + Input[n, ic, d+kd, h+kh, w+kw] * kernel[oc, ic, kd, kh, kw]

d_input_conv3d = torch.rand(batchsize, inchannels, indepth, inheight, inwidth).cuda()
d_kernel_conv3d = torch.rand(outchannels, inchannels, kdim, kdim, kdim).cuda()
d_output_conv3d = torch.zeros(batchsize, outchannels, outdepth, outheight, outwidth).cuda()
t_input = d_input_conv3d.clone()
t_kernel = d_kernel_conv3d.clone()
sdfg_fun_conv3d: dace.SDFG = dace_conv3d.to_sdfg(d_input_conv3d, d_kernel_conv3d, d_output_conv3d)
sdfg_fun_conv3d.apply_gpu_transformations()
sdfg_fun_conv3d(Input=d_input_conv3d, kernel=d_kernel_conv3d, Output=d_output_conv3d, 
             d_inchannels = inchannels, d_batchsize = batchsize, d_outchannels = outchannels,
             d_outdepth = outdepth, d_outheight = outheight, d_outwidth = outwidth, 
             d_kdim = kdim)

# dace_output = d_output_conv3d.cpu()
# print("dace output computed")
# refop = F.conv3d(t_input, t_kernel, stride=1, padding='valid')
# print("pytorch output computed")
# refop = refop.cpu()
# diff = np.linalg.norm(refop - dace_output) / (batchsize * outchannels * outdepth * outheight * outwidth )
# print(f"Verif for 3D conv:{diff}")
# # Pick the code from conv3D directly. 
return_Q_conv3D = soap_analysis(sdfg_fun_conv3d)
print(f'For 3D convolution leading order terms are: {return_Q_conv3D}')


@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=True)
def dace_conv2d( Input: dtype[d_batchsize, d_inchannels, d_outheight+d_kdim-1, d_outwidth+d_kdim-1] @dace.StorageType.GPU_Global ,
                kernel: dtype[d_outchannels, d_inchannels, d_kdim, d_kdim] @dace.StorageType.GPU_Global,
                Output: dtype[d_batchsize, d_outchannels, d_outheight, d_outwidth] @dace.StorageType.GPU_Global):
    for n, h, w, oc in dace.map[0:d_batchsize, 0:d_outheight, 0:d_outwidth, 0:d_outchannels]:
        r_tmp = np.zeros([1], dtype=Input.dtype)
        for kh, kw, ic in dace.map[0:d_kdim, 0:d_kdim, 0:d_inchannels]:
            r_tmp = r_tmp + Input[n, ic, h+kh, w+kw] * kernel[oc, ic, kh, kw]
        Output[n, oc, h, w] = r_tmp

@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=True)
def dace_conv2d_naive( Input: dtype[d_batchsize, d_inchannels, d_outheight+d_kdim-1, d_outwidth+d_kdim-1] @dace.StorageType.GPU_Global ,
                kernel: dtype[d_outchannels, d_inchannels, d_kdim, d_kdim] @dace.StorageType.GPU_Global,
                Output: dtype[d_batchsize, d_outchannels, d_outheight, d_outwidth] @dace.StorageType.GPU_Global):
    for n, h, w, oc, kh, kw, ic in dace.map[0:d_batchsize, 0:d_outheight, 0:d_outwidth, 0:d_outchannels, 0:d_kdim, 0:d_kdim, 0:d_inchannels]:
        Output[n, oc, h, w] = Output[n, oc, h, w] + Input[n, ic, h+kh, w+kw] * kernel[oc, ic, kh, kw]

d_input_conv2d = torch.rand(batchsize, inchannels, inheight, inwidth).cuda()
d_kernel_conv2d = torch.rand(outchannels, inchannels, kdim, kdim).cuda()
d_output_conv2d = torch.zeros(batchsize, outchannels, outheight, outwidth).cuda()
t_input = d_input_conv2d.clone()
t_kernel = d_kernel_conv2d.clone()
sdfg_fun_conv2d: dace.SDFG = dace_conv2d.to_sdfg(d_input_conv2d, d_kernel_conv2d, d_output_conv2d)
sdfg_fun_conv2d.apply_gpu_transformations()
sdfg_fun_conv2d(Input=d_input_conv2d, kernel=d_kernel_conv2d, Output=d_output_conv2d, 
             d_inchannels = inchannels, d_batchsize = batchsize, d_outchannels = outchannels,
             d_outheight = outheight, d_outwidth = outwidth, 
             d_kdim = kdim)
# dace_output = d_output_conv2d.cpu()
# print("dace output computed")
# refop = F.conv2d(t_input, t_kernel, stride=1, padding='valid')
# print("pytorch output computed")
# refop = refop.cpu()
# diff = np.linalg.norm(refop - dace_output) / (batchsize * outchannels * outheight * outwidth )
# print(f"Verif for 2D conv:{diff}")
# # # Pick the code from conv3D directly. 
return_Q_conv2D = soap_analysis(sdfg_fun_conv2d)
print(f'For 2D convolution leading order terms are: {return_Q_conv2D}')


@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=True)
def dace_conv1d( Input: dtype[d_batchsize, d_inchannels, d_outwidth+d_kdim-1] @dace.StorageType.GPU_Global ,
                kernel: dtype[d_outchannels, d_inchannels, d_kdim] @dace.StorageType.GPU_Global,
                Output: dtype[d_batchsize, d_outchannels, d_outwidth] @dace.StorageType.GPU_Global):
    for n, w, oc in dace.map[0:d_batchsize, 0:d_outwidth, 0:d_outchannels]:
        r_tmp = np.zeros([1], dtype=Input.dtype)
        for kw, ic in dace.map[0:d_kdim, 0:d_inchannels]:
            r_tmp = r_tmp + Input[n, ic, w+kw] * kernel[oc, ic, kw]
        Output[n, oc, w] = r_tmp

@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=True)
def dace_conv1d_naive( Input: dtype[d_batchsize, d_inchannels, d_outwidth+d_kdim-1] @dace.StorageType.GPU_Global ,
                kernel: dtype[d_outchannels, d_inchannels, d_kdim] @dace.StorageType.GPU_Global,
                Output: dtype[d_batchsize, d_outchannels, d_outwidth] @dace.StorageType.GPU_Global):
    for n, w, oc, kw, ic in dace.map[0:d_batchsize, 0:d_outwidth, 0:d_outchannels,0:d_kdim, 0:d_inchannels]:
        Output[n, oc, w] = Output[n, oc, w] + Input[n, ic, w+kw] * kernel[oc, ic, kw]

d_input_conv1d = torch.rand(batchsize, inchannels, inwidth).cuda()
d_kernel_conv1d = torch.rand(outchannels, inchannels, kdim).cuda()
d_output_conv1d = torch.zeros(batchsize, outchannels, outwidth).cuda()
t_input = d_input_conv1d.clone()
t_kernel = d_kernel_conv1d.clone()
sdfg_fun_conv1d: dace.SDFG = dace_conv1d.to_sdfg(d_input_conv1d, d_kernel_conv1d, d_output_conv1d)
sdfg_fun_conv1d.apply_gpu_transformations()
sdfg_fun_conv1d(Input=d_input_conv1d, kernel=d_kernel_conv1d, Output=d_output_conv1d, 
             d_inchannels = inchannels, d_batchsize = batchsize, d_outchannels = outchannels,
             d_outwidth = outwidth, 
             d_kdim = kdim)
# dace_output = d_output_conv1d.cpu()
# print("dace output computed")
# refop = F.conv1d(t_input, t_kernel, stride=1, padding='valid')
# print("pytorch output computed")
# refop = refop.cpu()
# diff = np.linalg.norm(refop - dace_output) / (batchsize * outchannels * outwidth )
# print(f"Verif for 1D conv:{diff}")
# # Pick the code from conv3D directly. 
return_Q_conv1D = soap_analysis(sdfg_fun_conv1d)
print(f'For 1D convolution leading order terms are: {return_Q_conv1D}')
print("All results computed")
print(f'For 1D convolution leading order terms are: {return_Q_conv1D}')
print(f'For 2D convolution leading order terms are: {return_Q_conv2D}')
print(f'For 3D convolution leading order terms are: {return_Q_conv3D}')