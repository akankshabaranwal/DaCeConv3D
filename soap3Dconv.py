#from soaptest import soap_analysis
import dace 
import numpy as np
from dace import dtypes
import torch
from dace.transformation.estimator.soap.io_analysis import perform_soap_analysis
from dace.transformation.estimator.soap.utils import get_lead_term

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
def dace_conv3D( Input: dtype[d_batchsize, d_inchannels, d_outdepth+d_kdim-1, d_outheight+d_kdim-1, d_outwidth+d_kdim-1] @dace.StorageType.GPU_Global ,
                kernel: dtype[d_outchannels, d_inchannels, d_kdim, d_kdim, d_kdim] @dace.StorageType.GPU_Global,
                Output: dtype[d_batchsize, d_outchannels, d_outdepth, d_outheight, d_outwidth] @dace.StorageType.GPU_Global):
    for n, d, h, w, oc, kd, kh, kw, ic in dace.map[0:d_batchsize, 0:d_outdepth, 0:d_outheight, 0:d_outwidth, 0:d_outchannels, 0:d_kdim, 0:d_kdim, 0:d_kdim, 0:d_inchannels]:
        Output[n, oc, d, h, w] = Output[n, oc, d, h, w] + Input[n, ic, d+kd, h+kh, w+kw] * kernel[oc, ic, kd, kh, kw]

d_input = torch.rand(batchsize, inchannels, indepth, inheight, inwidth).cuda()
d_kernel = torch.rand(outchannels, inchannels, kdim, kdim, kdim).cuda()
d_output = torch.zeros(batchsize, outchannels, outdepth, outheight, outwidth).cuda()

sdfg_conv3D: dace.SDFG = dace_conv3D.to_sdfg(d_input, d_kernel, d_output)
sdfg_conv3D.apply_gpu_transformations()
sdfg_conv3D(Input=d_input, kernel=d_kernel, Output=d_output, 
d_inchannels = inchannels, d_batchsize = batchsize, d_outchannels = outchannels, d_outdepth = outdepth, d_outheight=outheight, d_outwidth = outwidth, d_kdim = kdim)
result = perform_soap_analysis(sdfg_conv3D, generate_schedule=False, solver_timeout=60)
print("Result printing starts")
print(result)
print("Result printing ends")