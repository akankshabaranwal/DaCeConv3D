# Sample code to test out convolution on AMD GPU.
import dace
from implicitGemmNCDHWdace import *
import torch
from convutils import parsecsv
import torch.nn.functional as F
import torch.cuda
from daceml.testing.profiling import time_funcs, print_time_statistics

paramscsv = 'cosmoflow'
convparams =  parsecsv(paramscsv)
currconv = convparams.iloc[2]

inchannels = currconv["InChannel"]
indepth = currconv["InputDepth"]
inheight = currconv["InputHeight"]
inwidth = currconv["InputWidth"]
outchannels = currconv["OutputChannel"]
kdim = 3
outdepth = indepth - kdim + 1
outheight = inheight - kdim + 1
outwidth = inheight - kdim + 1
batchsize = 16

d_input = torch.rand(batchsize, inchannels, indepth, inheight, inwidth).cuda()
d_kernel = torch.rand(outchannels, inchannels, kdim, kdim, kdim).cuda()
d_output = torch.zeros(batchsize, outchannels, outdepth, outheight, outwidth).cuda()
ref_op = torch.ones(batchsize, outchannels, outdepth, outheight, outwidth).cuda()

t_input = d_input.clone()
t_kernel = d_kernel.clone()

#sdfg_fun: dace.SDFG = dace_conv3d.to_sdfg(d_input, d_kernel, d_output)
#optimize_for_gpu(sdfg_fun)
#optim_dace = sdfg_fun.compile()
from dace.sdfg.utils import load_precompiled_sdfg

optim_dace = load_precompiled_sdfg(f'/users/abaranwa/amdoutput/.dacecache/implicitGemmNCDHWdace_dace_conv3d')

optim_dace(Input=d_input, kernel=d_kernel, Output=d_output,
            d_inchannels=inchannels, d_outdepth=outdepth, d_outheight=outheight, d_outwidth=outwidth, 
            d_outchannels=outchannels, d_batchsize=batchsize, d_kdim=kdim)

ref_op = F.conv3d(t_input, t_kernel, stride=1, padding='valid')
diff = np.linalg.norm((d_output.cpu() - ref_op.cpu())) / (batchsize * outchannels * outdepth * outheight * outwidth )
print('Difference between torch and dace values:', diff)

def run_optim_dace():
    optim_dace(Input=d_input, kernel=d_kernel, Output=d_output,
            d_inchannels=inchannels, d_outdepth=outdepth, d_outheight=outheight,d_outwidth=outwidth, 
            d_outchannels=outchannels, d_batchsize=batchsize, d_kdim=kdim)

def run_torch():
    ref_op = F.conv3d(t_input, t_kernel, stride=1, padding='valid')

warmupiter = 5
totaliter = 20

times = time_funcs([run_optim_dace, run_torch],
                            func_names=["dace", "torch"],
                            warmups=warmupiter,
                            num_iters=totaliter)
print_time_statistics(times, ["dace", "torch"])