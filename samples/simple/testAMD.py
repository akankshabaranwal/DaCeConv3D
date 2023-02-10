# Sample code to test out convolution on AMD GPU.
import dace
from directConvNCDHWdace import *
import torch
from convutils import parsecsv

paramscsv = 'cosmoflow'
convparams =  parsecsv(paramscsv)
currconv = convparams.iloc[4]

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

sdfg_fun: dace.SDFG = dace_conv3d.to_sdfg(d_input, d_kernel, d_output)
optimize_for_gpu(sdfg_fun)
optim_dace = sdfg_fun.compile()

optim_dace(Input=d_input, kernel=d_kernel, Output=d_output,
            d_inchannels=inchannels, d_outdepth=outdepth, d_outheight=outheight, d_outwidth=outwidth, 
            d_outchannels=outchannels, d_batchsize=batchsize, d_kdim=kdim)