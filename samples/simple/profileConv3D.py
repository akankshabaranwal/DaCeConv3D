# abaranwal: Code to profile 3D convolution based on daceml functions

from daceml.testing.profiling import time_funcs, print_time_statistics
import argparse
import os
import statistics
import sys
import glob

useCudnn = 0

if(useCudnn):
    import pycuda.autoinit
    from pycuda import gpuarray
    import libcudnn
    from cudnnConv import cudnn_init, cudnnsetlayerdesc, destroydescinoutfilt

from dace.sdfg.utils import load_precompiled_sdfg
import numpy as np
from convutils import prepareinputs, parsecsv, run_fun, createsummaryfile, createplots
import dace
import torch
import torch.nn.functional as F

import pandas as pd 

parser = argparse.ArgumentParser(description='Process the input arguments')
# Reference for dace code: https://spcldace.readthedocs.io/en/latest/general/debugging.html
# Make changes in the generated code, compile it using: cd .dacecache/<fun>/build; make;
# Use this to call: DACE_compiler_use_cache=1 python profileConv3D.py <...>

parser.add_argument('--loadprecompiled', action='store_true', help='load from a precompiled dace folder')

parser.add_argument('-paramscsv','--paramscsv', type=str, default='cosmoflow', help='select which csv to profile')
parser.add_argument('--verify', action='store_true', help='run verification')
parser.add_argument('--compareprof', action='store_true', help='time funcs comparative summary time')
parser.add_argument('--proftorch', action='store_true', help='run torch code with markers')
parser.add_argument('--profoptimdace', action='store_true', help='run dace code with markers')
parser.add_argument('--profcudnn', action='store_true', help='run cudnn code with markers')

parser.add_argument('--setlaunchwait', action='store_true', help='set launch wait')
parser.add_argument('--enableplots', action='store_true', help='disable creating plots')
parser.add_argument('-warmupiter','--warmupiter', type=int, default=10, help='set number of warmup iterations')
parser.add_argument('-totaliter','--totaliter', type=int, default=100, help='set number of total iterations')
parser.add_argument('-lastlayer','--lastlayer', type=int, default=1, help='set number of total iterations')
parser.add_argument('-currlayer','--currlayer', type=int, default=0, help='set number of total iterations')

parser.add_argument('-implementation','--implementation', type=str, default='implicitGemmNCDHWdace', help='select which implementation to run')

# TODO: Check if you can plot and compare different versions of dace optimizations
# TODO: Automate the roofline analysis plot
# FIXME: Something is wrong with subplot when the csv file has just one row  

args = parser.parse_args()

paramscsv = args.paramscsv
convparams =  parsecsv(paramscsv)
loadprecompiled = args.loadprecompiled
verify = args.verify
compareprof = args.compareprof
proftorch = args.proftorch
profdace = False
profoptimdace = args.profoptimdace
profcudnn = args.profcudnn
setlaunchwait = args.setlaunchwait
warmupiter = args.warmupiter
totaliter = args.totaliter
currlayer = args.currlayer
enableplots = args.enableplots
lastlayer = min(args.lastlayer, convparams.shape[0])

batchsizes = [16]

if (verify and compareprof):
    sys.exit("!!! ERROR: Some pycuda context issue when both verif and compareprof are called together")

torch.cuda.empty_cache()

# Set the dace implementation to run
selectMethod = args.implementation

if selectMethod == 'implicitGemmNCDHWmatmul':
    from implicitGemmNCDHWmatmul import *
    layout = 'NCDHW'
elif selectMethod == 'implicitGemmNCDHWdace':
    from implicitGemmNCDHWdace import *
    layout = 'NCDHW'
elif selectMethod == 'implicitGemmNCDHWsoap':
    from implicitGemmNCDHWsoap import *
    layout = 'NCDHW'
elif selectMethod == 'implicitGemmsplitKdace':
    from implicitGemmsplitKdace import *
    layout = 'NCDHW'
elif selectMethod == 'implicitGemmNoIndexdace':
    from implicitGemmNoIndexdace import *
    layout = 'NCDHW'
elif selectMethod == 'directConvNCDHWdace':
    from directConvNCDHWdace import *
    layout = 'NCDHW'
elif selectMethod == 'directConvNCDHWIOdace':
    from directConvNCDHWIOdace import *
    layout = 'NCDHW'
elif selectMethod == 'directConvNCDHWnobuffer': # Fast code with no buffers
    from directConvNCDHWnobuffer import *
    layout = 'NCDHW'
elif selectMethod == 'directConvNDHWCtileddace': # Slow code with explicit buffers
    from directConvNDHWCtileddace import *
    layout = 'NDHWC'
elif selectMethod == 'directConvNCDHWtileddace': # Code with naive merge and sdfg optimization
    from directConvNCDHWtileddace import *
    layout = 'NCDHW'
elif selectMethod == 'directConvNCDHWzerodace': # Code with naive merge and sdfg optimization
    from directConvNCDHWzerodace import *
    layout = 'NCDHW'
elif selectMethod == 'directConvNCDHWmergeddace': # Code with naive merge and sdfg optimization
    from directConvNCDHWmergeddace import *
    layout = 'NCDHW'
else:
    sys.exit("!!ERROR: Select valid dace implementation")

if(useCudnn and layout!='NCDHW'):
    sys.exit("!!ERROR: Pytorch supports only NCHDW layout")

outdir = f'./outputplots/out_{selectMethod}_{batchsizes[0]}'
if not os.path.exists(outdir):
    os.mkdir(outdir)
    
with open(f'./{outdir}/params.txt', 'w') as f:
    f.writelines(f'csv: {paramscsv}\n')
    f.writelines(f'warmup iteration: {warmupiter}\n')
    f.writelines(f'total iteration: {totaliter}\n')
    f.writelines(f'set launch wait: {setlaunchwait}\n')

args = parser.parse_args()
d_input, d_kernel, d_output, inchannels, indepth, inheight, inwidth, outchannels, batchsize, kdim = prepareinputs(convparams.iloc[0], layout, batchsizes[0])

## Prepare inputs for pytorch fun
t_input = d_input.clone()
t_kernel = d_kernel.clone()

inchannels = np.int32(inchannels)
indepth = np.int32(indepth)
inheight = np.int32(inheight)
inwidth = np.int32(inwidth)
outchannels = np.int32(outchannels)
batchsize = np.int32(batchsize)
outdepth = indepth - kdim + 1
outheight = inheight - kdim + 1
outwidth = inwidth - kdim + 1

pad = 0
dil = 1
stride = 1

if(useCudnn):
    # Initializing cudnn
    conv_desc, cudnn_context, tensor_format, convolution_mode, convolution_algo, alpha, beta, c_int_p, outdimsinit, data_type, tensor_dim, conv_dim = cudnn_init(pad, stride, dil, layout)
    # cudnn variable parameters init, these change across different layers and are called multiple times
    cudnn_input, cudnn_kernel, cudnn_output, in_desc, in_data, in_data_g, out_desc, out_data, out_data_g, outdims,  filt_desc, filt_data, filt_data_g, ws_ptr, ws_data, ws_size = cudnnsetlayerdesc(cudnn_context, outdimsinit, conv_desc, convolution_algo, d_input,  d_kernel, d_output, batchsize, kdim, inchannels, indepth, inheight, inwidth, outchannels, data_type, tensor_dim, tensor_format)

ref_op = F.conv3d(t_input, t_kernel, stride=1, padding='valid')

if loadprecompiled:

    optim_dace = load_precompiled_sdfg(f'/users/abaranwa/amdoutput/.dacecache/{selectMethod}_dace_conv3d')
else:    
    sdfg_fun: dace.SDFG = dace_conv3d.to_sdfg(d_input, d_kernel, d_output)
    optimize_for_gpu(sdfg_fun)
    #optim_dace = dace_conv3d
    optim_dace = sdfg_fun.compile()

# Function call for pytorch 3D conv
def run_torch():
    ref_op = F.conv3d(t_input, t_kernel, stride=1, padding='valid')
    return ref_op

if(useCudnn):
    def run_cudnn():
        libcudnn.cudnnConvolutionForward(cudnn_context, alpha, in_desc, in_data, filt_desc, filt_data, 
                                    conv_desc, convolution_algo, ws_data, ws_size.value, 
                                    beta, out_desc, out_data)

# Function calls to run the optim dace function
def run_optim_dace():
    optim_dace(Input=d_input, kernel=d_kernel, Output=d_output,
            d_inchannels=inchannels, d_outdepth=outdepth, d_outheight=outheight,d_outwidth=outwidth, 
            d_outchannels=outchannels, d_batchsize=batchsize, d_kdim=kdim)
    
median_dace = []
median_cudnn = []
median_torch = []
layer_names = []
if (useCudnn):
    csv_columns = ['layer_name','dace_median','cudnn_median']
else:
    csv_columns = ['layer_name','dace_median','torch_median']

summary = []

if (useCudnn):
    # Clearing cudnn variables
    in_desc, out_desc, filt_desc, ws_ptr = destroydescinoutfilt(in_desc, out_desc, filt_desc, ws_ptr)

for layern in range(currlayer, lastlayer):
    for batchsize in batchsizes:
        d_input, d_kernel, d_output, inchannels, indepth, inheight, inwidth, outchannels, batchsize, kdim = prepareinputs(convparams.iloc[layern], layout, batchsize)
        layersummary = {}
        t_input = d_input.clone()
        t_kernel = d_kernel.clone()

        inchannels = np.int32(inchannels)
        indepth = np.int32(indepth)
        inheight = np.int32(inheight)
        inwidth = np.int32(inwidth)
        outchannels = np.int32(outchannels)
        batchsize = np.int32(batchsize)
        outdepth = np.int32(indepth - kdim + 1)
        outheight = np.int32(inheight - kdim + 1)
        outwidth = np.int32(inwidth - kdim + 1)

        layer_name = f'in_{batchsize}X{inchannels}X{indepth}X{inheight}X{inwidth}_k_{kdim}X{kdim}X{kdim}_och_{outchannels}'
        print(f'INFO: {layout} layout {layer_name}')
        layer_names.append(layer_name)

        if (useCudnn):
            cudnn_input, cudnn_kernel, cudnn_output, in_desc, in_data, in_data_g, out_desc, out_data, out_data_g, outdims, filt_desc, filt_data, filt_data_g, ws_ptr, ws_data, ws_size = cudnnsetlayerdesc(cudnn_context, outdimsinit, conv_desc, convolution_algo, t_input,  t_kernel, t_output, batchsize, kdim, inchannels, indepth, inheight, inwidth, outchannels, data_type, tensor_dim, tensor_format)

        # Code for verification
        if verify:
            if(useCudnn):
                print("INFO: Running verification to compare against cudnn output")
            else:
                print("INFO: Running verification to compare against torch output")
            if(useCudnn):
                run_cudnn()
            else:
                ref_op = run_torch()

            run_optim_dace()

            if(useCudnn):
                d_output = d_output.cpu()
                dace_output_g = gpuarray.to_gpu(d_output.numpy().astype(np.float32))
                diff = np.linalg.norm((out_data_g - dace_output_g).get()) / (batchsize * outchannels * outdepth * outheight * outwidth )
                print('Difference between cudnn and dace values:', diff)
            else:
                diff = np.linalg.norm((d_output.cpu() - ref_op.cpu())) / (batchsize * outchannels * outdepth * outheight * outwidth )

            if(diff<=1e-4): #TODO: Check if the threshold should be reduced
                print(f"Verification successfull")
            else:
                sys.exit(f"!!! ERROR: Incorrect verification layer number {layern}")


        # Comparitive profiling using time funcs
        print(f"INFO: Warmup for {warmupiter} iterations and total iterations {totaliter}")
        print(f"INFO: Statistics for layer number {layern} with batch size of {batchsize}")
        dace_median = []
        torch_median = []
        cudnn_median = []

        if compareprof:
            if useCudnn:
                times = time_funcs([run_optim_dace, run_cudnn],
                                func_names=["dace", "cudnn"],
                                warmups=warmupiter,
                                num_iters=totaliter,
                                launch_wait=setlaunchwait)
                print_time_statistics(times, [ "dace", "cudnn"])
                layersummary['layer_name'] = layer_name
                layersummary['times'] = times
                layersummary['dace_median'] = statistics.median(times[0])
                layersummary['cudnn_median'] = statistics.median(times[1])
                median_dace.append(statistics.median(times[0]))
                median_cudnn.append(statistics.median(times[1]))
                dace_median.append(times[0])
                cudnn_median.append(times[1])
                summary.append(layersummary)
            else:
                times = time_funcs([run_optim_dace, run_torch],
                                func_names=["dace", "torch"],
                                warmups=warmupiter,
                                num_iters=totaliter,
                                launch_wait=setlaunchwait)
                print_time_statistics(times, [ "dace", "torch"])
                layersummary['layer_name'] = layer_name
                layersummary['times'] = times
                layersummary['dace_median'] = statistics.median(times[0])
                layersummary['torch_median'] = statistics.median(times[1])
                median_dace.append(statistics.median(times[0]))
                median_torch.append(statistics.median(times[1]))
                dace_median.append(times[0])
                torch_median.append(times[1])
                summary.append(layersummary)

        # run using prof for torch
        if proftorch:
            times = time_funcs([run_torch],
                            func_names=["pytorch"],
                            warmups=warmupiter,
                            num_iters=totaliter,
                            launch_wait=setlaunchwait)
            print_time_statistics(times, [ "pytorch"])

        # run using prof for dace
        if profoptimdace:
            times = time_funcs([run_optim_dace],
                            func_names=["optimdace"],
                            warmups=warmupiter,
                            num_iters=totaliter,
                            launch_wait=setlaunchwait)
            print_time_statistics(times, [ "optimdace"])
        
        if profcudnn:
            if(not(useCudnn)):
                sys.exit("!!ERROR: Cudnn is not available for non NVIDIA GPUs")
            times = time_funcs([run_cudnn],
                            func_names=["cudnn"],
                            warmups=warmupiter,
                            num_iters=totaliter,
                            launch_wait=setlaunchwait)
            print_time_statistics(times, [ "cudnn"])
        
        if(useCudnn):
            in_desc, out_desc, filt_desc, ws_ptr = destroydescinoutfilt(in_desc, out_desc, filt_desc, ws_ptr)


createsummaryfile(summary, outdir, csv_columns)
if(useCudnn):
    createplots(enableplots, lastlayer, currlayer, warmupiter, totaliter, paramscsv, outdir, median_dace, median_cudnn, layer_names, summary)
else:
    createplots(enableplots, lastlayer, currlayer, warmupiter, totaliter, paramscsv, outdir, median_dace, median_torch, layer_names, summary)

# cudnn clear context
if(useCudnn):
    libcudnn.cudnnDestroyConvolutionDescriptor(conv_desc)
    libcudnn.cudnnDestroy(cudnn_context)