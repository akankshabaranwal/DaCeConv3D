# abaranwal: Code to profile 3D convolution based on daceml functions

from daceml.testing.profiling import time_funcs, print_time_statistics
import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
import sys

import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
import libcudnn, ctypes
import numpy as np
from convutils import prepareinputs, parsecsv, run_fun, createsummaryfile, createplots
import dace
import torch
import torch.nn.functional as F

# Make this as a choice depending on the kind of experiment you run
from directConv import *

import pandas as pd 

parser = argparse.ArgumentParser(description='Process some integers.')
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

# TODO: Check if you can plot and compare different versions of dace optimizations
# TODO: Automate the roofline analysis plot
# FIXME: Something is wrong with subplot when the csv file has just one row  

args = parser.parse_args()

paramscsv = args.paramscsv
convparams =  parsecsv(paramscsv)
verify = args.verify
compareprof = args.compareprof
runtorch = False
rundace = False
runcudnn = False
runoptimdace = False
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
#lastlayer = currlayer+1

torch.cuda.empty_cache()

#outdir = f'./outputplots/out{math.floor(time.time())}'
outdir = f'./outputplots/_out'
#os.mkdir(outdir)
with open(f'./{outdir}/params.txt', 'w') as f:
    f.writelines(f'csv: {paramscsv}\n')
    f.writelines(f'warmup iteration: {warmupiter}\n')
    f.writelines(f'total iteration: {totaliter}\n')
    f.writelines(f'set launch wait: {setlaunchwait}\n')

args = parser.parse_args()
d_input, d_kernel, d_output, inchannels, indepth, inheight, inwidth, outchannels, batchsize = prepareinputs(convparams.iloc[0], 'NCDHW', kdim)

## Prepare inputs for pytorch fun
t_input = d_input.clone()
t_kernel = d_kernel.clone()

inchannels = np.int32(inchannels)
indepth = np.int32(indepth)
inheight = np.int32(inheight)
inwidth = np.int32(inwidth)
outchannels = np.int32(outchannels)
batchsize = np.int32(batchsize)
pad = 0
dil = 1
stride = 1
sdfg_fun: dace.SDFG = dace_conv3d.to_sdfg(d_input, d_kernel, d_output)

## cudnn fixed parameters init
cudnn_context = libcudnn.cudnnCreate()
tensor_format = libcudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW']
data_type = libcudnn.cudnnDataType['CUDNN_DATA_FLOAT']
tensor_dim = 5
conv_dim = tensor_dim-2
convolution_mode = libcudnn.cudnnConvolutionMode['CUDNN_CROSS_CORRELATION']
convolution_algo = libcudnn.cudnnConvolutionFwdAlgo['CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM']
''' Available algorithms for 3d convolution cudnn are: 
CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING
'''
alpha = 1.0
beta = 0
c_int_p = ctypes.POINTER(ctypes.c_int)
outdimsinit = [0, 0, 0, 0, 0]
# cudnn convolution descriptor
conv_desc = libcudnn.cudnnCreateConvolutionDescriptor()
convpad = [pad, pad, pad]
filtstr = [stride, stride, stride]
convdil = [dil, dil, dil]
libcudnn.cudnnSetConvolutionNdDescriptor(conv_desc, conv_dim, convpad, filtstr, convdil, convolution_mode, data_type)


## cudnn variable parameters init, these change across different layers
# cudnn input
dims = [batchsize, inchannels, indepth, inheight, inwidth]
strides = [inchannels*indepth*inheight*inwidth, indepth*inheight*inwidth, inheight*inwidth, inwidth, 1]
in_desc = libcudnn.cudnnCreateTensorDescriptor()
libcudnn.cudnnSetTensorNdDescriptor(in_desc, data_type, tensor_dim, dims, strides)
# TODO: Maybe simplify this conversion to gpuarray ??
cudnn_input = d_input.detach().clone()
cudnn_kernel = d_kernel.detach().clone()
cudnn_output = d_output.detach().clone()
in_data_g = gpuarray.to_gpu(cudnn_input.cpu().numpy().astype(np.float32))
in_data = ctypes.c_void_p(int(in_data_g.gpudata))

# cudnn filter input
filt_desc = libcudnn.cudnnCreateFilterDescriptor()
filt_dims = [outchannels, inchannels, kdim, kdim, kdim]
libcudnn.cudnnSetFilterNdDescriptor(filt_desc, data_type, tensor_format, tensor_dim, filt_dims)
filt_data_g = gpuarray.to_gpu(cudnn_kernel.cpu().numpy().astype(np.float32))                                    
filt_data = ctypes.c_void_p(int(filt_data_g.gpudata))

# cudnn output
outdims = libcudnn.cudnnGetConvolutionNdForwardOutputDim(conv_desc, in_desc, filt_desc, tensor_dim, outdimsinit)
out_n, out_c, out_d, out_h, out_w = outdims[0], outdims[1], outdims[2], outdims[3], outdims[4]
outstrides = [ out_c*out_d*out_h*out_w, out_d*out_h*out_w, out_h*out_w, out_w, 1]
out_desc = libcudnn.cudnnCreateTensorDescriptor()
libcudnn.cudnnSetTensorNdDescriptor(out_desc, data_type, tensor_dim, outdims, outstrides)
out_data_g = gpuarray.to_gpu(cudnn_output.cpu().numpy().astype(np.float32))                               
out_data = ctypes.c_void_p(int(out_data_g.gpudata))
# Compute cudnn workspace size
ws_size = libcudnn.cudnnGetConvolutionForwardWorkspaceSize(cudnn_context, in_desc, filt_desc, conv_desc, out_desc, convolution_algo)
ws_ptr  = drv.mem_alloc(ws_size.value) if ws_size.value > 0 else 0
ws_data = ctypes.c_void_p(int(ws_ptr))


# Apply optimizations
optimize_for_gpu(sdfg_fun)
optim_dace = sdfg_fun.compile()

# Function call for pytorch 3D conv
def run_torch():
    op = F.conv3d(t_input, t_kernel, stride=1, padding='valid')

def run_cudnn():
    libcudnn.cudnnConvolutionForward(cudnn_context, alpha, in_desc, in_data, filt_desc, filt_data, 
                                conv_desc, convolution_algo, ws_data, ws_size.value, 
                                beta, out_desc, out_data)

# Function calls to run the optim dace function
def run_optim_dace():
    optim_dace(Input=d_input, kernel=d_kernel, Output=d_output,
            d_inchannels=inchannels, d_indepth=indepth, d_inheight=inheight,d_inwidth=inwidth, 
            d_outchannels=outchannels, d_batchsize=batchsize)

# Dace profiling method, Returns median values in ms
def rundaceprofiling(run_dace_fun, reps):
    # Temporarily set the DACE_profiling config to True
    with dace.config.set_temporary('profiling', value=True):
        # You can control the number of times a program is run with the treps configuration
        with dace.config.set_temporary('treps', value=reps):
            run_dace_fun()
    list_of_files = glob.glob(f'.dacecache/*/profiling/results-*.csv')
    latest_file = max(list_of_files, key=os.path.getctime)
    df = pd.read_csv(latest_file)
    return df['Runtime_sec']


    
median_dace = []
median_cudnn = []
layer_names = []
csv_columns = ['layer_name','dace_median','cudnn_median']
summary = []

# Clearing few cudnn variables
ws_ptr = None
libcudnn.cudnnDestroyTensorDescriptor(in_desc)
libcudnn.cudnnDestroyTensorDescriptor(out_desc)
libcudnn.cudnnDestroyFilterDescriptor(filt_desc)

for layern in range(currlayer, lastlayer):
    d_input, d_kernel, d_output, inchannels, indepth, inheight, inwidth, outchannels, batchsize = prepareinputs(convparams.iloc[layern], 'NCDHW', kdim)
    layersummary = {}
    t_input = d_input.clone()
    t_kernel = d_kernel.clone()

    inchannels = np.int32(inchannels)
    indepth = np.int32(indepth)
    inheight = np.int32(inheight)
    inwidth = np.int32(inwidth)
    outchannels = np.int32(outchannels)
    batchsize = np.int32(batchsize)
    layer_name = f'in_{batchsize}X{inchannels}X{indepth}X{inheight}X{inwidth}_k_{kdim}X{kdim}X{kdim}_och_{outchannels}'
    print(f'INFO: NCDHW layout {layer_name}') 
    layer_names.append(layer_name)

    ##  cudnn stuff
    dims = [batchsize, inchannels, indepth, inheight, inwidth]
    strides = [inchannels*indepth*inheight*inwidth, indepth*inheight*inwidth, inheight*inwidth, inwidth, 1]
    in_desc = libcudnn.cudnnCreateTensorDescriptor()
    libcudnn.cudnnSetTensorNdDescriptor(in_desc, data_type, tensor_dim, dims, strides)
    cudnn_input = d_input.detach().clone()
    cudnn_kernel = d_kernel.detach().clone()
    cudnn_output = d_output.detach().clone()
    in_data_g = gpuarray.to_gpu(cudnn_input.cpu().numpy().astype(np.float32))
    in_data = ctypes.c_void_p(int(in_data_g.gpudata))
    # cudnn filter input
    filt_desc = libcudnn.cudnnCreateFilterDescriptor()
    filt_dims = [outchannels, inchannels, kdim, kdim, kdim]
    libcudnn.cudnnSetFilterNdDescriptor(filt_desc, data_type, tensor_format, tensor_dim, filt_dims)
    filt_data_g = gpuarray.to_gpu(cudnn_kernel.cpu().numpy().astype(np.float32))                                    
    filt_data = ctypes.c_void_p(int(filt_data_g.gpudata))
    # cudnn output
    outdims = libcudnn.cudnnGetConvolutionNdForwardOutputDim(conv_desc, in_desc, filt_desc, tensor_dim, outdimsinit)
    out_n, out_c, out_d, out_h, out_w = outdims[0], outdims[1], outdims[2], outdims[3], outdims[4]
    outstrides = [ out_c*out_d*out_h*out_w, out_d*out_h*out_w, out_h*out_w, out_w, 1]
    out_desc = libcudnn.cudnnCreateTensorDescriptor()
    libcudnn.cudnnSetTensorNdDescriptor(out_desc, data_type, tensor_dim, outdims, outstrides)
    out_data_g = gpuarray.to_gpu(cudnn_output.cpu().numpy().astype(np.float32))                            
    out_data = ctypes.c_void_p(int(out_data_g.gpudata))
    # Compute cudnn workspace size
    ws_size = libcudnn.cudnnGetConvolutionForwardWorkspaceSize(cudnn_context, in_desc, filt_desc, conv_desc, out_desc, convolution_algo)
    ws_ptr  = drv.mem_alloc(ws_size.value) if ws_size.value > 0 else 0
    ws_data = ctypes.c_void_p(int(ws_ptr))

    
    # Code for verification
    if verify:
        print("INFO: Running verification to compare against cudnn output")
        run_cudnn()
        run_optim_dace()
        d_output = d_output.cpu()
        dace_output_g = gpuarray.to_gpu(d_output.numpy().astype(np.float32))                                    
        diff = np.linalg.norm((out_data_g - dace_output_g).get()) / (batchsize * outchannels * indepth * inheight * inwidth )
        print('Difference between cudnn and dace values:', diff)
        ## commented verif against pytorch
        #refop = F.conv3d(t_input, t_kernel, stride=1, padding='valid')
        #refop = refop.cpu()
        if(diff<=1e-5):
            print(f"Verification successfull")
        else:
            sys.exit("!!! ERROR: Incorrect verification")

    # Profiling pytorch using run
    if runtorch:
        run_fun(run_torch, warmupiter, totaliter)

    # Profiling optimized dace using run
    if runoptimdace:
        run_fun(run_optim_dace, warmupiter, totaliter)

    # Profiling optimized dace using run
    if runcudnn:
        run_fun(run_cudnn, warmupiter, totaliter)

    # Comparitive profiling using time funcs
    print(f"INFO: Warmup for {warmupiter} iterations and total iterations {totaliter}")
    print(f"INFO: Statistics for layer number {layern}")
    dace_median = []
    torch_median = []
    cudnn_median = []
    if compareprof:
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

    dace_median_array = np.array(dace_median)
    cudnn_median_array = np.array(cudnn_median)

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
        times = time_funcs([run_cudnn],
                        func_names=["cudnn"],
                        warmups=warmupiter,
                        num_iters=totaliter,
                        launch_wait=setlaunchwait)
        print_time_statistics(times, [ "cudnn"])
    
    libcudnn.cudnnDestroyTensorDescriptor(in_desc)
    libcudnn.cudnnDestroyTensorDescriptor(out_desc)
    libcudnn.cudnnDestroyFilterDescriptor(filt_desc)

createsummaryfile(summary, outdir, csv_columns)
createplots(enableplots, lastlayer, currlayer, warmupiter, totaliter, paramscsv, outdir, median_dace, median_cudnn, layer_names, summary)

# cudnn clear context
libcudnn.cudnnDestroyConvolutionDescriptor(conv_desc)
libcudnn.cudnnDestroy(cudnn_context)