# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

# Convolution benchmarking and verif code for 3D convolution

import click
import dace
import numpy as np
import tensorflow as tf
import dace.libraries.blas
import glob
import os
import pandas as pd
import json
import timeit

# For optimizations
from dace.transformation.dataflow import (DoubleBuffering, MapCollapse, MapExpansion, MapReduceFusion, StripMining,
                                          InLocalStorage, AccumulateTransient, Vectorization)
from dace.transformation import helpers as xfutil

# For library node implementations
import dace.libraries.blas

# Define symbolic sizes for arbitrary inputs
# Convention used for 3D convolution height, width, depth, channels
inchannels = dace.symbol('inchannels')
indepth = dace.symbol('indepth')
inheight = dace.symbol('inheight')
inwidth = dace.symbol('inwidth')

blockdim = dace.symbol('blockdim')

N = dace.symbol('N') # Batch size

outchannels = dace.symbol('outchannels')
outdepth = dace.symbol('outdepth')
outheight = dace.symbol('outheight')
outwidth = dace.symbol('outwidth')

kdepth = dace.symbol('kdepth')
kheight = dace.symbol('kheight')
kwidth = dace.symbol('kwidth')

# Define data type to use
dtype = dace.float64
np_dtype = np.float64

                
# Trying out some optimizations
def optimize_for_gpu(sdfg: dace.SDFG, n: int, indepth: int, inheight: int, inwidth: int, inchannels: int, w: int, outchannels: int):
    """ Optimize 3D convolution example for GPUs. """
    # Ensure integers are 32-bit by default
    dace.Config.set('compiler', 'default_data_types', value='C')
    # Apply GPU transformation
    sdfg.apply_gpu_transformations()


# Simple parallel 3D convolution
@dace.program(device=dace.DeviceType.GPU)
def dace_simple(Input: dtype[N, indepth, inheight, inwidth, inchannels],
                      kernel: dtype[kdepth, kheight, kwidth, inchannels, outchannels],
                      Output: dtype[N, indepth, inheight, inwidth, outchannels]):
    Output[:] = 0
    for n, oc, d, h, w in dace.map[0:N, 0:outchannels, 0:indepth-kdepth+1, 0:inheight-kheight+1, 0:inwidth-kwidth+1]:
        tmp = np.zeros([1], dtype=Input.dtype)
        for ic, kd, kh, kw in dace.map[0:inchannels, 0:kdepth, 0:kheight, 0:kwidth]:
            tmp = tmp + Input[n, d+kd, h+kh, w+kw, ic] * kernel[kd, kh, kw, ic, oc]
        Output[n, d, h, w, oc] = tmp


# # Loop tiling for outputs
# # Simple parallel 3D convolution
# # TODO: Is it important to use the SDFG graphs for this? Or do I just keep it as a tool to use if I need but if I can code it without it then its okay??
# @dace.program(device=dace.DeviceType.GPU)
# def dace_looptiling(Input: dtype[N, indepth, inheight, inwidth, inchannels],
#                       kernel: dtype[kdepth, kheight, kwidth, inchannels, outchannels],
#                       Output: dtype[N, indepth, inheight, inwidth, outchannels]):
#     Output[:] = 0
#     for n, oc, d, h, w in dace.map[0:N, 0:outchannels, kdepth/2:indepth-kdepth/2, kheight/2:inheight-kheight/2, kwidth/2:inwidth-kwidth/2]:
#         tmp = np.zeros([1], dtype=Input.dtype)
#         localInput = np.copy(Input[n, d-kdepth/2:d+kdepth/2, h-kheight/2:h+kheight/2, w-kwidth/2:w+kwidth/2, 0:inchannels])
#         localKernel = np.copy(kernel)
#         for kd, kh, kw, ic in dace.map[0:kdepth, 0:kheight, 0:kwidth, 0:inchannels]:
#             tmp = tmp + localInput[kd, kh, kw, ic] * localKernel[kd, kh, kw, ic, oc]
#         Output[n, d, h, w, oc] = tmp

# Take blocks of input and then compute all output channels one by one in the innermost loop. 
# It should be like you pick up a certain set of inputs which are needed for a particular output value 
# and then once its done only then you move on to the next set of inputs

@dace.program(device=dace.DeviceType.GPU)
def dace_blocktiling(Input: dtype[N, indepth, inheight, inwidth, inchannels],
                      kernel: dtype[kdepth, kheight, kwidth, inchannels, outchannels],
                      Output: dtype[N, indepth, inheight, inwidth, outchannels],
                    ):
    Output[:] = 0
    for n in dace.map[0:N]:# TODO should this be in the innermost loop? 
        SingleInput = np.copy(Input[n,:,:,:,:])
        SingleOutput = np.copy(Output[n,:,:,:,:])
        for d, h, w in dace.map[0:indepth-kdepth+1, 0:inheight-kheight+1, 0:inwidth-kwidth+1]:
            tmp = np.zeros([outchannels], dtype=Input.dtype)
            localInput = np.copy(SingleInput[d:d+kdepth, h:h+kheight, w:w+kwidth, 0:inchannels])
            localKernel = np.copy(kernel)
            for kd, kh, kw, ic, oc in dace.map[0:kdepth, 0:kheight, 0:kwidth, 0:inchannels, 0:outchannels]:
                tmp[oc] = tmp[oc] + localInput[kd, kh, kw, ic] * localKernel[kd, kh, kw, ic, oc]
            SingleOutput[ d, h, w, :] = tmp[:]
        Output[n,:,:,:,:] = SingleOutput


@dace.program(device=dace.DeviceType.GPU)
def dace_blocktilingv1(Input: dtype[N, indepth, inheight, inwidth, inchannels],
                      kernel: dtype[kdepth, kheight, kwidth, inchannels, outchannels],
                      Output: dtype[N, indepth, inheight, inwidth, outchannels],
                    ):
    Output[:] = 0
    for n in dace.map[0:N]:# TODO should this be in the innermost loop? 
        SingleInput = np.copy(Input[n,:,:,:,:])
        SingleOutput = np.copy(Output[n,:,:,:,:])
        for d, h, w in dace.map[0:indepth-kdepth+1, 0:inheight-kheight+1, 0:inwidth-kwidth+1]:
            tmp = np.zeros([outchannels], dtype=Input.dtype)
            localInput = np.copy(SingleInput[d:d+kdepth, h:h+kheight, w:w+kwidth, 0:inchannels])
            localKernel = np.copy(kernel)
            for kd, kh, kw, ic, oc in dace.map[0:kdepth, 0:kheight, 0:kwidth, 0:inchannels, 0:outchannels]:
                tmp[oc] = tmp[oc] + localInput[kd, kh, kw, ic] * localKernel[kd, kh, kw, ic, oc]
            SingleOutput[ d, h, w, :] = tmp[:]
        Output[n,:,:,:,:] = SingleOutput


enableFun = [dace_blocktilingv1, dace_blocktiling, dace_simple]

# Dace profiling method, Returns median values in ms
def rundaceprofiling(dace_fun, Input, kernel, Output, reps):
    # Temporarily set the DACE_profiling config to True
    with dace.config.set_temporary('profiling', value=True):
        # You can control the number of times a program is run with the treps configuration
        with dace.config.set_temporary('treps', value=reps):
            dace_fun(Input, kernel, Output)
    list_of_files = glob.glob(f'.dacecache/*/profiling/results-*.csv')
    latest_file = max(list_of_files, key=os.path.getctime)
    df = pd.read_csv(latest_file)
    return df['Runtime_sec'].median()*1000


# Place holder function for tf reference code for profiling.
def timetfgpu_conv3D(input, filter):
    op=tf.nn.conv3d(input, filter, strides=[1, 1, 1, 1, 1], padding="VALID")

# Verification
def verify_with_ref_conv3D(dace_fun, refop, Input, kernel, Output, n, inchannels, outchannels, indepth, inheight, inwidth, w):
    dace_fun(Input, kernel, Output)
    opdace = tf.convert_to_tensor(Output)
    opdace = opdace[:, 0 : (indepth-w+1), 0 : (inheight-w+1) , 0 : (inwidth-w+1) , :]
    diff = np.linalg.norm(opdace - refop) / (n * outchannels * indepth * inheight * inwidth)
    print('Difference:', diff)
    if(diff<=1e-6):
        print(f"Verification successfull")
    else:
        print(f"!!! Incorrect convolution with parameters:"
              f"{inchannels}x{indepth}x{inheight}x{inwidth} with "
              f"kernel {outchannels}x{inchannels}x{w}x{w}x{w}")

# Parse csv file to return a pandas array
def parsecsv(csv):
    header = ["InChannel","InputDepth","InputHeight","InputWidth","KernelDepth","KernelHeight","KernelWidth","OutputChannel","OutputDepth","OutputHeight","OutputWidth"]
    csvfile = f'convparam/{csv}.csv'
    convparams = pd.read_csv(csvfile)
    convparams = pd.DataFrame(convparams, columns=header)
    print("Parsing conv3D parameters")
    # Reset index to iterate through each 2D convolution parameter from the csv file
    convparams = convparams.reset_index()
    return convparams

def prepareinputs(currconv):
    n = 1
    inchannels = currconv["InChannel"]
    indepth = currconv["InputDepth"]
    inheight = currconv["InputHeight"]
    inwidth = currconv["InputWidth"]
    kdepth = currconv["KernelDepth"]
    kheight = currconv["KernelHeight"]
    kwidth = currconv["KernelWidth"]
    outchannels = currconv["OutputChannel"]

    if (kdepth!=kheight or kheight!=kwidth or kdepth!=kwidth):
        print("ERROR: !!! Conv3D with unequal kernel spatial dimensions not implemented")
        return 1
    w = kdepth
    # Prepare data with numpy
    Input = np.random.rand(n, indepth, inheight, inwidth, inchannels).astype(np_dtype)
    kernel = np.random.rand(w, w, w, inchannels, outchannels).astype(np_dtype)
    Output = np.zeros((n, indepth, inheight, inwidth, outchannels), dtype=np_dtype)
    print(f'3D Convolution {inchannels}x{indepth}x{inheight}x{inwidth} '
            f'with kernel {outchannels}x{inchannels}x{w}x{w}x{w}')

    return Input, kernel, Output
    

# Code to run verif on csv
def verifyconv3D(csv):
    convparams =  parsecsv(csv)
    ALLPARAMSTIMES = {}
    for index, currconv in convparams.iterrows():
        # Extract parameters to prepare data for convolution
        Input, kernel, Output = prepareinputs(currconv)
        input = tf.convert_to_tensor(Input)
        filter = tf.convert_to_tensor(kernel)
        op = tf.nn.conv3d(input, filter, strides=[1, 1, 1, 1, 1], padding="VALID")
        n = 1
        for functionname in enableFun:
            verify_with_ref_conv3D(functionname, op, Input, kernel, Output, n, currconv["InChannel"], currconv["OutputChannel"], currconv["InputDepth"], currconv["InputHeight"], currconv["InputWidth"], currconv["KernelWidth"])
        print("INFO: Verification done")

# Code to run the benchmarking on csv compared with tensorflow
def benchmarkconv3D(csv, benchmark_fun, suffix):
    convparams = parsecsv(csv)
    ALLPARAMSTIMES = {}
    for index, currconv in convparams.iterrows():
        # Prepare inputs for dace fun
        Input, kernel, Output = prepareinputs(currconv)
        # Prepare inputs for tensorflow fun
        input = tf.convert_to_tensor(Input)
        filter = tf.convert_to_tensor(kernel)
        # TF warm up
        timeit.Timer(lambda: timetfgpu_conv3D(input, filter)).repeat(repeat=5, number=1)
        # Dace Warmup
        rundaceprofiling(benchmark_fun, Input, kernel, Output, 10)
        print("INFO: Warmup done")

        print("Start benchmarking instance")
        # Main benchmarking
        TIMES = {}
        nrepeat = 100
        TIMES['dace_fun'] = rundaceprofiling(benchmark_fun, Input, kernel, Output, nrepeat)
        x = timeit.Timer(lambda: timetfgpu_conv3D(input, filter)).repeat(repeat=100, number=2)
        TIMES['tfgpu'] = np.median(x)
        ALLPARAMSTIMES[f'{index}'] = TIMES
        print("End benchmarking instance")
    jsonfile = f'benchmarkout/dace_fun{suffix}{csv}.json'
    json.dump(ALLPARAMSTIMES, open(jsonfile, 'w'))


def optimizewithsdfgfun(csv, dace_fun):
    print("Will create sdfgs and you can optimize stuff")
    if csv == 'nocsv':
        n = 1
        indepth = 8
        inheight = 8
        inwidth = 8
        w = 3
        inchannels = 5
        outchannels = 6   
        # Prepare data with numpy
        Input = np.random.rand(n, indepth, inheight, inwidth, inchannels).astype(np_dtype)
        kernel = np.random.rand(w, w, w, inchannels, outchannels).astype(np_dtype)
        Output = np.zeros((n, indepth, inheight, inwidth, outchannels), dtype=np_dtype)
    else:
        convparams = parsecsv(csv)
        for index, currconv in convparams.iterrows():
            Input, kernel, Output = prepareinputs(currconv)
            n = 1
            inchannels = currconv["InChannel"]
            indepth = currconv["InputDepth"]
            inheight = currconv["InputHeight"]
            inwidth = currconv["InputWidth"]
            w = currconv["KernelDepth"]
            outchannels = currconv["OutputChannel"]
            print("For SDFGs code only parses first row of inputs")
            break

    sdfg_fun: dace.SDFG = dace_fun.to_sdfg()
    optimize_for_gpu(sdfg_fun, n, indepth, inheight, inwidth, inchannels, w, outchannels)
    sdfg_fun(Input,kernel,Output, N=n, indepth=indepth, inheight=inheight, inwidth=inwidth, inchannels=inchannels, kdepth=w, kheight=w, kwidth=w, outchannels=outchannels)
    return sdfg_fun

@click.command()
@click.option('--csv', type=str, default='None')
@click.option('--mode',
              type=click.Choice(
                  ('benchmarkoptimized', 'benchmarkunoptimized', 'verify')),
              default='verify')
# Different available command lines are
# --mode verify
# --mode verify --csv sample
# --mode benchmark
# --mode benchmark --csv sample
# --mode optimize
# --mode optimize --csv sample
def cli(csv, mode):
    # Select which dace functions you want to enable
    # Select if you want it to read from a csv file and run the functions or if you want to hard code some values??
    blockdim = 2
    if(csv == 'None'):
        csv = 'sample'
    if mode == 'benchmarkunoptimized':
        iter = 0
        for fun_name in enableFun:
            benchmarkconv3D(csv, fun_name, iter)
            iter = iter+1
    elif mode == 'verify':
        verifyconv3D(csv)
    elif mode == 'benchmarkoptimized':
        if(csv == 'None'):
            csv = 'nocsv'
        iter = 0
        for fun_name in enableFun:
            sdfg_fun = optimizewithsdfgfun(csv, fun_name)
            benchmarkconv3D(csv, fun_name, iter)
            iter = iter + 1
    else:
        print("Not sure what you wanted to do. Choose between benchmarkunoptimized, verify and benchmarkoptimized")
    return 0


if __name__ == "__main__":
    cli()