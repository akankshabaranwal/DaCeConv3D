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
from dace.sdfg.utils import load_precompiled_sdfg

from dace.transformation.dataflow import (DoubleBuffering, MapCollapse, MapExpansion, MapReduceFusion, StripMining,
                                          InLocalStorage, AccumulateTransient, Vectorization)

# For library node implementations
import dace.libraries.blas

# TODO: Try out different tranformations and see its effect on the generated graph.
# TODO: Fix different parts of your code to make it also work with parameters.
# Define symbolic sizes for arbitrary inputs
# Convention used for 3D convolution height, width, depth, channels
global indepth, inheight, inwidth
global N, w
global kwidth, kdepth, kheight
global inchannels, outchannels
global maxd, maxh, maxw
global blockStep
blockStep = 2

N = 1

# Define symbolic sizes for arbitrary inputs
d_indepth = dace.symbol('indepth')
d_inheight = dace.symbol('inheight')
d_inwidth = dace.symbol('inwidth')

# Define data type to use
dtype = dace.float32
np_dtype = np.float32

n = 1
indepth = 32
inheight = 32
inwidth = 32
w = 3
inchannels = 4
outchannels = 8 
# Prepare data with numpy
Input = np.random.rand(n, indepth, inheight, inwidth, inchannels).astype(np_dtype)
kernel = np.random.rand(w, w, w, inchannels, outchannels).astype(np_dtype)
Output = np.zeros((n, indepth, inheight, inwidth, outchannels), dtype=np_dtype)
kdepth = w
kheight = w
kwidth = w
maxd = int((indepth-kdepth+1)/blockStep)
maxh = int((inheight-kheight+1)/blockStep)
maxw = int((inwidth-kwidth+1)/blockStep)
                
# Trying out some optimizations
def optimize_for_gpu(sdfg: dace.SDFG):
    """ Optimize 3D convolution example for GPUs. """
    # Ensure integers are 32-bit by default
    dace.Config.set('compiler', 'default_data_types', value='C')
    sdfg.simplify()
    # Apply GPU transformation
    sdfg.apply_gpu_transformations()

# TODO: Parameterize the indepth, inheight, inwidth, inchannels, outchannels and then try out the optimized code
# TODO: Clean up the code to remove unnecessary variables, functions etc. 
# Simple parallel 3D convolution
@dace.program(device=dace.DeviceType.GPU, auto_optimize=True)
def dace_simple( Input, kernel, Output):
    #Output[:] = 0
    for n, d, h, w, oc in dace.map[0:N, 0:indepth-kdepth+1, 0:inheight-kheight+1, 0:inwidth-kwidth+1, 0:outchannels]:
        r_tmp = np.zeros([1], dtype=Input.dtype)
        r_kernel = np.copy(kernel[:,:,:,:,oc])
        for kd, kh, kw, ic in dace.map[0:kdepth, 0:kheight, 0:kwidth, 0:inchannels]:
            r_tmp = r_tmp + Input[n, d+kd, h+kh, w+kw, ic] * r_kernel[kd, kh, kw, ic]
        Output[n, d, h, w, oc] = r_tmp


@dace.program(device=dace.DeviceType.GPU)
def dace_tiling_2X2X2( Input, kernel, Output):
    Output[:] = 0
    for n, dparti, hparti, wparti, wind, winh, winw, oc in dace.map[0:N, 
    0:maxd, 0:maxh, 0:maxw, 
    0:blockStep, 0:blockStep, 0:blockStep, 
    0:outchannels]:# Should be done in parallel
        for kd, kh, kw, ic in dace.map[0:kdepth, 0:kheight, 0:kwidth, 0:inchannels]: #Compulsory sequential part
            Output[n, 
            blockStep*dparti+wind, 
            blockStep*hparti+winh, 
            blockStep*wparti+winw, 
            oc] += kernel[kd, kh, kw, ic, oc] * Input[n,
                                                    blockStep*dparti+wind+kd, 
                                                    blockStep*hparti+winh+kh, 
                                                    blockStep*wparti+winw+kw, 
                                                    ic]


enableFun = [dace_simple]

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


# Dace profiling method, Returns median values in ms
def rundacesdfgprofiling(dace_fun, Input, kernel, Output, reps):
    # Temporarily set the DACE_profiling config to True
    with dace.config.set_temporary('profiling', value=True):
        # You can control the number of times a program is run with the treps configuration
        with dace.config.set_temporary('treps', value=reps):
            dace_fun(Input=Input,kernel=kernel,Output=Output)

    list_of_files = glob.glob(f'.dacecache/*/profiling/results-*.csv')
    latest_file = max(list_of_files, key=os.path.getctime)
    df = pd.read_csv(latest_file)
    return df['Runtime_sec'].median()*1000

# Place holder function for tf reference code for profiling.
def timetfgpu_conv3D(input, filter):
    op=tf.nn.conv3d(input, filter, strides=[1, 1, 1, 1, 1], padding="VALID")

# Verification
def verify_with_ref_conv3D(dace_fun, refop, Input, kernel, Output):
    global indepth, inheight, inwidth
    global N
    global kwidth, kdepth, kheight,w
    global inchannels, outchannels
    dace_fun(Input, kernel, Output)
    opdace = tf.convert_to_tensor(Output)
    opdace = opdace[:, 0 : (indepth-w+1), 0 : (inheight-w+1) , 0 : (inwidth-w+1) , :]
    diff = np.linalg.norm(opdace - refop) / (N * outchannels * indepth * inheight * inwidth)
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
    global kwidth, kdepth, kheight
    global w, N
    global inchannels, outchannels
    global indepth, inheight, inwidth
    global maxd, maxh, maxw
    inchannels = currconv["InChannel"]
    indepth = currconv["InputDepth"]
    inheight = currconv["InputHeight"]
    inwidth = currconv["InputWidth"]
    kdepth = currconv["KernelDepth"]
    kheight = currconv["KernelHeight"]
    kwidth = currconv["KernelWidth"]
    outchannels = currconv["OutputChannel"]
    maxd = int((indepth-kdepth+1)/blockStep)
    maxh = int((inheight-kheight+1)/blockStep)
    maxw = int((inwidth-kwidth+1)/blockStep)

    if (kdepth!=kheight or kheight!=kwidth or kdepth!=kwidth):
        print("ERROR: !!! Conv3D with unequal kernel spatial dimensions not implemented")
        return 1
    w = kdepth
    # Prepare data with numpy
    Input = np.random.rand(N, indepth, inheight, inwidth, inchannels).astype(np_dtype)
    kernel = np.random.rand(w, w, w, inchannels, outchannels).astype(np_dtype)
    Output = np.zeros((N, indepth, inheight, inwidth, outchannels), dtype=np_dtype)
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
        for functionname in enableFun:
            verify_with_ref_conv3D(functionname, op, Input, kernel, Output)
        print("INFO: Verification done")


# Code to run the benchmarking on csv compared with tensorflow
def benchmarkconv3D(csv, benchmark_fun, suffix, issdfg):
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
        n = 1
        inchannels = currconv["InChannel"]
        indepth = currconv["InputDepth"]
        inheight = currconv["InputHeight"]
        inwidth = currconv["InputWidth"]
        w = currconv["KernelDepth"]
        outchannels = currconv["OutputChannel"]

        # Dace Warmup
        if not (issdfg):
            rundaceprofiling(benchmark_fun, Input, kernel, Output, 10)
        else:            
            rundacesdfgprofiling(benchmark_fun, Input, kernel, Output, 10)
        print("INFO: Warmup done")

        print("Start benchmarking instance")
        # Main benchmarking
        TIMES = {}
        nrepeat = 100
        if not (issdfg):
           TIMES['dace_optimized_fun'] = rundaceprofiling(benchmark_fun, Input, kernel, Output, nrepeat)
        else:            
            TIMES['dace_optimized_fun'] = rundacesdfgprofiling(benchmark_fun, Input, kernel, Output, 10, nrepeat, indepth, inheight, inwidth, inchannels, w, outchannels)
        x = timeit.Timer(lambda: timetfgpu_conv3D(input, filter)).repeat(repeat=100, number=2)
        TIMES['tfgpu'] = np.median(x)
        ALLPARAMSTIMES[f'{index}'] = TIMES
        print("End benchmarking instance")
    jsonfile = f'benchmarkout/dace_fun{suffix}{csv}.json'
    json.dump(ALLPARAMSTIMES, open(jsonfile, 'w'))


def optimizewithsdfgfun(csv, dace_fun):
    print("Will create sdfgs to optimize")
    global kwidth, kdepth, kheight
    global w, N
    global inchannels, outchannels
    global indepth, inheight, inwidth
    global maxd, maxh, maxw
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
            N = 1
            inchannels = currconv["InChannel"]
            indepth = currconv["InputDepth"]
            inheight = currconv["InputHeight"]
            inwidth = currconv["InputWidth"]
            w = currconv["KernelDepth"]
            outchannels = currconv["OutputChannel"]

            print("For SDFGs code only parses first row of inputs")
            break
    kdepth = w
    kheight = w
    kwidth = w
    maxd = int((indepth-kdepth+1)/blockStep)
    maxh = int((inheight-kheight+1)/blockStep)
    maxw = int((inwidth-kwidth+1)/blockStep)

    sdfg_fun: dace.SDFG = dace_fun.to_sdfg(Input,kernel,Output)
    optimize_for_gpu(sdfg_fun)
    sdfg_fun(Input,kernel,Output)
    return sdfg_fun

@click.command()
@click.option('--csv', type=str, default='None')
@click.option('--mode',
              type=click.Choice(
                  ('benchmarkoptimized', 'benchmarkunoptimized', 'verify', 'optimizeforgpu')),
              default='optimizeforgpu')
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
            issdfg = True
            benchmarkconv3D(csv, sdfg_fun, iter, issdfg)
            iter = iter + 1
    elif mode == 'optimizeforgpu':
        for fun_name in enableFun:
            sdfg_fun: dace.SDFG = fun_name.to_sdfg(Input,kernel,Output)
            optimize_for_gpu(sdfg_fun)
            sdfg_fun(Input,kernel,Output)
            print("Done calling optimize for gpu")
        
        # Prepare inputs for tensorflow fun
        input = tf.convert_to_tensor(Input)
        filter = tf.convert_to_tensor(kernel)
        # TF warm up
        timeit.Timer(lambda: timetfgpu_conv3D(input, filter)).repeat(repeat=5, number=1)
        x = timeit.Timer(lambda: timetfgpu_conv3D(input, filter)).repeat(repeat=100, number=2)
        print("Tensorflow in ms: ", (np.median(x))*1000)
        sdfg_fun(Input,kernel,Output)
        print("*\n*\n*")
        print("Running code with no optimizations")        
        rundacesdfgprofiling(sdfg_fun, Input, kernel, Output,10)
        rundacesdfgprofiling(sdfg_fun, Input, kernel, Output,100)
        print("*\n*\n*")
        print("Running code with optimizations")
        optimized_v1 = load_precompiled_sdfg('/home/akanksha/spcl/dacelocal/.dacecache/dace_simple_0/')
        optimized_v1(Input=Input, kernel=kernel, Output=Output)
        rundacesdfgprofiling(optimized_v1, Input, kernel, Output,10)
        rundacesdfgprofiling(optimized_v1, Input, kernel, Output,100)    
    else:
        print("Not sure what you wanted to do. Choose between benchmarkunoptimized, verify and benchmarkoptimized")
    return 0


if __name__ == "__main__":
    cli()