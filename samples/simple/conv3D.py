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

# For library node implementations
import dace.libraries.blas

# TODO: Fix different parts of your code to make it also work with parameters.

# Define constants for batch size and filter dimensions
global N
global kdim
N = 1
kdim = 3

# Define symbolic sizes for arbitrary inputs
d_indepth = dace.symbol('d_indepth')
d_inheight = dace.symbol('d_inheight')
d_inwidth = dace.symbol('d_inwidth')
d_inchannels = dace.symbol('d_inchannels')
d_outdepth = dace.symbol('d_outdepth')
d_outheight = dace.symbol('d_outheight')
d_outwidth = dace.symbol('d_outwidth')
d_outchannels = dace.symbol('d_outchannels')

# Define data type to use
dtype = dace.float32
np_dtype = np.float32
                
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
@dace.program(device=dace.DeviceType.GPU)
def dace_simple( Input: dtype[N, d_indepth, d_inheight, d_inwidth, d_inchannels], 
                kernel: dtype[kdim, kdim, kdim, d_inchannels, d_outchannels], 
                Output: dtype[N, d_outdepth, d_outheight, d_outwidth, d_outchannels]):
    for n, d, h, w, oc in dace.map[0:N, 0:d_outdepth, 0:d_outheight, 0:d_outwidth, 0:d_outchannels]:
        r_tmp = np.zeros([1], dtype=Input.dtype)
        r_kernel = np.copy(kernel[:,:,:,:,oc])
        for kd, kh, kw, ic in dace.map[0:kdim, 0:kdim, 0:kdim, 0:d_inchannels]:
            r_tmp = r_tmp + Input[n, d+kd, h+kh, w+kw, ic] * r_kernel[kd, kh, kw, ic]
        Output[n, d, h, w, oc] = r_tmp


# Dace profiling method, Returns median values in ms
def rundacesdfgprofiling(dace_fun, Input, kernel, Output, inchannels,indepth,inheight,inwidth,outdepth, outheight, outwidth,outchannels,reps):
    # Temporarily set the DACE_profiling config to True
    with dace.config.set_temporary('profiling', value=True):
        # You can control the number of times a program is run with the treps configuration
        with dace.config.set_temporary('treps', value=reps):
            dace_fun(Input=Input, kernel=kernel, Output=Output,d_inchannels=inchannels, d_indepth=indepth, d_inheight=inheight,d_inwidth=inwidth, d_outdepth=outdepth, d_outheight=outheight, d_outwidth = outwidth, d_outchannels=outchannels)

    list_of_files = glob.glob(f'.dacecache/*/profiling/results-*.csv')
    latest_file = max(list_of_files, key=os.path.getctime)
    df = pd.read_csv(latest_file)
    return df['Runtime_sec'].median()*1000

# Place holder function for tf reference code for profiling.
def timetfgpu_conv3D(input, filter):
    op=tf.nn.conv3d(input, filter, strides=[1, 1, 1, 1, 1], padding="VALID")

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
    global kdim, N
    inchannels = currconv["InChannel"]
    indepth = currconv["InputDepth"]
    inheight = currconv["InputHeight"]
    inwidth = currconv["InputWidth"]
    outchannels = currconv["OutputChannel"]
    outdepth = indepth-kdim+1
    outheight = inheight-kdim+1
    outwidth = inheight-kdim+1
    # Prepare data with numpy
    Input = np.random.rand(N, indepth, inheight, inwidth, inchannels).astype(np_dtype)
    kernel = np.random.rand(kdim, kdim, kdim, inchannels, outchannels).astype(np_dtype)
    Output = np.zeros((N, outdepth, outheight, outwidth, outchannels), dtype=np_dtype)
    print(f'3D Convolution {inchannels}x{indepth}x{inheight}x{inwidth} '
            f'with kernel {outchannels}x{inchannels}x{kdim}x{kdim}x{kdim}')

    return Input, kernel, Output, inchannels, indepth, inheight, inwidth, outchannels
    

# Code to run verif on csv
def verifyconv3D(csv):
    convparams =  parsecsv(csv)
    ALLPARAMSTIMES = {}
    for index, currconv in convparams.iterrows():
        # Extract parameters to prepare data for convolution
        Input, kernel, Output, inchannels, indepth, inheight, inwidth, outchannels = prepareinputs(currconv)
        input = tf.convert_to_tensor(Input)
        filter = tf.convert_to_tensor(kernel)
        refop = tf.nn.conv3d(input, filter, strides=[1, 1, 1, 1, 1], padding="VALID")
        global N
        global kdim
        dace_simple(Input, kernel, Output)
        opdace = tf.convert_to_tensor(Output)
        diff = np.linalg.norm(opdace - refop) / (N * outchannels * indepth * inheight * inwidth)
        print('Difference:', diff)
        if(diff<=1e-6):
            print(f"Verification successfull")
        else:
            print(f"!!! Incorrect convolution with parameters:"
                f"{inchannels}x{indepth}x{inheight}x{inwidth} with "
                f"kernel {outchannels}x{inchannels}x{kdim}x{kdim}x{kdim}")
            print("INFO: Verification done")

def optimizewithsdfgfun(csv, dace_fun,layer):
    print("Will create sdfgs to optimize")
    global kdim
    global N
    convparams = parsecsv(csv)
    currconv = convparams.iloc[layer]
    Input, kernel, Output, inchannels, indepth, inheight, inwidth, outchannels = prepareinputs(currconv)
    outdepth = indepth-kdim+1
    outheight = inheight-kdim+1
    outwidth = inheight-kdim+1
    print("For SDFGs code only parses first row of inputs")
    sdfg_fun: dace.SDFG = dace_fun.to_sdfg(Input,kernel,Output)
    optimize_for_gpu(sdfg_fun)
    sdfg_fun(Input=Input,kernel=kernel,Output=Output, d_inchannels=inchannels, d_indepth=indepth, d_inheight=inheight, d_inwidth=inwidth, d_outdepth = outdepth, d_outheight=outheight, d_outwidth = outwidth, d_outchannels=outchannels)
    return sdfg_fun, Input, kernel, Output, inchannels, indepth, inheight, inwidth,outdepth, outheight, outwidth, outchannels


@click.command()
@click.option('--csv', type=str, default='None')
@click.option('--mode',
              type=click.Choice(
                  ('benchmarkoptimized', 'verify', 'comparewithfixedsdfg')),
              default='optimizeforgpu')
def cli(csv, mode):
    if(csv == 'None'):
        csv = 'sample'
    if mode == 'verify':
        verifyconv3D(csv)
    elif mode == 'comparewithfixedsdfg':
        layer = 2
        sdfg_fun, Input, kernel, Output,inchannels,indepth,inheight,inwidth,outdepth, outheight, outwidth, outchannels = optimizewithsdfgfun(csv, dace_simple,layer)
        
        # Prepare inputs for tensorflow fun
        input = tf.convert_to_tensor(Input)
        filter = tf.convert_to_tensor(kernel)
        # TF warm up
        timeit.Timer(lambda: timetfgpu_conv3D(input, filter)).repeat(repeat=5, number=1)
        x = timeit.Timer(lambda: timetfgpu_conv3D(input, filter)).repeat(repeat=100, number=2)
        print("Tensorflow in ms: ", (np.median(x))*1000)
        print("*\n*\n*")
        print("Running code with no optimizations")        
        rundacesdfgprofiling(sdfg_fun, Input, kernel, Output, inchannels, indepth, inheight, inwidth, outdepth, outheight, outwidth, outchannels, 10)
        rundacesdfgprofiling(sdfg_fun, Input, kernel, Output, inchannels, indepth, inheight, inwidth, outdepth, outheight, outwidth, outchannels, 100)
        
        print("*\n*\n*")
        print("Running code with optimizations")
        optimized_v1 = load_precompiled_sdfg('/home/akanksha/spcl/dacelocal/.dacecache/dace_generic_optim1/')
        rundacesdfgprofiling(optimized_v1, Input, kernel, Output, inchannels, indepth, inheight, inwidth, outdepth, outheight, outwidth, outchannels, 10)
        rundacesdfgprofiling(optimized_v1, Input, kernel, Output, inchannels, indepth, inheight, inwidth, outdepth, outheight, outwidth, outchannels, 100)
    return 0


if __name__ == "__main__":
    cli()