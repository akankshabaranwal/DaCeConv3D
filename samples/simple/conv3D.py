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
from dace.transformation.dataflow import (MapCollapse, MapExpansion) 
# Define constants for filter dimensions
global kdim
kdim = 3

#####################################################################
# Data-centric optimization helpers copied from matmul.py 


def find_map_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    return next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry) and pname in n.params)

# Define symbolic sizes for arbitrary inputs
d_indepth = dace.symbol('d_indepth')
d_inheight = dace.symbol('d_inheight')
d_inwidth = dace.symbol('d_inwidth')
d_inchannels = dace.symbol('d_inchannels')
d_outchannels = dace.symbol('d_outchannels')
d_batchsize = dace.symbol('d_batchsize')

# Define data type to use
dtype = dace.float32
np_dtype = np.float32
                
# Optimize code on the GPU
def optimize_for_gpu(sdfg: dace.SDFG):
    """ Optimize 3D convolution example for GPUs. """
    dace.Config.set('compiler', 'default_data_types', value='C')
    sdfg.simplify()
    # Apply GPU transformation
    sdfg.apply_gpu_transformations()
    
    # Expand the maps
    m_expandparams = find_map_by_param(sdfg, 'd')
    MapExpansion.apply_to(sdfg, map_entry=m_expandparams)

    # Collapse the maps grouped appropriately
    m_d = find_map_by_param(sdfg, 'd')
    m_h = find_map_by_param(sdfg, 'h')
    MapCollapse.apply_to(sdfg, outer_map_entry=m_d, inner_map_entry=m_h)
    m_d = find_map_by_param(sdfg, 'd')
    m_w = find_map_by_param(sdfg, 'w')
    MapCollapse.apply_to(sdfg, outer_map_entry=m_d, inner_map_entry=m_w)
    
    # Schedule the collapsed maps on the GPU
    m_h = find_map_by_param(sdfg, 'h')
    m_oc = find_map_by_param(sdfg, 'oc')
    m_h.map.schedule = dace.ScheduleType.GPU_ThreadBlock
    m_oc.map.schedule = dace.ScheduleType.GPU_ThreadBlock

# Simple parallel 3D convolution
@dace.program(device=dace.DeviceType.GPU)
def dace_conv3d( Input: dtype[d_batchsize, d_indepth, d_inheight, d_inwidth, d_inchannels], 
                kernel: dtype[kdim, kdim, kdim, d_inchannels, d_outchannels], 
                Output: dtype[d_batchsize, d_indepth-kdim+1, d_inheight-kdim+1, d_inwidth-kdim+1, d_outchannels]):
    for n, d, h, w, oc in dace.map[0:d_batchsize, 0:d_indepth-kdim+1, 0:d_inheight-kdim+1, 0:d_inwidth-kdim+1, 0:d_outchannels]:
        r_tmp = np.zeros([1], dtype=Input.dtype)
        r_kernel = np.copy(kernel[:,:,:,:,oc])
        for kd, kh, kw, ic in dace.map[0:kdim, 0:kdim, 0:kdim, 0:d_inchannels]:
            r_tmp = r_tmp + Input[n, d+kd, h+kh, w+kw, ic] * r_kernel[kd, kh, kw, ic]
        Output[n, d, h, w, oc] = r_tmp


# Dace profiling method, Returns median values in ms
def rundacesdfgprofiling(dace_fun, Input, kernel, Output, inchannels, indepth, inheight, inwidth, outchannels, batchsize, reps):
    # Temporarily set the DACE_profiling config to True
    with dace.config.set_temporary('profiling', value=True):
        # You can control the number of times a program is run with the treps configuration
        with dace.config.set_temporary('treps', value=reps):
            dace_fun(Input=Input, kernel=kernel, Output=Output,d_inchannels=inchannels, d_indepth=indepth, d_inheight=inheight,d_inwidth=inwidth, d_outchannels=outchannels, d_batchsize=batchsize)
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
    global kdim
    inchannels = currconv["InChannel"]
    indepth = currconv["InputDepth"]
    inheight = currconv["InputHeight"]
    inwidth = currconv["InputWidth"]
    outchannels = currconv["OutputChannel"]
    outdepth = indepth-kdim+1
    outheight = inheight-kdim+1
    outwidth = inheight-kdim+1
    batchsize = 1
    # Prepare data with numpy
    Input = np.random.rand(batchsize, indepth, inheight, inwidth, inchannels).astype(np_dtype)
    kernel = np.random.rand(kdim, kdim, kdim, inchannels, outchannels).astype(np_dtype)
    Output = np.zeros((batchsize, outdepth, outheight, outwidth, outchannels), dtype=np_dtype)
    print(f'\n***** \n***** \n Begin 3D Convolution for Input parameters {inchannels}x{indepth}x{inheight}x{inwidth} '
            f'with kernel {outchannels}x{inchannels}x{kdim}x{kdim}x{kdim}')
    return Input, kernel, Output, inchannels, indepth, inheight, inwidth, outchannels, batchsize


# Ideally for each input it should first verify. only if verification is successful should it go to benchmark
@click.command()
@click.option('--csv', type=str, default='None')
@click.option('--mode',
              type=click.Choice(
                  ('verify', 'comparewithfixedsdfg')),
              default='verify')
def cli(csv, mode):
    global kdim
    if(csv == 'None'):
        csv = 'sample'
    if mode == 'verify':
        convparams =  parsecsv(csv)
        for index, currconv in convparams.iterrows():
            # Extract parameters to prepare data for convolution
            Input, kernel, Output, inchannels, indepth, inheight, inwidth, outchannels, batchsize = prepareinputs(currconv)
            input = tf.convert_to_tensor(Input)
            filter = tf.convert_to_tensor(kernel)
            refop = tf.nn.conv3d(input, filter, strides=[1, 1, 1, 1, 1], padding="VALID")
            dace_conv3d(Input, kernel, Output)
            opdace = tf.convert_to_tensor(Output)
            diff = np.linalg.norm(opdace - refop) / (batchsize * outchannels * indepth * inheight * inwidth )
            print('Difference between tensorflow and dace values:', diff)
            if(diff<=1e-4):
                print(f"Verification successfull")
            else:
                print(f"!!! ERROR: Incorrect verification")
                exit()

    elif mode == 'comparewithfixedsdfg':
        convparams = parsecsv(csv)
        for index, currconv in convparams.iterrows():            
            Input, kernel, Output, inchannels, indepth, inheight, inwidth, outchannels, batchsize = prepareinputs(currconv)
            inchannels = np.int32(inchannels)
            indepth = np.int32(indepth)
            inheight = np.int32(inheight)
            inwidth = np.int32(inwidth)
            outchannels = np.int32(outchannels)
            batchsize = np.int32(batchsize)
            
            sdfg_fun: dace.SDFG = dace_conv3d.to_sdfg(Input,kernel,Output)

            ## Prepare inputs for tensorflow fun
            input = tf.convert_to_tensor(Input)
            filter = tf.convert_to_tensor(kernel)
            ## TF warm up
            timeit.Timer(lambda: timetfgpu_conv3D(input, filter)).repeat(repeat=5, number=1)
            x = timeit.Timer(lambda: timetfgpu_conv3D(input, filter)).repeat(repeat=100, number=2)
            print("INFO: Tensorflow in ms: ", (np.median(x))*1000)
            # Apply GPU transformation
            sdfg_fun.apply_gpu_transformations()
            print("INFO: Running GPU transformed baseline sdfg with no other optimizations")        
            rundacesdfgprofiling(sdfg_fun, Input, kernel, Output, inchannels, indepth, inheight, inwidth, outchannels, batchsize, 10)
            rundacesdfgprofiling(sdfg_fun, Input, kernel, Output, inchannels, indepth, inheight, inwidth, outchannels, batchsize, 100)

            optimize_for_gpu(sdfg_fun)
            print("*\n*\n*")
            print("INFO: Running optimized sdfg code")
            rundacesdfgprofiling(sdfg_fun, Input, kernel, Output, inchannels, indepth, inheight, inwidth, outchannels, batchsize, 10)
            rundacesdfgprofiling(sdfg_fun, Input, kernel, Output, inchannels, indepth, inheight, inwidth, outchannels, batchsize, 100)
            
    return 0

if __name__ == "__main__":
    cli()
