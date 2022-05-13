# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
# Convolution benchmarking code
import click
import dace
import numpy as np
import dace.libraries.blas
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
import os
import pandas as pd
import json
import timeit

# Define symbolic sizes for arbitrary inputs
rows = dace.symbol('rows')
cols = dace.symbol('cols')
indepth = dace.symbol('indepth')
inputimages = dace.symbol('inputimages')
outdepth = dace.symbol('outdepth')
chunklength = dace.symbol('chunklength', dtype=dace.int64, integer=True, positive=True)

w = dace.symbol('w')

# Define data type to use
dtype = dace.float64
np_dtype = np.float64

# Different convolution variants
# Simple convolution
@dace.program(auto_optimize=True, device=dace.DeviceType.GPU)
def dace_simple(Input: dtype[inputimages, rows, cols, indepth],
                      kernel: dtype[ w, w, indepth, outdepth],
                      Output: dtype[inputimages, rows, cols, outdepth]):
    Output[:] = 0
    for i,j,d,od,m,n in dace.map[w/2:rows-w/2, w/2:cols-w/2,0:indepth,0:outdepth, 0:w, 0:w]:
            Output[0, i, j, od] += Input[0, i - w / 2 + m, j - w / 2 + n, d] * kernel[ m, n, d, od]


# Split into parallel and non parallel maps
# TODO: Find why auto optimize is not working for simpleparallel
@dace.program(device=dace.DeviceType.GPU)
def dace_simpleparallel(Input: dtype[inputimages, rows, cols, indepth],
                              kernel: dtype[ w, w, indepth, outdepth],
                              Output: dtype[inputimages, rows, cols, outdepth]
                              ):
    Output[:] = 0

    for i, j, od in dace.map[w/2:rows-w/2, w/2:cols-w/2, 0:outdepth]:
        tmp = np.zeros([1], dtype = Input.dtype)
        for d,m,n in dace.map[0:indepth,0:w,0:w]:
            tmp = tmp + Input[0, i - w / 2 + m, j - w / 2 + n, d] * kernel[m, n, d, od]
        Output[0,i,j,od] = tmp

# Place holder function for tf reference code for profiling.
def timetfgpu(input, filter):
    op=tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')

# Verification
def verify_with_ref(dace_fun, dace_fun_name, refop, Input, kernel, Output, Rows, Cols, W):
    dace_fun(Input, kernel, Output)
    opdace = tf.convert_to_tensor(Output)
    opdace = opdace[:,int(W/2):Rows-int(W/2),int(W/2):Cols-int(W/2),:]
    if(sum(sum(sum(sum(opdace-refop))))==0):
        print(f"Verification successfull for {dace_fun_name}")
    else:
        print(f"!!! Incorrect convolution for {dace_fun_name}")


# Dace profiling method, Returns median values in ms
def rundaceprofiling(dace_fun, dace_fun_name, Input, kernel, Output, reps):
    # Temporarily set the DACE_profiling config to True
    with dace.config.set_temporary('profiling', value=True):
        # You can control the number of times a program is run with the treps configuration
        with dace.config.set_temporary('treps', value=reps):
            dace_fun(Input, kernel, Output)
    dace_profile_fun = dace_fun_name
    list_of_files = glob.glob(f'.dacecache/{dace_profile_fun}/profiling/results-*.csv')
    latest_file = max(list_of_files, key=os.path.getctime)
    df = pd.read_csv(latest_file)
    return df['Runtime_sec'].median()*1000


# Code to run the benchmarking and verif
def benchmarkandverifyconv2d(csv):
    # Maybe write a script which goes through these csvs one by one and generates the reports for each of them
    header = ["InputDepth", "InputRow", "InputCol", "OutputDepth", "KernelRow", "KernelCol", "Stride"]
    csvfile = f'{csv}.csv'
    convparams = pd.read_csv(csvfile)
    convparams = pd.DataFrame(convparams, columns=header)

    print("Running benchmark for conv2D")

    # Reset index to iterate through each 2D convolution parameter from the csv file
    convparams = convparams.reset_index()

    ALLPARAMSTIMES = {}
    for index, currconv in convparams.iterrows():
        # Extract parameters to prepare data for convolution
        InputImages = 1
        Rows = currconv["InputRow"]
        Cols = currconv["InputCol"]
        InChannels = currconv["InputDepth"]
        OutChannels = currconv["OutputDepth"]
        W = currconv["KernelRow"]
        Stride = currconv["Stride"]

        # Prepare data with numpy
        Input = np.random.rand(InputImages, Rows, Cols, InChannels).astype(np_dtype)
        kernel = np.random.rand(W, W, InChannels, OutChannels).astype(np_dtype)
        Output = np.zeros((InputImages, Rows, Cols, OutChannels), dtype=np_dtype)

        # ** Start Verification
        # TODO: Fix when kernel dimension is even
        input = tf.convert_to_tensor(Input)
        filter = tf.convert_to_tensor(kernel)
        op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding="VALID")
        verify_with_ref(dace_simpleparallel, 'dace_simpleparallel', op, Input, kernel, Output, Rows, Cols, W)
        # TODO: Maybe exit and don't run benchmarking when the verification fails?
        print("INFO: Verification done")
        # ** End Verification

        # ** Start Benchmarking **
        # Warm up
        timeit.Timer(dace_simpleparallel).repeat(repeat=5, number=1)
        timeit.Timer(lambda: timetfgpu(input, filter)).repeat(repeat=5, number=1)
        print("INFO: Warmup done")

        # Main benchmarking
        TIMES = {}
        nrepeat = 100
        TIMES['dace_simpleparallel'] = rundaceprofiling(dace_simpleparallel, 'dace_simpleparallel', Input, kernel, Output, nrepeat)
        x = timeit.Timer(lambda: timetfgpu(input, filter)).repeat(repeat=100, number=10)
        TIMES['tfgpu'] = np.median(x)
        ALLPARAMSTIMES[f'{index}'] = TIMES
        print(f"INFO: Benchmarking with params as "
              f"InputImages:{InputImages}, "
              f"InputRow:{Rows}, "
              f"InputCols:{Cols}, "
              f"InputDepth:{InChannels}, "
              f"OutputDepth:{OutChannels} done")

    # TODO: Fix file name so that it corresponds to the csv file that was read
    jsonfile = f'../benchmarkout/{csv}.json'
    json.dump(ALLPARAMSTIMES, open(jsonfile, 'w'))


####################################################################
# Main function

@click.command()
@click.option('--csv', type=str, default='sample')
@click.option('--type',
              type=click.Choice(
                  ('conv1d','conv2d','conv3d','depthconv')),
              default='conv2d')

def cli(csv, type):
    if type == 'conv2d':
        benchmarkandverifyconv2d(csv)
    else:
        raise ValueError('Invalid/Not yet implemented type %s' % type)


if __name__ == "__main__":
    cli()