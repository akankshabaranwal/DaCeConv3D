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

# TODO: Add stride padding parameters
# TODO: Check what should be the expected behaviour for even dimension of filter size

# Define symbolic sizes for arbitrary inputs
# Convention used for 3D convolution height, width, depth, channels
inchannels = dace.symbol('inchannels')
indepth = dace.symbol('indepth')
inheight = dace.symbol('inheight')
inwidth = dace.symbol('inwidth')

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

# TODO: Compare outputs of convolution simpleparallel and convolutionsimple
# layer with input size (N, Cin, D, H, W) output (N, Cout, Dout, Hout, Wout)
# Simple 3D convolution
@dace.program(auto_optimize=True, device=dace.DeviceType.GPU)
def dace_simple(Input: dtype[N, indepth, inheight, inwidth, inchannels],
                      kernel: dtype[kdepth, kheight, kwidth, inchannels, outchannels],
                      Output: dtype[N, indepth, inheight, inwidth, outchannels]):
    Output[:] = 0
    for n, oc, ic, d, h, w, kd, kh, kw in dace.map[0:N, 0:outchannels, 0:inchannels, kdepth/2:indepth-kdepth/2, kheight/2:inheight-kheight/2, kwidth/2:inwidth-kwidth/2, 0:kdepth, 0:kheight, 0:kwidth]:
            Output[n, d, h, w, oc] += Input[n, d-kdepth/2+kd, h-kheight/2+kh, w-kwidth/2+kw, ic] * kernel[kd, kh, kw, ic, oc]


# Simple parallel 3D convolution
# TODO: Check for autooptimize for this
@dace.program(device=dace.DeviceType.GPU)
def dace_simpleparallel(Input: dtype[N, indepth, inheight, inwidth, inchannels],
                      kernel: dtype[kdepth, kheight, kwidth, inchannels, outchannels],
                      Output: dtype[N, indepth, inheight, inwidth, outchannels]):
    Output[:] = 0
    for n, oc, d, h, w in dace.map[0:N, 0:outchannels, kdepth/2:indepth-kdepth/2, kheight/2:inheight-kheight/2, kwidth/2:inwidth-kwidth/2]:
        tmp = np.zeros([1], dtype=Input.dtype)
        for ic, kd, kh, kw in dace.map[0:inchannels, 0:kdepth, 0:kheight, 0:kwidth]:
            tmp = tmp + Input[n, d-kdepth/2+kd, h-kheight/2+kh, w-kwidth/2+kw, ic] * kernel[kd, kh, kw, ic, oc]
        Output[n, d, h, w, oc] = tmp

# Place holder function for tf reference code for profiling.
def timetfgpu_conv3D(input, filter):
    op=tf.nn.conv3d(input, filter, strides=[1, 1, 1, 1, 1], padding="VALID")

# Verification
def verify_with_ref_conv3D(dace_fun, dace_fun_name, refop, Input, kernel, Output, n, outchannels, indepth, inheight, inwidth, w):
    dace_fun(Input, kernel, Output)
    opdace = tf.convert_to_tensor(Output)
    opdace = opdace[:, int(w / 2):indepth - int(w / 2), int(w / 2):inheight - int(w / 2), int(w / 2):inwidth - int(w / 2), :]

    diff = np.linalg.norm(opdace - refop) / (n * outchannels * indepth * inheight * inwidth)
    print('Difference:', diff)
    if(diff<=1e-6):
        print(f"Verification successfull for {dace_fun_name}")
    else:
        print(f"!!! Incorrect convolution for {dace_fun_name} with parameters:"
              f"{inchannels}x{indepth}x{inheight}x{inwidth} with "
              f"kernel {outchannels}x{inchannels}x{w}x{w}x{w}")

# Code to run the benchmarking and verif
def benchmarkandverifyconv3D(csv, nobenchmark):
    # Maybe write a script which goes through these csvs one by one and generates the reports for each of them
    header = ["InChannel","InputDepth","InputHeight","InputWidth","KernelDepth","KernelHeight","KernelWidth","OutputChannel","OutputDepth","OutputHeight","OutputWidth"]

    csvfile = f'convparam/{csv}.csv'
    convparams = pd.read_csv(csvfile)
    convparams = pd.DataFrame(convparams, columns=header)

    print("Parsing conv2D parameters")

    # Reset index to iterate through each 2D convolution parameter from the csv file
    convparams = convparams.reset_index()

    ALLPARAMSTIMES = {}
    for index, currconv in convparams.iterrows():
        # Extract parameters to prepare data for convolution
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

        # ** Start Verification
        input = tf.convert_to_tensor(Input)
        filter = tf.convert_to_tensor(kernel)
        op = tf.nn.conv3d(input, filter, strides=[1, 1, 1, 1, 1], padding="VALID")
        verify_with_ref_conv3D(dace_simpleparallel, 'dace_simpleparallel', op, Input, kernel, Output, n, outchannels, indepth, inheight, inwidth, w)
        verify_with_ref_conv3D(dace_simple, 'dace_simple', op, Input, kernel, Output, n, outchannels, indepth, inheight, inwidth, w)
        print("INFO: Verification done")
        # ** End Verification

        if nobenchmark:
            print("WARN: Skipping benchmarking")
            continue

        # ** Start Benchmarking **
        # Warm up
        timeit.Timer(dace_simpleparallel).repeat(repeat=5, number=1)
        timeit.Timer(lambda: timetfgpu_conv3D(input, filter)).repeat(repeat=5, number=1)
        print("INFO: Warmup done")

        # Main benchmarking
        TIMES = {}
        nrepeat = 100
        TIMES['dace_simpleparallel'] = rundaceprofiling(dace_simpleparallel, 'dace_simpleparallel', Input, kernel, Output, nrepeat)
        x = timeit.Timer(lambda: timetfgpu_conv3D(input, filter)).repeat(repeat=100, number=2)
        TIMES['tfgpu'] = np.median(x)
        ALLPARAMSTIMES[f'{index}'] = TIMES
        print(f"INFO: Done benchmarking with params as "
              f"Batch size:{n}, "
              f"InputChannels:{inchannels}, "
              f"InputDepth X InputHeight X InputWidth:{indepth}X{inheight}X{inwidth}, "
              f"Outputchannels:{outchannels}, "
              f"Kernel dimension:{w}")
        print("\n\n")

    # TODO: Fix file name so that it corresponds to the csv file that was read
    jsonfile = f'benchmarkout/{csv}.json'
    json.dump(ALLPARAMSTIMES, open(jsonfile, 'w'))

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


@click.command()
@click.option('--csv', type=str, default='cosmoflow')
@click.option('--nobenchmark/--benchmark', default=True)

def cli(csv, nobenchmark):
    benchmarkandverifyconv3D(csv, nobenchmark)

if __name__ == "__main__":
    cli()