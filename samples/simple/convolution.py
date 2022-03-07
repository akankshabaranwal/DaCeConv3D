# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

# Convolution sample code

import click
import dace
import numpy as np

import dace.libraries.blas

# TODO: Add stride padding parameters
# TODO: Modify the code for more dimensions

# Define symbolic sizes for arbitrary inputs
rows = dace.symbol('rows')
cols = dace.symbol('cols')
indepth = dace.symbol('indepth')
w = dace.symbol('w')

# Define data type to use
dtype = dace.float64
np_dtype = np.float64

# Simple unoptimized convolution code
@dace.program
def convolution3D(Input: dtype[rows, cols, indepth], kernel: dtype[w, w, indepth], Output: dtype[rows, cols]):
    tmp = np.zeros([rows, cols, indepth*w*w], dtype = Input.dtype)
    for i,j,d,m,n in dace.map[w/2:rows-w/2, w/2:cols-w/2,0:indepth, 0:w, 0:w]:
        with dace.tasklet:
            in_A << Input[i - w/2 + m, j - w/2 + n,d]
            in_B << kernel[ w-1-m, w-1-n,d]
            out >> tmp[i, j, (d*(w*w)) + (m*w)+n]

            out = in_A * in_B

    dace.reduce(lambda a,b:a+b, tmp, Output, axis=2, identity=0)


# Normal code for reference
def refconvolution2D(Input, kernel):

    Refw = kernel.shape[0]
    Refrows = Input.shape[0]
    Refcols = Input.shape[1]
    Refindepth = Input.shape[2]
    RefOutput = np.zeros((Refrows, Refcols), dtype=np_dtype)
    Reftmpii = np.int_(Refw/2)
    Reftmpjj = np.int_(Refw/2)
    RefwCenter = np.int_(Refw / 2)
    RefrowsEnd = Refrows - RefwCenter
    RefcolsEnd = Refcols - RefwCenter
    for i in range(RefwCenter,RefrowsEnd):
        for j in range(RefwCenter, RefcolsEnd):
            RefOutput[i,j] = 0
            for m in range(0,Refw):
                Refii = i - Reftmpii + m
                Refmm = Refw - 1 - m
                for n in range(0,Refw):
                    for d in range(0,Refindepth):
                        Refjj = j - Reftmpjj + n
                        Refnn = Refw - 1 - n
                        RefOutput[i,j] += Input[Refii][Refjj][d]*kernel[Refmm][Refnn][d]
    return RefOutput


#####################################################################
# Main function

@click.command()
@click.option('-rows', type=int, default=7)
@click.option('-cols', type=int, default=7)
@click.option('-indepth', type=int, default=3)
@click.option('-w', type=int, default=3)
@click.option('--version',
              type=click.Choice(
                  ('unoptimized','reference')),
              default='unoptimized')
@click.option('--verify/--no-verify', default=True)
def cli(rows, cols, indepth, w, version, verify):
    """
    Different available versions:
    unoptimized: Run `convolution2D` without optimizations;
    """

    # Prepare data with numpy
    Input = np.random.rand(rows, cols, indepth).astype(np_dtype)
    kernel = np.random.rand(w, w, indepth).astype(np_dtype)
    Output = np.zeros((rows, cols), dtype=np_dtype)

    # # Prepare data with numpy for debug
    # Input = np.ones((rows, cols), dtype = np_dtype)
    # kernel = np.ones((w, w), dtype = np_dtype)
    # Output = np.zeros((rows, cols), dtype=np_dtype)

    print(f'Convolution 2D {rows}x{cols}x{indepth} with kernel {w}x{w}x{indepth}(version: {version})')

    if version == 'unoptimized':
        # print("skipped call")
        # Simply call the program to run it
        convolution3D(Input, kernel, Output)
    else:
        raise ValueError('Invalid version %s' % version)

    if verify:
        #print("Computed from dace")
        #print(Output)
        expected = refconvolution2D(Input, kernel)
        #print("Computed reference")
        #print(expected)
        diff = np.linalg.norm(Output - expected) / (rows * cols)
        print('Difference:', diff)
        return 0 if diff <= 1e-6 else 1

if __name__ == "__main__":
    cli()
