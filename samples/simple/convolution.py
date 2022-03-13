# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

# Convolution sample code

import click
import dace
import numpy as np

import dace.libraries.blas

# TODO: Add stride padding parameters

# Define symbolic sizes for arbitrary inputs
rows = dace.symbol('rows')
cols = dace.symbol('cols')
indepth = dace.symbol('indepth')
outdepth = dace.symbol('outdepth')
w = dace.symbol('w')

# Define data type to use
dtype = dace.float64
np_dtype = np.float64

# Simple convolution code using map reduce approach
@dace.program
def convolutionallparallel(Input: dtype[indepth, rows, cols], kernel: dtype[outdepth, indepth, w, w], Output: dtype[outdepth, rows, cols]):
    tmp = np.zeros([outdepth, rows, cols, indepth*w*w], dtype = Input.dtype)
    for i,j,d,od,m,n in dace.map[w/2:rows-w/2, w/2:cols-w/2,0:indepth,0:outdepth, 0:w, 0:w]:
        with dace.tasklet:
            in_A << Input[d, i - w/2 + m, j - w/2 + n]
            in_B << kernel[od, d, w-1-m, w-1-n]
            out >> tmp[od, i, j, (d*(w*w)) + (m*w)+n]

            out = in_A * in_B

    dace.reduce(lambda a,b:a+b, tmp, Output, axis=3, identity=0)


# Reducing memory footprint
@dace.program
def convolutionoutdepthserial(Input: dtype[indepth, rows, cols], kernel: dtype[outdepth, indepth, w, w], Output: dtype[outdepth, rows, cols]):
    for od in range(0,outdepth):
        tmp = np.zeros([rows, cols, indepth * w * w], dtype=Input.dtype)
        for i,j,d,m,n in dace.map[w/2:rows-w/2, w/2:cols-w/2,0:indepth, 0:w, 0:w]:
            with dace.tasklet:
                in_A << Input[d, i - w/2 + m, j - w/2 + n]
                in_B << kernel[od, d, w-1-m, w-1-n]
                out >> tmp[ i, j, (d*(w*w)) + (m*w)+n]

                out = in_A * in_B

        dace.reduce(lambda a,b:a+b, tmp, Output[od][:][:], axis=2, identity=0)


# No map reduce. Not working!!. Output is all zeros
@dace.program
def convolutionsimple(Input: dtype[indepth, rows, cols], kernel: dtype[outdepth, indepth, w, w], Output: dtype[outdepth, rows, cols]):
    Output = np.zeros([outdepth, rows, cols], dtype = Input.dtype)
    for i,j,d,od,m,n in dace.map[w/2:rows-w/2, w/2:cols-w/2,0:indepth,0:outdepth, 0:w, 0:w]:
        with dace.tasklet:
            in_A << Input[d, i - w / 2 + m, j - w / 2 + n]
            in_B << kernel[od, d, w - 1 - m, w - 1 - n]
            out >> Output[od, i, j]

            out += in_A * in_B


# Reduction along input depth
@dace.program
def convolutionindepthreduce(Input: dtype[indepth, rows, cols], kernel: dtype[outdepth, indepth, w, w], Output: dtype[outdepth, rows, cols]):
    for i, j, od in dace.map[w/2:rows-w/2, w/2:cols-w/2,0:outdepth]:
        tmp = np.zeros([indepth*w*w], dtype = Input.dtype)
        for d,m,n in dace.map[0:indepth,0:w,0:w]:
            with dace.tasklet:
                in_A << Input[d, i - w / 2 + m, j - w / 2 + n]
                in_B << kernel[od, d, w - 1 - m, w - 1 - n]
                out >> tmp[(d*(w*w)) + (m*w)+n]

                out = in_A * in_B
        Output[od][i][j] = dace.reduce(lambda a, b: a + b, tmp, identity=0)


# Normal code for reference
def refconvolution(Input, kernel):

    Refw = kernel.shape[2]
    Refrows = Input.shape[1]
    Refcols = Input.shape[2]
    Refindepth = Input.shape[0]
    Refoutdepth = kernel.shape[0]
    RefOutput = np.zeros((Refoutdepth, Refrows, Refcols), dtype=np_dtype)
    Reftmpii = np.int_(Refw/2)
    Reftmpjj = np.int_(Refw/2)
    RefwCenter = np.int_(Refw / 2)
    RefrowsEnd = Refrows - RefwCenter
    RefcolsEnd = Refcols - RefwCenter
    for od in range(0, Refoutdepth):
        for i in range(RefwCenter,RefrowsEnd):
            for j in range(RefwCenter, RefcolsEnd):
                for m in range(0,Refw):
                    Refii = i - Reftmpii + m
                    Refmm = Refw - 1 - m
                    for n in range(0,Refw):
                        for d in range(0,Refindepth):
                            Refjj = j - Reftmpjj + n
                            Refnn = Refw - 1 - n
                            RefOutput[od, i, j] += Input[d][Refii][Refjj]*kernel[od][d][Refmm][Refnn]
    return RefOutput


#####################################################################
# Main function

@click.command()
@click.option('-rows', type=int, default=8)
@click.option('-cols', type=int, default=8)
@click.option('-indepth', type=int, default=5)
@click.option('-outdepth', type=int, default=10)
@click.option('-w', type=int, default=5)
@click.option('--version',
              type=click.Choice(
                  ('allparallel','outdepthserial','indepthreduce','simple','reference')),
              default='indepthreduce')
@click.option('--verify/--no-verify', default=True)
def cli(rows, cols, indepth, outdepth, w, version, verify):
    """
    Different available versions:
    unoptimized: Run `convolution` without optimizations;
    """

    # Prepare data with numpy
    Input = np.random.rand(indepth, rows, cols).astype(np_dtype)
    kernel = np.random.rand(outdepth, indepth, w, w).astype(np_dtype)
    Output = np.zeros((outdepth, rows, cols), dtype=np_dtype)

    print(f'Convolution {rows}x{cols}x{indepth} with kernel {outdepth}x{w}x{w}x{indepth}(version: {version})')

    if version == 'allparallel':
        # Simply call the program to run it
        convolutionallparallel(Input, kernel, Output)
    elif version == 'outdepthserial':
        convolutionoutdepthserial(Input, kernel, Output)
    elif version == 'indepthreduce':
        convolutionindepthreduce(Input, kernel, Output)
    elif version == 'simple':
        convolutionsimple(Input, kernel, Output)
    else:
        raise ValueError('Invalid version %s' % version)

    if verify:
        #print("From dace:")
        #print(Output)
        expected = refconvolution(Input, kernel)
        #print("From reference")
        #print(expected)
        diff = np.linalg.norm(Output - expected) / (rows * cols * outdepth)
        print('Difference:', diff)
        return 0 if diff <= 1e-6 else 1

if __name__ == "__main__":
    cli()
