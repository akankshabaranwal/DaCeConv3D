# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

# Convolution sample code

import click
import dace
import numpy as np
from pprint import pprint

import dace.libraries.blas

# TODO: Add stride padding parameters
# TODO: Check what should be the expected behaviour for even dimension of filter size

# Define symbolic sizes for arbitrary inputs
rows = dace.symbol('rows')
cols = dace.symbol('cols')
indepth = dace.symbol('indepth')
outdepth = dace.symbol('outdepth')
chunklength = dace.symbol('chunklength', dtype=dace.int64, integer=True, positive=True)

w = dace.symbol('w')

# Define data type to use
dtype = dace.float64
np_dtype = np.float64
#x = dace.DeviceType.GPU
# Simple convolution code using map reduce approach
@dace.program(auto_optimize=True, device=dace.DeviceType.GPU)
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
@dace.program(auto_optimize=True, device=dace.DeviceType.GPU)
def convolutionoutdepthserial(Input: dtype[indepth, rows, cols], kernel: dtype[outdepth, indepth, w, w], Output: dtype[outdepth, rows, cols]):
    for od in range(0,outdepth):
        tmp = np.zeros([rows, cols, indepth * w * w], dtype=Input.dtype)
        for i,j,d,m,n in dace.map[w/2:rows-w/2, w/2:cols-w/2,0:indepth, 0:w, 0:w]:
            with dace.tasklet:
                in_A << Input[d, i - w/2 + m, j - w/2 + n]
                in_B << kernel[od, d, w-1-m, w-1-n]
                out >> tmp[ i, j, (d*(w*w)) + (m*w)+n]

                out = in_A * in_B

        dace.reduce(lambda a,b:a+b, tmp, Output[od,:,:], axis=2, identity=0)


# Simple convolution
@dace.program(auto_optimize=True, device=dace.DeviceType.GPU)
def convolutionsimple(Input: dtype[indepth, rows, cols], kernel: dtype[outdepth, indepth, w, w], Output: dtype[outdepth, rows, cols]):
    Output[:] = 0
    for i,j,d,od,m,n in dace.map[w/2:rows-w/2, w/2:cols-w/2,0:indepth,0:outdepth, 0:w, 0:w]:
            Output[od, i, j] += Input[d, i - w / 2 + m, j - w / 2 + n] * kernel[od, d, w - 1 - m, w - 1 - n]


# Reduction along input depth
@dace.program(auto_optimize=True, device=dace.DeviceType.GPU)
def convolutionindepthreduce(Input: dtype[indepth, rows, cols], kernel: dtype[outdepth, indepth, w, w], Output: dtype[outdepth, rows, cols]):
    for i, j, od in dace.map[w/2:rows-w/2, w/2:cols-w/2, 0:outdepth]:
        tmp = np.zeros([indepth*w*w], dtype = Input.dtype)
        for d,m,n in dace.map[0:indepth,0:w,0:w]:
            with dace.tasklet:
                in_A << Input[d, i - w / 2 + m, j - w / 2 + n]
                in_B << kernel[od, d, w - 1 - m, w - 1 - n]
                out >> tmp[(d*(w*w)) + (m*w)+n]

                out = in_A * in_B
        Output[od,i,j] = dace.reduce(lambda a, b: a + b, tmp, identity=0)


# Split into parallel and non parallel maps
@dace.program(device=dace.DeviceType.GPU)
def convolutionsimpleparallel(Input: dtype[indepth, rows, cols],
                              kernel: dtype[outdepth, indepth, w, w],
                              Output: dtype[outdepth, rows, cols]
                              ):
    Output[:] = 0

    for i, j, od in dace.map[w/2:rows-w/2, w/2:cols-w/2, 0:outdepth]:
        tmp = np.zeros([1], dtype = Input.dtype)
        for d,m,n in dace.map[0:indepth,0:w,0:w]:
            tmp = tmp + Input[d, i - w / 2 + m, j - w / 2 + n] * kernel[od, d, w - 1 - m, w - 1 - n]
        Output[od,i,j] = tmp


# Block parallel computation
@dace.program(auto_optimize=True, device=dace.DeviceType.GPU)
def convolutionblockparallel(Input: dtype[indepth, rows, cols],kernel: dtype[outdepth, indepth, w, w],Output: dtype[outdepth, rows, cols],
                             chunklength
                              ):
    Output[:] = 0
    tmpBlock = np.zeros([chunklength, chunklength], dtype = Input.dtype)
    for i, j, od in dace.map[w/2:rows-w/2, w/2:cols-w/2, 0:outdepth]:
        tmp = np.zeros([1], dtype = Input.dtype)
        for d,m,n in dace.map[0:indepth,0:w,0:w]:
            tmp = tmp + Input[d, i - w / 2 + m, j - w / 2 + n] * kernel[od, d, w - 1 - m, w - 1 - n]
        Output[od,i,j] = tmp


# Normal code for reference
def refconvolutionnew(Input, kernel):
    Refw = kernel.shape[2]
    RefwCenter = np.int_(Refw / 2)
    Refrows = Input.shape[1]
    Refcols = Input.shape[2]
    Refindepth = Input.shape[0]
    Refoutdepth = kernel.shape[0]
    RefOutput = np.zeros((Refoutdepth, Refrows, Refcols), dtype=np_dtype)

    print("Before pad")
    print(Input)
    Input = np.pad(Input, ((0,0),(RefwCenter,RefwCenter),(RefwCenter,RefwCenter)), mode='constant')
    print("After pad")
    print(Input)
    print("Padding done")

    for od in range(0, Refoutdepth):
        for i in range(RefwCenter, Refrows+RefwCenter):
            for j in range(RefwCenter, Refcols+RefwCenter):
                for m in range(0, Refw):
                    for n in range(0, Refw):
                        for d in range(0, Refindepth):
                            RefOutput[od, i-RefwCenter, j-RefwCenter] += Input[d, i+m-RefwCenter, j+n-RefwCenter]*kernel[od, d, m, n]
    return RefOutput

# # Normal code for reference
# def refblockconvolution(Input, kernel):
#     Refw = kernel.shape[2]
#     Refrows = Input.shape[1]
#     Refcols = Input.shape[2]
#     Refindepth = Input.shape[0]
#     Refoutdepth = kernel.shape[0]
#     RefOutput = np.zeros((Refoutdepth, Refrows+1, Refcols+1), dtype=np_dtype)
#     wlen = 4
#     RefwCenter = np.int_(Refw / 2)
#
#     for od in range(0, Refoutdepth):
#         for i in range(RefwCenter, Refrows-RefwCenter, wlen):
#             for j in range(RefwCenter, Refcols-RefwCenter, wlen):
#                 tmp = np.zeros([wlen, wlen], dtype=Input.dtype)
#                 for wr in range(0, wlen):
#                     for wc in range(0, wlen):
#                         if ((i + wr >= 0) and (i + wr < Refrows) and (j + wc >= 0) and (j + wc < Refcols)):
#                             for m in range(0, Refw):
#                                 for n in range(0, Refw):
#                                     for d in range(0,Refindepth):
#                                         if ((i - RefwCenter + m + wr >= 0) and (i - RefwCenter + m + wr < Refrows) and (
#                                                 j - RefwCenter + n + wc >= 0) and (j - RefwCenter + n + wc < Refcols)):
#                                             tmp[wr,wc] += Input[d, i-RefwCenter+m+wr, j-RefwCenter+n+wc]*kernel[od,d,m,n]
#                 print("Block called with: ", i, j)
#                 RefOutput[od, i:i+wlen, j:j+wlen] = tmp
#
#     return RefOutput


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
                            RefOutput[od,i,j] += Input[d,Refii,Refjj]*kernel[od,d,Refmm,Refnn]
    return RefOutput


#####################################################################
# Main function

@click.command()
@click.option('-rows', type=int, default=7)
@click.option('-cols', type=int, default=7)
@click.option('-indepth', type=int, default=1)
@click.option('-outdepth', type=int, default=1)
@click.option('-w', type=int, default=3)
@click.option('--version',
              type=click.Choice(
                  ('allparallel','outdepthserial','indepthreduce','simple','simpleparallel','blockparallel','reference')),
              default='simpleparallel')
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

    #Input = np.ones((indepth, rows, cols), dtype=np_dtype)
    #kernel = np.ones((outdepth, indepth, w, w), dtype=np_dtype)
    #Output = np.zeros((outdepth, rows, cols), dtype=np_dtype)

    print(f'Convolution {rows}x{cols}x{indepth} with kernel {outdepth}x{w}x{w}x{indepth}(version: {version})')

    chunklength = 2
    if version == 'allparallel':
        # Simply call the program to run it
        convolutionallparallel(Input, kernel, Output)
    elif version == 'outdepthserial':
        convolutionoutdepthserial(Input, kernel, Output)
    elif version == 'indepthreduce':
        convolutionindepthreduce(Input, kernel, Output)
    elif version == 'simple':
        convolutionsimple(Input, kernel, Output)
    elif version == 'simpleparallel':
        convolutionsimpleparallel(Input, kernel, Output)
    elif version == 'blockparallel':
        convolutionblockparallel(Input, kernel, Output, chunklength)
    else:
        raise ValueError('Invalid version %s' % version)

    if verify:
        #pprint("From block:")
        #Output = refblockconvolution(Input, kernel)
        #pprint(Output)
        expected = refconvolution(Input, kernel)
        #pprint("From reference")
        #pprint(expected)

        diff = np.linalg.norm(Output - expected) / (rows * cols * outdepth)
        print('Difference:', diff)
        return 0 if diff <= 1e-6 else 1

if __name__ == "__main__":
    cli()
