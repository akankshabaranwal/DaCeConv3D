# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

# Convolution sample code
# An image in(x,y) with a kernel kernel(x,y).
#
# Produce a new image out(x,y)
'''
kCenterX = kCols / 2;
kCenterY = kRows / 2;
for (i = 0; i < rows; ++i) // rows  {
    for (j = 0; j < cols; ++j) //columns    {
        for (m = 0; m < kRows; ++m)        {
            mm = kRows - 1 - m; // row index of flipped kernel
            for ( n = 0; n < kCols; ++n )            {
                nn = kCols - 1 - n;
                ii = i + (kCenterY - mm);
                jj = j + (kCenterX - nn);
                if ( ii >= 0 && ii < rows && jj >=0 && jj < cols )
                    out[i][j] += in[ii][jj] * kernel[mm][nn];
            }
        }
    }
}
'''

import click
import dace
import numpy as np

import dace.libraries.blas

# Define symbolic sizes for arbitrary inputs
rows = dace.symbol('rows')
cols = dace.symbol('cols')
w = dace.symbol('w')

# Define data type to use
dtype = dace.float64
np_dtype = np.float64

# Simple unoptimized convolution code
@dace.program
def convolution2D(Input: dtype[rows, cols], kernel: dtype[w, w], Output: dtype[rows, cols]):
    tmp = np.zeros([rows, cols, w*w], dtype = Input.dtype)
    for i,j,m,n in dace.map[w/2:rows-w/2, w/2:cols-w/2, 0:w, 0:w]:
        with dace.tasklet:
            in_A << Input[i - w/2 - 1 - m, j - w/2 -1 - n]
            in_B << kernel[w - 1 - m, w - 1 - n]
            out >> tmp[i, j, m*w+n]

            out = in_A * in_B

    dace.reduce(lambda a,b:a+b, tmp, Output, axis=2, identity=0)


# Normal code for reference
def refconvolution2D(Input, kernel):
    RefOutput = np.zeros_like(Input)
    Refw = kernel.shape[0]
    Refrows = Input.shape[0]
    Refcols = Input.shape[1]
    Reftmpii = np.int_(Refw/2 + 1)
    Reftmpjj = np.int_(Refw/2 + 1)
    RefwCenter = np.int_(Refw / 2)
    RefrowsEnd = Refrows - RefwCenter
    RefcolsEnd = Refcols - RefwCenter
    for i in range(RefwCenter,RefrowsEnd):
        for j in range(RefwCenter, RefcolsEnd):
            RefOutput[i,j] = 0
            for m in range(0,Refw):
                Refii = i - Reftmpii - m
                Refmm = Refw - 1 - m
                for n in range(0,Refw):
                    Refjj = j - Reftmpjj - n
                    Refnn = Refw - 1 - n
                    RefOutput[i,j] += Input[Refii][Refjj]*kernel[Refmm][Refnn]
    return RefOutput


#####################################################################
# Main function

@click.command()
@click.option('-rows', type=int, default=5)
@click.option('-cols', type=int, default=5)
@click.option('-w', type=int, default=2)
@click.option('--version',
              type=click.Choice(
                  ('unoptimized','reference')),
              default='unoptimized')
@click.option('--verify/--no-verify', default=True)
def cli(rows, cols, w, version, verify):
    """
    Different available versions:
    unoptimized: Run `convolution2D` without optimizations;
    """

    # Prepare data with numpy
    Input = np.random.rand(rows, cols).astype(np_dtype)
    kernel = np.random.rand(w, w).astype(np_dtype)
    Output = np.zeros((rows, cols), dtype=np_dtype)

    print(f'Convolution 2D {rows}x{cols}x{1} with kernel {w}x{w}(version: {version})')

    if version == 'unoptimized':
        # print("skipped call")
        # Simply call the program to run it
        convolution2D(Input, kernel, Output)
    else:
        raise ValueError('Invalid version %s' % version)

    if verify:
        expected = refconvolution2D(Input, kernel)
        diff = np.linalg.norm(Output - expected) / (rows * cols)
        print('Difference:', diff)
        return 0 if diff <= 1e-6 else 1

if __name__ == "__main__":
    cli()
