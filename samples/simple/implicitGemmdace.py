import dace
import numpy as np
from dace import dtypes

# Define constants for filter dimensions
global kdim
kdim = 3

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
