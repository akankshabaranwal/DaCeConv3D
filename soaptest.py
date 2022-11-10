import dace
import numpy as np
from dace import dtypes
import torch

from dace.transformation.estimator.soap.io_analysis import perform_soap_analysis
from dace.transformation.estimator.soap.utils import get_lead_term
import sympy

# Launch matlab using: /home/akanksha/bin/matlab -nosplash -r "cd '/home/akanksha/Downloads/matlab'; BackgroundSolver();exit"
def soap_analysis(sdfg: dace.SDFG):
    result = perform_soap_analysis(sdfg, generate_schedule=False, solver_timeout=60)
    Q = get_lead_term(result.Q)

    # "Ss": max elements in fast memory
    # Example values! Iterate over arrays in SDFG to determine data type
    bytes_per_element = 4.0
    cache_size = 1024 * 1024
    num_elements = int(cache_size / bytes_per_element)
    
    # SOAP messes with the symbols in the SDFG, e.g., changes the case
    symbol_map = {"Ss": num_elements}
    symbol_map['d_inheight'] = inheight
    symbol_map['d_inwidth'] = inwidth
    symbol_map['d_indepth'] = indepth
    symbol_map['d_inchannels'] = inchannels
    symbol_map['d_outchannels'] = outchannels
    symbol_map['d_batchsize'] = batchsize
    for sym in Q.free_symbols:
        print(sym)
        if str(sym) in sdfg.constants:
            symbol_map[sym] = sdfg.constants[str(sym)]
            continue

        s = str(sym).upper()
        if s in sdfg.constants:
            symbol_map[sym] = sdfg.constants[s]
    
    print(f"AB: symbol map is: {symbol_map}")
    # Now: symbol map contains all known symbol values
    # Try to get the actual value
    print(f"AB: Q is {Q}")
    simplified_Q = sympy.simplify(Q, symbols=symbol_map)
    Q_ = dace.symbolic.evaluate(simplified_Q, symbols=symbol_map)
    return Q_

d_indepth = dace.symbol('d_indepth')
d_inheight = dace.symbol('d_inheight')
d_inwidth = dace.symbol('d_inwidth')
d_inchannels = dace.symbol('d_inchannels')
d_outchannels = dace.symbol('d_outchannels')
d_batchsize = dace.symbol('d_batchsize')
global kdim
kdim = 3

dtype = dace.float32
np_dtype = np.float32

@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=True)
def dace_conv3d( Input: dtype[d_batchsize, d_inchannels, d_indepth, d_inheight, d_inwidth] @dace.StorageType.GPU_Global ,
                kernel: dtype[d_outchannels, d_inchannels, kdim, kdim, kdim] @dace.StorageType.GPU_Global,
                Output: dtype[d_batchsize, d_outchannels, d_indepth-kdim+1, d_inheight-kdim+1, d_inwidth-kdim+1] @dace.StorageType.GPU_Global):
    for n, d, h, w, oc in dace.map[0:d_batchsize, 0:d_indepth-kdim+1, 0:d_inheight-kdim+1, 0:d_inwidth-kdim+1, 0:d_outchannels]:
        r_tmp = np.zeros([1], dtype=Input.dtype)
        for kd, kh, kw, ic in dace.map[0:kdim, 0:kdim, 0:kdim, 0:d_inchannels]:
            r_tmp = r_tmp + Input[n, ic, d+kd, h+kh, w+kw] * kernel[oc, ic, kd, kh, kw]
        Output[n, oc, d, h, w] = r_tmp

inchannels = 4
indepth = 128
inheight = 128
inwidth = 128
outchannels = 16
outdepth = indepth - kdim + 1
outheight = inheight - kdim + 1
outwidth = inheight - kdim + 1
batchsize = 4

d_input = torch.rand(batchsize, inchannels, indepth, inheight, inwidth).cuda()
d_kernel = torch.rand(outchannels, inchannels, kdim, kdim, kdim).cuda()
d_output = torch.zeros(batchsize, outchannels, outdepth, outheight, outwidth).cuda()
    
sdfg_fun: dace.SDFG = dace_conv3d.to_sdfg(d_input, d_kernel, d_output)

dace.Config.set('compiler', 'default_data_types', value='C')
    # Fuse the map and reduce nodes
    # Apply GPU transformation
    #sdfg.apply_transformations_repeated(StateFusion)
    #sdfg.simplify()
sdfg_fun.apply_gpu_transformations()

# d_inchannels = inchannels
# d_outchannels = outchannels
# d_indepth = indepth
# d_inheight = inheight
# d_inwidth = inwidth 
# d_batchsize = batchsize

sdfg_fun(Input=d_input, kernel=d_kernel, Output=d_output, 
             d_inchannels = inchannels, d_batchsize = batchsize, d_outchannels = outchannels,
             d_indepth = indepth, d_inheight = inheight, d_inwidth = inwidth, 
             )
return_Q = soap_analysis(sdfg_fun)
print(return_Q)