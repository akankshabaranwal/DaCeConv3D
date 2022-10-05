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
from dace.transformation.interstate import StateFusion

import matplotlib.pyplot as plt
import seaborn as sns
import torch.cuda.profiler as profiler
from typing import Tuple
import torch

from dace.transformation.dataflow import MapTiling, MapExpansion, MapCollapse, MapCollapse, MapExpansion
from dace.transformation.optimizer import Optimizer
from dace.transformation.auto import auto_optimize
from dace import dtypes

# Define constants for filter dimensions
global kdim
kdim = 3

#####################################################################
# Data-centric optimization helpers copied from matmul.py 
def find_map_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    return next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry) and pname in n.params)

def find_map_and_state_by_param(sdfg: dace.SDFG, pname: str) -> Tuple[dace.nodes.MapEntry, dace.SDFGState]:
    """ Finds the first map entry node by the given parameter name. """
    return next(
        (n, p) for n, p in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry) and pname in n.params)

def find_mapentry_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
    """ Finds the first map exit node by the given parameter name. """
    entry, state = find_map_and_state_by_param(sdfg, pname)
    return state.entry_node(entry)

def find_mapexit_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapExit:
    """ Finds the first map exit node by the given parameter name. """
    entry, state = find_map_and_state_by_param(sdfg, pname)
    return state.exit_node(entry)

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

def find_map_by_name(sdfg: dace.SDFG, name: str) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    return next((n, s) for n, s in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.MapEntry) and n.label == name)
import re
# Optimize code on the GPU
def optimize_for_gpu(sdfg: dace.SDFG):
    """ Optimize 3D convolution example for GPUs. """
    dace.Config.set('compiler', 'default_data_types', value='C')
    # Fuse the map and reduce nodes
    # Apply GPU transformation
    sdfg.apply_transformations_repeated(StateFusion)
    sdfg.simplify()
    sdfg.apply_gpu_transformations()
    
    #return
    # Expand the maps
    m_expandparams = find_map_by_param(sdfg, 'd')
    MapExpansion.apply_to(sdfg, map_entry=m_expandparams)
    # # Collapse the maps grouped appropriately
    m_d = find_map_by_param(sdfg, 'd')
    m_h = find_map_by_param(sdfg, 'h')
    MapCollapse.apply_to(sdfg, outer_map_entry=m_d, inner_map_entry=m_h)
    m_d = find_map_by_param(sdfg, 'd')
    m_w = find_map_by_param(sdfg, 'w')
    MapCollapse.apply_to(sdfg, outer_map_entry=m_d, inner_map_entry=m_w)
    m_w = find_map_by_param(sdfg, 'w')
    mapname = 'conv3D_dace_conv3d_130_d'
    for xform in Optimizer(sdfg).get_pattern_matches(patterns=[MapTiling]):
       print('Match:', xform.print_match(sdfg))
       matches = xform.print_match(sdfg)
       nameconv = re.match(r'MapTiling in \[MapEntry \((.*)_d\[d=0:d_indepth - 2, h=0:d_inheight - 2, w=0:d_inwidth - 2\].*', matches, flags=0)
       if(nameconv):
        mapname = f'{nameconv.group(1)}_d'
        break
    
    # Tiling
    state = sdfg.node(0)
    conv_exit = next(n for n in state.nodes() if isinstance(n, dace.nodes.MapExit) and n.label == mapname)
    conv_entry = next(n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry) and n.label == mapname)
    MapTiling.apply_to(sdfg, map_entry = conv_entry, map_exit = conv_exit)
    m_d = find_map_by_param(sdfg, 'd')
    m_d.map.schedule = dace.ScheduleType.GPU_ThreadBlock
    m_tiled = find_map_by_param(sdfg, 'tile_d')
    m_tiled.map.schedule = dace.ScheduleType.GPU_ThreadBlock

    return

    

# Simple parallel 3D convolution
@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=True)
def dace_conv3d( Input: dtype[d_batchsize, d_indepth, d_inheight, d_inwidth, d_inchannels] @dace.StorageType.GPU_Global,
                kernel: dtype[kdim, kdim, kdim, d_inchannels, d_outchannels] @dace.StorageType.GPU_Global,
                Output: dtype[d_batchsize, d_indepth-kdim+1, d_inheight-kdim+1, d_inwidth-kdim+1, d_outchannels] @dace.StorageType.GPU_Global):
    for n, d, h, w, oc in dace.map[0:d_batchsize, 0:d_indepth-kdim+1, 0:d_inheight-kdim+1, 0:d_inwidth-kdim+1, 0:d_outchannels]:
        r_tmp = np.zeros([1], dtype=Input.dtype)
        for kd, kh, kw, ic in dace.map[0:kdim, 0:kdim, 0:kdim, 0:d_inchannels]:
            r_tmp = r_tmp + Input[n, d+kd, h+kh, w+kw, ic] * kernel[kd, kh, kw, ic, oc]
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
    return df['Runtime_sec']

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
    Input = torch.rand(batchsize, indepth, inheight, inwidth, inchannels).cuda()
    kernel = torch.rand(kdim, kdim, kdim, inchannels, outchannels).cuda()
    Output = torch.zeros(batchsize, outdepth, outheight, outwidth, outchannels).cuda()
    
    print(f'\n***** \n***** \n Parsed 3D Convolution Input parameters {inchannels}x{indepth}x{inheight}x{inwidth} '
            f'and Kernel parameters {outchannels}x{inchannels}x{kdim}x{kdim}x{kdim}')
    return Input, kernel, Output, inchannels, indepth, inheight, inwidth, outchannels, batchsize