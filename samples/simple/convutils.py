# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
# Convolution benchmarking and verif code for 3D convolution

import click
import dace
import numpy as np
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
import torch.nn.functional as F

from dace.transformation.dataflow import MapTiling, MapExpansion, MapCollapse, MapCollapse, MapExpansion, MapInterchange
from dace.transformation import helpers as xfutil
from dace.transformation.optimizer import Optimizer
from dace.transformation.auto import auto_optimize
from copy import deepcopy
import csv

import sys

#TODO: Recheck the cosmoflow parameters

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

def find_map_by_name(sdfg: dace.SDFG, name: str) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    return next((n, s) for n, s in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.MapEntry) and n.label == name)


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

# Dace profiling method, Returns median values in ms
def rundaceprofiling(run_dace_fun, reps):
    # Temporarily set the DACE_profiling config to True
    with dace.config.set_temporary('profiling', value=True):
        # You can control the number of times a program is run with the treps configuration
        with dace.config.set_temporary('treps', value=reps):
            run_dace_fun()
    list_of_files = glob.glob(f'.dacecache/*/profiling/results-*.csv')
    latest_file = max(list_of_files, key=os.path.getctime)
    df = pd.read_csv(latest_file)
    return df['Runtime_sec']

# Place holder function for pytorch reference code for profiling.
def timetorchgpu_conv3D(input, filter):
    op=F.conv3d(input, filter, stride=1, padding='valid')

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


# Data layout is NCDHW for pytorch
def prepareinputs(currconv, layout):
    inchannels = currconv["InChannel"]
    indepth = currconv["InputDepth"]
    inheight = currconv["InputHeight"]
    inwidth = currconv["InputWidth"]
    outchannels = currconv["OutputChannel"]
    kdim = 3
    outdepth = indepth - kdim + 1
    outheight = inheight - kdim + 1
    outwidth = inheight - kdim + 1
    batchsize = 16 # Maximum experimentally runnable batch size is 16. Theoretical is 64.
    if layout == 'NCDHW':
        # Prepare data with pytorch
        Input = torch.rand(batchsize, inchannels, indepth, inheight, inwidth).cuda()
        kernel = torch.rand(outchannels, inchannels, kdim, kdim, kdim).cuda()
        Output = torch.zeros(batchsize, outchannels, outdepth, outheight, outwidth).cuda()
        print(f'\n***** \n***** \n Parsed 3D Convolution Input parameters {inchannels}x{indepth}x{inheight}x{inwidth} '
                f'and Kernel parameters {outchannels}x{inchannels}x{kdim}x{kdim}x{kdim}')
    elif layout == 'NDHWC':
        # For implicit gemm
        Input = torch.rand(batchsize, indepth, inheight, inwidth, inchannels).cuda()
        kernel = torch.rand(outchannels, kdim, kdim, kdim, inchannels).cuda()
        Output = torch.zeros(batchsize, outdepth, outheight, outwidth, outchannels).cuda()
    else:
        sys.exit("!!! ERROR: Layout not yet implemented")
    
    return Input, kernel, Output, inchannels, indepth, inheight, inwidth, outchannels, batchsize, kdim
    

def run_fun(fun_name, warmupiter, totaliter):
    for i in range(0,warmupiter):
        fun_name()
    for i in range(0, totaliter):
        fun_name()


# Helper functions to create summary plots, csvs to analyze performance

def createsummaryfile(summary, outdir, csv_columns):
    csv_data = deepcopy(summary)
    csv_file = f'{outdir}/summary.csv'
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in csv_data:
            del data['times']
            writer.writerow(data)

def addlabels(x,y):
    for i in range(len(x)):
        y[i] =round(y[i],2)
        plt.text(i,y[i],y[i])


def createplots(enableplots, lastlayer, currlayer, warmupiter, totaliter, paramscsv, outdir, median_dace, median_cudnn, layer_names, summary):
    nrow = 2
    ncol = lastlayer-currlayer
    row = 0
    col = 0
    fig, axes = plt.subplots(nrow, ncol, figsize=(18, 10), sharey=True, sharex=True)
    full_df = pd.DataFrame(columns = ['layer_name','fun_name','times'])
    if enableplots:
        layern = 0
        for layersummary in summary:
            times = layersummary["times"]
            layer_name = layersummary["layer_name"]
            d = {'cudnn': pd.Series(times[1]), 'dace': pd.Series(times[0])}
            df = pd.DataFrame(d)
            if (col==ncol):
                row = row+1
                col = 0
            if (row==nrow):
                print("WARNING: Some plots could not be generated")
                exit()
            axtorch = sns.violinplot(ax = axes[0, col], y=df["cudnn"], cut=0, color = 'pink')
            axtorch.set(xlabel=f'{paramscsv}layer{layern}', ylabel='cudnn runtime in ms')
            axd = sns.violinplot(ax = axes[1, col], y=df["dace"], cut=0, color = 'skyblue')
            axd.set( ylabel=f'dace runtime in ms', xlabel=  f'{paramscsv}layer{layern}')
            layern = layern+1
            col = col+1

            layer_name_column = [f'layer{layern}']*totaliter
            fun_name_column = ['dace']*totaliter
            data_dace = {'layer_name': layer_name_column, 'fun_name': fun_name_column, 'times': times[0]}
            df_dace = pd.DataFrame(data_dace)
            layer_name_column = [f'layer{layern}']*totaliter
            fun_name_column = ['cudnn']*totaliter
            data_torch = {'layer_name': layer_name_column, 'fun_name': fun_name_column, 'times': times[1]}
            df_torch = pd.DataFrame(data_torch)
            
            full_df = pd.concat([full_df, df_torch, df_dace], ignore_index=True)


        figurename = f'{outdir}/separate_runtime.png'
        fig.savefig(figurename)
        plt.cla()
        plt.clf()

        full_df[["times"]] = full_df[["times"]].apply(pd.to_numeric)
        
        # Grouped violinplot
        ax = sns.violinplot(x="layer_name", y="times", hue="fun_name", data=full_df, palette="Pastel1", cut=0, scale='width', inner='quartile',split=True)
        ax.set(ylabel='Runtime in ms')
        ax.set(xlabel=f'Variation across {paramscsv} layers for warmup iterations {warmupiter} and total iterations {totaliter}')
        fig = ax.get_figure()
        figurename = f'{outdir}/violin_allruntime.png'
        plt.savefig(figurename)
        
        plt.cla()
        plt.clf()
        if len(median_dace) != 0 and len(median_cudnn) !=0:
            print("INFO: Plotting summary graph")
            # set width of bar
            barWidth = 0.2
            fig = plt.subplots(figsize =(12, 8))
            # Set position of bar on X axis
            br1 = np.arange(len(median_cudnn))
            br2 = [x + barWidth for x in br1]
            
            # Make the plot
            plt.bar(br1, median_cudnn, color ='pink', width = barWidth, edgecolor ='grey', label ='cudnn')
            plt.bar(br2, median_dace, color ='skyblue', width = barWidth, edgecolor ='grey', label ='dace')
            addlabels(br1, median_cudnn)
            addlabels(br2, median_dace)
            # Adding Xticks
            plt.xlabel('Variation across different layers', fontweight ='bold', fontsize = 15)
            plt.ylabel('Median runtime in ms', fontweight ='bold', fontsize = 15)
            plt.xticks([r + barWidth for r in range(len(median_dace))], layer_names)
            plt.legend()
            plt.xticks(rotation=45, ha='right')
            plt.savefig(f'{outdir}/median_runtime', bbox_inches='tight')

        elif len(median_dace)!=0 or len(median_cudnn)!=0:
            print("!!ERROR: Plotting single function graph is not implemented")
