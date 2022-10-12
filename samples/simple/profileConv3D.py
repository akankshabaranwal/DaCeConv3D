# abaranwal: Code to profile 3D convolution based on daceml functions

from copy import deepcopy
from conv3D import *
from daceml.testing.profiling import time_funcs, print_time_statistics
import argparse
import time
import math
import os
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
import csv

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-paramscsv','--paramscsv', type=str, default='cosmoflow', help='select which csv to profile')
parser.add_argument('--verify', action='store_true', help='run verification')
parser.add_argument('--compareprof', action='store_true', help='time funcs comparative summary time')
parser.add_argument('--proftf', action='store_true', help='run tf code with markers')
parser.add_argument('--setlaunchwait', action='store_true', help='set launch wait')
parser.add_argument('--profoptimdace', action='store_true', help='run dace code with markers')
parser.add_argument('--enableplots', action='store_true', help='disable creating plots')

parser.add_argument('-warmupiter','--warmupiter', type=int, default=10, help='set number of warmup iterations')
parser.add_argument('-totaliter','--totaliter', type=int, default=100, help='set number of total iterations')
parser.add_argument('-lastlayer','--lastlayer', type=int, default=1, help='set number of total iterations')

# Charts
# TODO: Fix the bar plot to violin plot instead
# TODO: Check if you can plot and compare different versions of optimizations

# FIXME: Something is wrong with subplot when the csv file has just one row  
# FIXME: verif fails on ault23 
args = parser.parse_args()

paramscsv = args.paramscsv
convparams =  parsecsv(paramscsv)
verify = args.verify
compareprof = args.compareprof
runtf = False
rundace = False
runoptimdace = False
proftf = args.proftf
profdace = False
profoptimdace = args.profoptimdace
setlaunchwait = args.setlaunchwait
warmupiter = args.warmupiter
totaliter = args.totaliter
currlayer = 0
enableplots = args.enableplots
#lastlayer = convparams.shape[0]
lastlayer = args.lastlayer

#outdir = f'./outputplots/out{math.floor(time.time())}'
outdir = f'./outputplots/_out'
#os.mkdir(outdir)
with open(f'./{outdir}/params.txt', 'w') as f:
    f.writelines(f'csv: {paramscsv}\n')
    f.writelines(f'warmup iteration: {warmupiter}\n')
    f.writelines(f'total iteration: {totaliter}\n')
    f.writelines(f'set launch wait: {setlaunchwait}\n')

args = parser.parse_args()
d_input, d_kernel, d_output, inchannels, indepth, inheight, inwidth, outchannels, batchsize = prepareinputs(convparams.iloc[0])

## Prepare inputs for tensorflow fun
tmp_input = d_input.cpu().clone()
tmp_kernel = d_kernel.cpu().clone()
t_input = tf.convert_to_tensor(tmp_input.detach().numpy())
t_kernel = tf.convert_to_tensor(tmp_kernel.detach().numpy())

inchannels = np.int32(inchannels)
indepth = np.int32(indepth)
inheight = np.int32(inheight)
inwidth = np.int32(inwidth)
outchannels = np.int32(outchannels)
batchsize = np.int32(batchsize)

sdfg_fun: dace.SDFG = dace_conv3d.to_sdfg(d_input, d_kernel, d_output)

# Apply optimizations
optimize_for_gpu(sdfg_fun)
optim_dace = sdfg_fun.compile()

# Function call for original dace conv3D

# Function call for tensorflow 3D conv
def run_tf():
    op=tf.nn.conv3d(t_input, t_kernel, strides=[1, 1, 1, 1, 1], padding="VALID")
 
# Function calls to run the optim dace function
def run_optim_dace():
    optim_dace(Input=d_input, kernel=d_kernel, Output=d_output,d_inchannels=inchannels, d_indepth=indepth, d_inheight=inheight,d_inwidth=inwidth, d_outchannels=outchannels, d_batchsize=batchsize)

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

def run_fun(fun_name):
    for i in range(0,warmupiter):
        fun_name()
    for i in range(0, totaliter):
        fun_name()
    
median_dace = []
median_tf = []
layer_names = []
csv_columns = ['layer_name','dace_median','tf_median']
summary = []
for layern in range(currlayer, lastlayer):
    d_input, d_kernel, d_output, inchannels, indepth, inheight, inwidth, outchannels, batchsize = prepareinputs(convparams.iloc[layern])
    layersummary = {}
    ## Prepare inputs for tensorflow fun ABCD
    tmp_input = d_input.cpu().clone()
    tmp_kernel = d_kernel.cpu().clone()
    t_input = tf.convert_to_tensor(tmp_input.detach().numpy())
    t_kernel = tf.convert_to_tensor(tmp_kernel.detach().numpy())

    inchannels = np.int32(inchannels)
    indepth = np.int32(indepth)
    inheight = np.int32(inheight)
    inwidth = np.int32(inwidth)
    outchannels = np.int32(outchannels)
    batchsize = np.int32(batchsize)
    layer_name = f'in_{batchsize}X{indepth}X{inheight}X{inwidth}X{inchannels}_k_{kdim}X{kdim}X{kdim}_och_{outchannels}'
    print(f'INFO: {layer_name}') 
    layer_names.append(layer_name)
    
    #TODO: incorrect answer on ault 23, output is all zeroes. why?
    # Code for verification
    if verify:
        print("INFO: Running verification to compare against tensorflow output")
        refop = tf.nn.conv3d(t_input, t_kernel, strides=[1, 1, 1, 1, 1], padding="VALID")
        optim_dace(Input=d_input, kernel=d_kernel, Output=d_output,d_inchannels=inchannels, d_indepth=indepth, d_inheight=inheight,d_inwidth=inwidth, d_outchannels=outchannels, d_batchsize=batchsize)
        tmp_output = d_output.cpu()
        opdace = tf.convert_to_tensor(tmp_output.detach().numpy())
        diff = np.linalg.norm(opdace - refop) / (batchsize * outchannels * indepth * inheight * inwidth )
        print('Difference between tensorflow and dace values:', diff)
        if(diff<=1e-4):
            print(f"Verification successfull")
        else:
            print(f"!!! ERROR: Incorrect verification")

    # Profiling tensorflow using run
    if runtf:
        run_fun(run_tf)
    
    # Profiling optimized dace using run
    if runoptimdace:
        run_fun(run_optim_dace)

    # Comparitive profiling using time funcs
    print(f"INFO: Warmup for {warmupiter} iterations and total iterations {totaliter}")
    print(f"INFO: Statistics for layer number {layern}")
    dace_median = []
    tf_median = []
    if compareprof:
        times = time_funcs([run_optim_dace, run_tf],
                        func_names=["dace", "tf"],
                        warmups=warmupiter,
                        num_iters=totaliter,
                        launch_wait=setlaunchwait)
        print_time_statistics(times, [ "dace", "tf"])
        layersummary['layer_name'] = layer_name
        layersummary['times'] = times
        layersummary['dace_median'] = statistics.median(times[0])
        layersummary['tf_median'] = statistics.median(times[1])
        median_dace.append(statistics.median(times[0]))
        median_tf.append(statistics.median(times[1]))
        dace_median.append(times[0])
        tf_median.append(times[1])
        summary.append(layersummary)

    dace_median_array = np.array(dace_median)
    tf_median_array = np.array(tf_median)

    # run using prof for tf
    if proftf:
        times = time_funcs([run_tf],
                        func_names=["tf"],
                        warmups=warmupiter,
                        num_iters=totaliter,
                        launch_wait=setlaunchwait)
        print_time_statistics(times, [ "tf"])

    # run using prof for dace
    if profoptimdace:
        times = time_funcs([run_optim_dace],
                        func_names=["optimdace"],
                        warmups=warmupiter,
                        num_iters=totaliter,
                        launch_wait=setlaunchwait)
        print_time_statistics(times, [ "optimdace"])

#TODO: create summary csv with all values mean median mode
def addlabels(x,y):
    for i in range(len(x)):
        y[i] =round(y[i],2)
        plt.text(i,y[i],y[i])

csv_data = deepcopy(summary)
csv_file = f'{outdir}/summary.csv'
with open(csv_file, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()
    for data in csv_data:
        del data['times']
        writer.writerow(data)

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
        d = {'tensorflow': pd.Series(times[1]), 'dace': pd.Series(times[0])}
        df = pd.DataFrame(d)
        if (col==ncol):
            row = row+1
            col = 0
        if (row==nrow):
            print("WARNING: Some plots could not be generated")
            exit()
        axtf = sns.violinplot(ax = axes[0, col], y=df["tensorflow"], cut=0, color = 'pink')
        axtf.set(xlabel=f'{paramscsv}layer{layern}', ylabel='tensorflow runtime in ms')
        axd = sns.violinplot(ax = axes[1, col], y=df["dace"], cut=0, color = 'skyblue')
        axd.set( ylabel=f'dace runtime in ms', xlabel=  f'{paramscsv}layer{layern}')
        layern = layern+1
        col = col+1

        layer_name_column = [f'layer{layern}']*totaliter
        fun_name_column = ['dace']*totaliter
        data_dace = {'layer_name': layer_name_column, 'fun_name': fun_name_column, 'times': times[0]}
        df_dace = pd.DataFrame(data_dace)
        layer_name_column = [f'layer{layern}']*totaliter
        fun_name_column = ['tensorflow']*totaliter
        data_tf = {'layer_name': layer_name_column, 'fun_name': fun_name_column, 'times': times[1]}
        df_tf = pd.DataFrame(data_tf)
        
        full_df = pd.concat([full_df, df_tf, df_dace], ignore_index=True)


    figurename = f'{outdir}/separate_runtime.png'
    fig.savefig(figurename)
    plt.cla()
    plt.clf()

    #sns.set(style="darkgrid")
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
    if len(median_dace) != 0 and len(median_tf) !=0:
        print("INFO: Plotting summary graph")
        # set width of bar
        barWidth = 0.2
        fig = plt.subplots(figsize =(12, 8))
        # Set position of bar on X axis
        br1 = np.arange(len(median_tf))
        br2 = [x + barWidth for x in br1]
        
        # Make the plot
        plt.bar(br1, median_tf, color ='pink', width = barWidth, edgecolor ='grey', label ='tensorflow')
        plt.bar(br2, median_dace, color ='skyblue', width = barWidth, edgecolor ='grey', label ='dace')
        addlabels(br1, median_tf)
        addlabels(br2, median_dace)
        # Adding Xticks
        plt.xlabel('Variation across different layers', fontweight ='bold', fontsize = 15)
        plt.ylabel('Median runtime in ms', fontweight ='bold', fontsize = 15)
        plt.xticks([r + barWidth for r in range(len(median_dace))], layer_names)
        plt.legend()
        plt.xticks(rotation=45, ha='right')
        plt.savefig(f'{outdir}/median_runtime', bbox_inches='tight')

    elif len(median_dace)!=0 or len(median_tf)!=0:
        print("INFO: Plot single function graph")