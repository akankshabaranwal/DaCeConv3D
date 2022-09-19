# abaranwal: Code to profile 3D convolution based on daceml functions

from conv3D import *
from daceml.testing.profiling import time_funcs, print_time_statistics
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-csv','--csv', type=str, default='cosmoflow', help='select which csv to profile')
parser.add_argument('-warmupiter','--warmupiter', type=int, default=10, help='set number of warmup iterations')
parser.add_argument('-totaliter','--totaliter', type=int, default=100, help='set number of total iterations')
# Flags to set kind of profiling
parser.add_argument('--verify', action='store_true', help='run verification')
parser.add_argument('--profileendtoend', action='store_true', help='run end to end profiling')
parser.add_argument('--profiletimefuncs', action='store_true', help='time funcs comparative summary time')
parser.add_argument('--runtf', action='store_true', help='run tf code with warmup')
parser.add_argument('--rundace', action='store_true', help='run dace code with warmup')
parser.add_argument('--runoptimizeddace', action='store_true', help='run optimized dace code with warmup')
parser.add_argument('--timefuncstf', action='store_true', help='run tf code with markers')
parser.add_argument('--timefuncsdace', action='store_true', help='run dace code with markers')
parser.add_argument('--timefuncsoptimizeddace', action='store_true', help='run dace code with markers')

args = parser.parse_args()
csv = args.csv
convparams =  parsecsv(csv)
verify = args.verify
profileendtoend = args.profileendtoend
profiletimefuncs = args.profiletimefuncs
runtf = args.runtf
rundace = args.rundace
runoptimizeddace = args.runoptimizeddace
timefuncstf = args.timefuncstf
timefuncsdace = args.timefuncsdace
timefuncsoptimizeddace = args.timefuncsoptimizeddace
warmupiter = args.warmupiter
totaliter = args.totaliter
currlayer = 0
lastlayer = convparams.shape[0]

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
sdfg_fun1: dace.SDFG = dace_conv3d.to_sdfg(d_input, d_kernel, d_output)
baseline_dace = sdfg_fun1.compile()
# Apply optimizations
optimize_for_gpu(sdfg_fun)
optimized_dace = sdfg_fun.compile()

# Function call for original dace conv3D
def run_dace():
    baseline_dace(Input=d_input, kernel=d_kernel, Output=d_output,d_inchannels=inchannels, d_indepth=indepth, d_inheight=inheight,d_inwidth=inwidth, d_outchannels=outchannels, d_batchsize=batchsize)

# Function call for tensorflow 3D conv
def run_tf():
    op=tf.nn.conv3d(t_input, t_kernel, strides=[1, 1, 1, 1, 1], padding="VALID")
 
# Function calls to run the optimized dace function
def run_optimized_dace():
    optimized_dace(Input=d_input, kernel=d_kernel, Output=d_output,d_inchannels=inchannels, d_indepth=indepth, d_inheight=inheight,d_inwidth=inwidth, d_outchannels=outchannels, d_batchsize=batchsize)

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


for layern in range(currlayer, lastlayer):
    d_input, d_kernel, d_output, inchannels, indepth, inheight, inwidth, outchannels, batchsize = prepareinputs(convparams.iloc[layern])

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

    # Code for verification
    if verify:
        print("INFO: Running verification to compare against tensorflow output")
        refop = tf.nn.conv3d(t_input, t_kernel, strides=[1, 1, 1, 1, 1], padding="VALID")
        optimized_dace(Input=d_input, kernel=d_kernel, Output=d_output,d_inchannels=inchannels, d_indepth=indepth, d_inheight=inheight,d_inwidth=inwidth, d_outchannels=outchannels, d_batchsize=batchsize)
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
        for i in range(0,warmupiter):
            run_tf()
        for i in range(0, totaliter):
            run_tf()
    
    # Profiling baseline dace using run
    if rundace:
        for i in range(0, warmupiter):
            run_dace()
        for i in range(0, totaliter):
            run_dace()

    # Profiling optimized dace using run
    if runoptimizeddace:
        for i in range(0, warmupiter):
            run_optimized_dace()
        for i in range(0, totaliter):
            run_optimized_dace()

    # Comparitive profiling using time funcs
    if profiletimefuncs:
        times = time_funcs([run_optimized_dace, run_tf],
                        func_names=["optimizeddace", "tf"],
                        warmups=warmupiter,
                        num_iters=totaliter,
                        launch_wait=True)
        print(f"INFO: Statistics for layer number {layern}")
        print_time_statistics(times, [ "optimizeddace", "tf"])

    # run using timefuncs for tf
    if timefuncstf:
        print(f"INFO: Warmup for {warmupiter} iterations and total iterations {totaliter}")
        times = time_funcs([run_tf],
                        func_names=["tf"],
                        warmups=warmupiter,
                        num_iters=totaliter)
        print(f"INFO: Statistics for layer number {layern}")
        print_time_statistics(times, [ "tf"])

     # run using timefuncs for dace
    if timefuncsdace:
        print(f"INFO: Warmup for {warmupiter} iterations and total iterations {totaliter}")
        times = time_funcs([run_dace],
                        func_names=["baselinedace"],
                        warmups=warmupiter,
                        num_iters=totaliter,
                        launch_wait=True)
        print(f"INFO: Statistics for layer number {layern}")
        print_time_statistics(times, [ "baselinedace"])

    # run using timefuncs for dace
    if timefuncsoptimizeddace:
        print(f"INFO: Warmup for {warmupiter} iterations and total iterations {totaliter}")
        times = time_funcs([run_optimized_dace],
                        func_names=["optimizeddace"],
                        warmups=warmupiter,
                        num_iters=totaliter,
                        launch_wait=True)
        print(f"INFO: Statistics for layer number {layern}")
        print_time_statistics(times, [ "optimizeddace"])


    # Code for end to end runtime
    if profileendtoend:
        print(f"INFO: End to end profiling")
        print(f"INFO: Warmup for {warmupiter} iterations and total iterations {totaliter}")        

        ## TF warm up
        timeit.Timer(lambda: timetfgpu_conv3D(t_input, t_kernel)).repeat(repeat=5, number=1)
        tf_runtime = timeit.Timer(lambda: timetfgpu_conv3D(t_input, t_kernel)).repeat(repeat=100, number=2)
        print("INFO: Tensorflow in ms: ", (np.median(tf_runtime))*1000)

        # print("*\n*\n*")
        # print("INFO: Running baseline sdfg code")
        # rundaceprofiling(run_dace, warmupiter)
        # dace_runtime = rundaceprofiling(run_dace, totalite)
        # print("INFO: Dace time in ms: ",np.median(dace_runtime)*1000)

        print("*\n*\n*")
        print("INFO: Running optimized sdfg code")
        rundaceprofiling(run_optimized_dace, warmupiter)
        dace_optimized = rundaceprofiling(run_optimized_dace, totaliter)
        print("INFO: Dace time in ms: ",np.median(dace_optimized)*1000)
