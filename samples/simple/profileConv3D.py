# abaranwal: Code to profile 3D convolution based on daceml functions

from conv3D import *
from daceml.testing.profiling import time_funcs, print_time_statistics
import argparse

# Prepare inputs and kernel
csv = 'cosmoflow'
warmupiter = 10
totaliter = 100
convparams =  parsecsv(csv)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-currlayer','--cl', type=int, required=True, help='select which layer to profile')
# Flags to set kind of profiling
parser.add_argument('--verify', action='store_true', help='run verification')
parser.add_argument('--profileendtoend', action='store_true', help='run end to end profiling')
parser.add_argument('--profiletimefuncs', action='store_true', help='time funcs comparative summary time')
parser.add_argument('--nvproftf', action='store_true', help='run tf code with warmup')
parser.add_argument('--nvprofdace', action='store_true', help='run dace code with warmup')
parser.add_argument('--nvprofoptimizeddace', action='store_true', help='run optimized dace code with warmup')
parser.add_argument('--nvtimefuncstf', action='store_true', help='run tf code with markers')
parser.add_argument('--nvtimefuncsdace', action='store_true', help='run dace code with markers')

args = parser.parse_args()
currlayer = args.cl
verify = args.verify
profileendtoend = args.profileendtoend
profiletimefuncs = args.profiletimefuncs
nvproftf = args.nvproftf
nvprofdace = args.nvprofdace
nvprofoptimizeddace = args.nvprofoptimizeddace
nvtimefuncstf = args.nvtimefuncstf
nvtimefuncsdace = args.nvtimefuncsdace



for layern in range(currlayer, currlayer+1):
    d_input, d_kernel, d_output, inchannels, indepth, inheight, inwidth, outchannels, batchsize = prepareinputs(convparams.iloc[layern])

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

   # Function call for original dace conv3D
    def run_dace():
        dace_conv3d(Input=d_input, kernel=d_kernel, Output=d_output,d_inchannels=inchannels, d_indepth=indepth, d_inheight=inheight,d_inwidth=inwidth, d_outchannels=outchannels, d_batchsize=batchsize)

    # Function call for tensorflow 3D conv
    def run_tf():
        op=tf.nn.conv3d(t_input, t_kernel, strides=[1, 1, 1, 1, 1], padding="VALID")
     
     # Function calls to run the optimized dace function
    def run_optimized_dace():
        sdfg_fun(Input=d_input, kernel=d_kernel, Output=d_output,d_inchannels=inchannels, d_indepth=indepth, d_inheight=inheight,d_inwidth=inwidth, d_outchannels=outchannels, d_batchsize=batchsize)

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

    # Code for verification
    if verify:
        print("INFO: Running verification to compare against tensorflow output")
        refop = tf.nn.conv3d(t_input, t_kernel, strides=[1, 1, 1, 1, 1], padding="VALID")
        sdfg_fun(Input=d_input, kernel=d_kernel, Output=d_output,d_inchannels=inchannels, d_indepth=indepth, d_inheight=inheight,d_inwidth=inwidth, d_outchannels=outchannels, d_batchsize=batchsize)
        tmp_output = d_output.cpu()
        opdace = tf.convert_to_tensor(tmp_output.detach().numpy())
        diff = np.linalg.norm(opdace - refop) / (batchsize * outchannels * indepth * inheight * inwidth )
        print('Difference between tensorflow and dace values:', diff)
        if(diff<=1e-4):
            print(f"Verification successfull")
        else:
            print(f"!!! ERROR: Incorrect verification")

    # Profiling tensorflow using nvprof
    if nvproftf:
        for i in range(0,warmupiter):
            run_tf()
        for i in range(0, totaliter):
            run_tf()
    
    # Profiling baseline dace using nvprof
    if nvprofdace:
        for i in range(0, warmupiter):
            run_dace()
        for i in range(0, totaliter):
            run_dace()

    # Profiling optimized dace using nvprof
    if nvprofoptimizeddace:
        for i in range(0, warmupiter):
            run_optimized_dace()
        for i in range(0, totaliter):
            run_optimized_dace()

    # Comparitive profiling using time funcs
    if profiletimefuncs:
        print(f"INFO: d_input is {d_input.is_cuda}")
        times = time_funcs([run_optimized_dace, run_tf],
                        func_names=["optimizeddace", "tf"],
                        warmups=warmupiter,
                        num_iters=totaliter)
        print(f"INFO: Statistics for layer number {layern}")
        print_time_statistics(times, [ "optimizeddace", "tf"])

    # nvprof using timefuncs for tf
    if nvtimefuncstf:
        print(f"INFO: Warmup for {warmupiter} iterations and total iterations {totaliter}")
        times = time_funcs([run_tf],
                        func_names=["tf"],
                        warmups=warmupiter,
                        num_iters=totaliter)
        print(f"INFO: Statistics for layer number {layern}")
        print_time_statistics(times, [ "tf"])

    # nvprof using timefuncs for dace
    if nvtimefuncsdace:
        print(f"INFO: d_input {d_input.is_cuda}, d_kernel {d_kernel.is_cuda}, d_output {d_output.is_cuda}")
        print(f"INFO: Warmup for {warmupiter} iterations and total iterations {totaliter}")
        times = time_funcs([run_optimized_dace],
                        func_names=["optimizeddace"],
                        warmups=warmupiter,
                        num_iters=totaliter)
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
