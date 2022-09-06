# abaranwal: Code to profile 3D convolution based on daceml functions

from conv3D import *
from daceml.testing.profiling import time_funcs, print_time_statistics

# Prepare inputs and kernel
csv = 'cosmoflow'
warmupiter = 10
totaliter = 100
convparams =  parsecsv(csv)
layern = 0
verify = False
profileendtoend = False
profiletimefuncs = True

for layern in range(0,7):
    d_input, d_kernel, d_output, inchannels, indepth, inheight, inwidth, outchannels, batchsize = prepareinputs(convparams.iloc[layern])

    ## Prepare inputs for tensorflow fun
    t_input = tf.convert_to_tensor(d_input)
    t_kernel = tf.convert_to_tensor(d_kernel)

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

    # Code for verification
    if verify:
        print("INFO: Running verification to compare against tensorflow output")
        refop = tf.nn.conv3d(t_input, t_kernel, strides=[1, 1, 1, 1, 1], padding="VALID")
        sdfg_fun(Input=d_input, kernel=d_kernel, Output=d_output,d_inchannels=inchannels, d_indepth=indepth, d_inheight=inheight,d_inwidth=inwidth, d_outchannels=outchannels, d_batchsize=batchsize)
        opdace = tf.convert_to_tensor(d_output)
        diff = np.linalg.norm(opdace - refop) / (batchsize * outchannels * indepth * inheight * inwidth )
        print('Difference between tensorflow and dace values:', diff)
        if(diff<=1e-4):
            print(f"Verification successfull")
        else:
            print(f"!!! ERROR: Incorrect verification")

    # Profiling using time funcs
    if profiletimefuncs:
        print(f"INFO: Warmup for {warmupiter} iterations and total iterations {totaliter}")
        times = time_funcs([run_dace, run_optimized_dace, run_tf],
                        func_names=["baselinedace", "optimizeddace", "tf"],
                        warmups=warmupiter,
                        num_iters=totaliter)
        print(f"INFO: Statistics for layer number {layern}")
        print_time_statistics(times, ["baselinedace", "optimizeddace", "tf"])

    # Code for end to end runtime
    if profileendtoend:
        print(f"INFO: End to end profiling")
        print(f"INFO: Warmup for {warmupiter} iterations and total iterations {totaliter}")        

        ## TF warm up
        timeit.Timer(lambda: timetfgpu_conv3D(t_input, t_kernel)).repeat(repeat=5, number=1)
        tf_runtime = timeit.Timer(lambda: timetfgpu_conv3D(t_input, t_kernel)).repeat(repeat=100, number=2)
        print("INFO: Tensorflow in ms: ", (np.median(tf_runtime))*1000)

        print("*\n*\n*")
        print("INFO: Running optimized sdfg code")
        rundacesdfgprofiling(sdfg_fun, d_input, d_kernel, d_output, inchannels, indepth, inheight, inwidth, outchannels, batchsize, warmupiter)
        dace_optimized = rundacesdfgprofiling(sdfg_fun, d_input, d_kernel, d_output, inchannels, indepth, inheight, inwidth, outchannels, batchsize, totaliter)
        print("INFO: Dace time in ms: ",np.median(dace_optimized)*1000)


# for layern in range(0,1):
    
#     d_input, d_kernel, d_output, inchannels, indepth, inheight, inwidth, outchannels, batchsize = prepareinputs(convparams.iloc[layern])

#     t_input = tf.convert_to_tensor(d_input)
#     t_kernel = tf.convert_to_tensor(d_kernel)

#     inchannels = np.int32(inchannels)
#     indepth = np.int32(indepth)
#     inheight = np.int32(inheight)
#     inwidth = np.int32(inwidth)
#     outchannels = np.int32(outchannels)
#     batchsize = np.int32(batchsize)

#     # Function calls to dace_fun and tensorflow function
#     def run_dace():
#         dace_conv3d(Input=d_input, kernel=d_kernel, Output=d_output,d_inchannels=inchannels, d_indepth=indepth, d_inheight=inheight,d_inwidth=inwidth, d_outchannels=outchannels, d_batchsize=batchsize)

#     def run_tf():
#         op=tf.nn.conv3d(t_input, t_kernel, strides=[1, 1, 1, 1, 1], padding="VALID")

#     # Optimize dace function
#     sdfg_fun: dace.SDFG = dace_conv3d.to_sdfg(d_input, d_kernel, d_output)
#     sdfg_fun.apply_gpu_transformations()
#     optimize_for_gpu(sdfg_fun)

#     # Function calls to run the optimized dace function
#     def run_optimized_dace():
#         sdfg_fun(Input=d_input, kernel=d_kernel, Output=d_output,d_inchannels=inchannels, d_indepth=indepth, d_inheight=inheight,d_inwidth=inwidth, d_outchannels=outchannels, d_batchsize=batchsize)

#     if verify:
#         print("INFO: Running verification to compare against tensorflow output")
#         refop = tf.nn.conv3d(t_input, t_kernel, strides=[1, 1, 1, 1, 1], padding="VALID")
#         sdfg_fun(Input=d_input, kernel=d_kernel, Output=d_output,d_inchannels=inchannels, d_indepth=indepth, d_inheight=inheight,d_inwidth=inwidth, d_outchannels=outchannels, d_batchsize=batchsize)
#         opdace = tf.convert_to_tensor(d_output)
#         diff = np.linalg.norm(opdace - refop) / (batchsize * outchannels * indepth * inheight * inwidth )
#         print('Difference between tensorflow and dace values:', diff)
#         if(diff<=1e-4):
#             print(f"Verification successfull")
#         else:
#             print(f"!!! ERROR: Incorrect verification")

#     if profiletimefuncs:
#         print(f"INFO: Warmup for {warmupiter} iterations and total iterations {totaliter}")
#         times = time_funcs([run_dace, run_optimized_dace, run_tf],
#                         func_names=["baselinedace", "optimizeddace", "tf"],
#                         warmups=warmupiter,
#                         num_iters=totaliter)
#         print(f"INFO: Statistics for layer number {layern}")
#         print_time_statistics(times, ["baselinedace", "optimizeddace", "tf"])


#     if profileendtoend:
#         print(f"INFO: End to end profiling")
#         print(f"INFO: Warmup for {warmupiter} iterations and total iterations {totaliter}")        
#         ## TF warm up
#         timeit.Timer(lambda: timetfgpu_conv3D(t_input, t_kernel)).repeat(repeat=10, number=1)
#         tf_runtime = timeit.Timer(lambda: timetfgpu_conv3D(t_input, t_kernel)).repeat(repeat=100, number=2)
#         print("INFO: Tensorflow in ms: ", (np.median(tf_runtime))*1000)
#         rundacesdfgprofiling(sdfg_fun, d_input, d_kernel, d_kernel, inchannels, indepth, inheight, inwidth, outchannels, batchsize, warmupiter)
#         dace_optimized = rundacesdfgprofiling(sdfg_fun, d_input, d_kernel, d_output, inchannels, indepth, inheight, inwidth, outchannels, batchsize, totaliter)
#         print("INFO: Dace time in ms: ",np.median(dace_optimized)*1000)