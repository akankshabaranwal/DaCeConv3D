# abaranwal: Code to profile 3D convolution based on daceml functions

from conv3D import *
from daceml.testing.profiling import time_funcs, print_time_statistics

# Prepare inputs and kernel
csv = 'cosmoflow'
convparams =  parsecsv(csv)
layern = 0
for layern in range(0,7):
    d_input, d_kernel, d_output, inchannels, indepth, inheight, inwidth, outchannels, batchsize = prepareinputs(convparams.iloc[layern])

    t_input = tf.convert_to_tensor(d_input)
    t_kernel = tf.convert_to_tensor(d_kernel)

    inchannels = np.int32(inchannels)
    indepth = np.int32(indepth)
    inheight = np.int32(inheight)
    inwidth = np.int32(inwidth)
    outchannels = np.int32(outchannels)
    batchsize = np.int32(batchsize)


    # Function calls to dace_fun and tensorflow function
    def run_dace():
        dace_conv3d(Input=d_input, kernel=d_kernel, Output=d_output,d_inchannels=inchannels, d_indepth=indepth, d_inheight=inheight,d_inwidth=inwidth, d_outchannels=outchannels, d_batchsize=batchsize)

    def run_tf():
        op=tf.nn.conv3d(t_input, t_kernel, strides=[1, 1, 1, 1, 1], padding="VALID")

    # Optimize dace function
    sdfg_fun: dace.SDFG = dace_conv3d.to_sdfg(Input=d_input, kernel=d_kernel, Output=d_output,d_inchannels=inchannels, d_indepth=indepth, d_inheight=inheight,d_inwidth=inwidth, d_outchannels=outchannels, d_batchsize=batchsize)
    sdfg_fun.apply_gpu_transformations()
    optimize_for_gpu(sdfg_fun)

    # Function calls to run the optimized dace function
    def run_optimized_dace():
        sdfg_fun(Input=d_input, kernel=d_kernel, Output=d_output,d_inchannels=inchannels, d_indepth=indepth, d_inheight=inheight,d_inwidth=inwidth, d_outchannels=outchannels, d_batchsize=batchsize)


    times = time_funcs([run_dace, run_optimized_dace, run_tf],
                    func_names=["baselinedace", "optimizeddace", "tf"],
                    warmups=20,
                    num_iters=100)
    print(f"INFO: Statistics for layer number {layern}")
    print_time_statistics(times, ["baselinedace", "optimizeddace", "tf"])