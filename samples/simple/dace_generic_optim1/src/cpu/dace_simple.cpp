/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct dace_simple_t {
    dace::cuda::Context *gpu_context;
};

DACE_EXPORTED void __dace_runkernel_dace_simple_58_d_0_0_21(dace_simple_t *__state, const float * __restrict__ gpu_Input, float * __restrict__ gpu_Output, const float * __restrict__ gpu_kernel, int d_inchannels, int d_indepth, int d_inheight, int d_inwidth, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth);
void __program_dace_simple_internal(dace_simple_t *__state, float * __restrict__ Input, float * __restrict__ Output, float * __restrict__ kernel, int d_inchannels, int d_indepth, int d_inheight, int d_inwidth, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth)
{

    {
        float * gpu_Input;
        cudaMalloc((void**)&gpu_Input, (((d_inchannels * d_indepth) * d_inheight) * d_inwidth) * sizeof(float));
        float * gpu_Output;
        cudaMalloc((void**)&gpu_Output, (((d_outchannels * d_outdepth) * d_outheight) * d_outwidth) * sizeof(float));
        float * gpu_kernel;
        cudaMalloc((void**)&gpu_kernel, ((27 * d_inchannels) * d_outchannels) * sizeof(float));

        cudaMemcpyAsync(gpu_Input, Input, (((d_inchannels * d_indepth) * d_inheight) * d_inwidth) * sizeof(float), cudaMemcpyHostToDevice, __state->gpu_context->streams[0]);
        cudaMemcpyAsync(gpu_kernel, kernel, ((27 * d_inchannels) * d_outchannels) * sizeof(float), cudaMemcpyHostToDevice, __state->gpu_context->streams[1]);

        cudaEventRecord(__state->gpu_context->events[0], __state->gpu_context->streams[1]);
        cudaStreamWaitEvent(__state->gpu_context->streams[0], __state->gpu_context->events[0], 0);

        __dace_runkernel_dace_simple_58_d_0_0_21(__state, gpu_Input, gpu_Output, gpu_kernel, d_inchannels, d_indepth, d_inheight, d_inwidth, d_outchannels, d_outdepth, d_outheight, d_outwidth);
        cudaMemcpyAsync(Output, gpu_Output, (((d_outchannels * d_outdepth) * d_outheight) * d_outwidth) * sizeof(float), cudaMemcpyDeviceToHost, __state->gpu_context->streams[0]);
        cudaStreamSynchronize(__state->gpu_context->streams[0]);

        cudaFree(gpu_Input);
        cudaFree(gpu_Output);
        cudaFree(gpu_kernel);

    }
}

DACE_EXPORTED void __program_dace_simple(dace_simple_t *__state, float * __restrict__ Input, float * __restrict__ Output, float * __restrict__ kernel, int d_inchannels, int d_indepth, int d_inheight, int d_inwidth, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth)
{
    __program_dace_simple_internal(__state, Input, Output, kernel, d_inchannels, d_indepth, d_inheight, d_inwidth, d_outchannels, d_outdepth, d_outheight, d_outwidth);
}
DACE_EXPORTED int __dace_init_cuda(dace_simple_t *__state, int d_inchannels, int d_indepth, int d_inheight, int d_inwidth, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth);
DACE_EXPORTED int __dace_exit_cuda(dace_simple_t *__state);

DACE_EXPORTED dace_simple_t *__dace_init_dace_simple(int d_inchannels, int d_indepth, int d_inheight, int d_inwidth, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth)
{
    int __result = 0;
    dace_simple_t *__state = new dace_simple_t;


    __result |= __dace_init_cuda(__state, d_inchannels, d_indepth, d_inheight, d_inwidth, d_outchannels, d_outdepth, d_outheight, d_outwidth);

    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED void __dace_exit_dace_simple(dace_simple_t *__state)
{
    __dace_exit_cuda(__state);
    delete __state;
}

