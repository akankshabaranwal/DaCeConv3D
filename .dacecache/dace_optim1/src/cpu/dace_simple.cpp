/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct dace_simple_t {
    dace::cuda::Context *gpu_context;
};

DACE_EXPORTED void __dace_runkernel_assign_75_4_map_0_1_4(dace_simple_t *__state, float * __restrict__ gpu_Output);
DACE_EXPORTED void __dace_runkernel_dace_simple_76_d_0_0_19(dace_simple_t *__state, const float * __restrict__ gpu_Input, float * __restrict__ gpu_Output, const float * __restrict__ gpu_kernel);
void __program_dace_simple_internal(dace_simple_t *__state, float * __restrict__ Input, float * __restrict__ Output, float * __restrict__ kernel)
{
    float * gpu_Input;
    cudaMalloc((void**)&gpu_Input, 131072 * sizeof(float));
    float * gpu_Output;
    cudaMalloc((void**)&gpu_Output, 262144 * sizeof(float));
    float * gpu_kernel;
    cudaMalloc((void**)&gpu_kernel, 864 * sizeof(float));

    {

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                cudaMemcpyAsync(gpu_Input, Input, 131072 * sizeof(float), cudaMemcpyHostToDevice, __state->gpu_context->streams[1]);
            } // End omp section
            #pragma omp section
            {
                cudaMemcpyAsync(gpu_kernel, kernel, 864 * sizeof(float), cudaMemcpyHostToDevice, __state->gpu_context->streams[2]);
            } // End omp section
            #pragma omp section
            {
                __dace_runkernel_assign_75_4_map_0_1_4(__state, gpu_Output);
            } // End omp section
        } // End omp sections
        cudaStreamSynchronize(__state->gpu_context->streams[0]);
        cudaStreamSynchronize(__state->gpu_context->streams[1]);
        cudaStreamSynchronize(__state->gpu_context->streams[2]);


    }
    {

        __dace_runkernel_dace_simple_76_d_0_0_19(__state, gpu_Input, gpu_Output, gpu_kernel);
        cudaMemcpyAsync(Output, gpu_Output, 262144 * sizeof(float), cudaMemcpyDeviceToHost, __state->gpu_context->streams[0]);
        cudaStreamSynchronize(__state->gpu_context->streams[0]);


    }
    cudaFree(gpu_Input);
    cudaFree(gpu_Output);
    cudaFree(gpu_kernel);
}

DACE_EXPORTED void __program_dace_simple(dace_simple_t *__state, float * __restrict__ Input, float * __restrict__ Output, float * __restrict__ kernel)
{
    __program_dace_simple_internal(__state, Input, Output, kernel);
}
DACE_EXPORTED int __dace_init_cuda(dace_simple_t *__state);
DACE_EXPORTED int __dace_exit_cuda(dace_simple_t *__state);

DACE_EXPORTED dace_simple_t *__dace_init_dace_simple()
{
    int __result = 0;
    dace_simple_t *__state = new dace_simple_t;


    __result |= __dace_init_cuda(__state);

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

