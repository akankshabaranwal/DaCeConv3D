
#include <cuda_runtime.h>
#include <dace/dace.h>


struct dace_simple_0_t {
    dace::cuda::Context *gpu_context;
};



DACE_EXPORTED int __dace_init_cuda(dace_simple_0_t *__state);
DACE_EXPORTED void __dace_exit_cuda(dace_simple_0_t *__state);



int __dace_init_cuda(dace_simple_0_t *__state) {
    int count;

    // Check that we are able to run cuda code
    if (cudaGetDeviceCount(&count) != cudaSuccess)
    {
        printf("ERROR: GPU drivers are not configured or cuda-capable device "
               "not found\n");
        return 1;
    }
    if (count == 0)
    {
        printf("ERROR: No cuda-capable devices found\n");
        return 2;
    }

    // Initialize cuda before we run the application
    float *dev_X;
    cudaMalloc((void **) &dev_X, 1);
    cudaFree(dev_X);

    __state->gpu_context = new dace::cuda::Context(3, 1);

    // Create cuda streams and events
    for(int i = 0; i < 3; ++i) {
        cudaStreamCreateWithFlags(&__state->gpu_context->streams[i], cudaStreamNonBlocking);
    }
    for(int i = 0; i < 1; ++i) {
        cudaEventCreateWithFlags(&__state->gpu_context->events[i], cudaEventDisableTiming);
    }

    

    return 0;
}

void __dace_exit_cuda(dace_simple_0_t *__state) {
    

    // Destroy cuda streams and events
    for(int i = 0; i < 3; ++i) {
        cudaStreamDestroy(__state->gpu_context->streams[i]);
    }
    for(int i = 0; i < 1; ++i) {
        cudaEventDestroy(__state->gpu_context->events[i]);
    }

    delete __state->gpu_context;
}

__global__ void assign_75_4_map_0_1_4(float * __restrict__ gpu_Output) {
    {
        {
            {
                {
                    {
                        int __i4 = (blockIdx.x * 32 + threadIdx.x);
                        int __i3 = (blockIdx.y * 1 + threadIdx.y);
                        int __i2 = (blockIdx.z / (32));
                        int __i1 = (blockIdx.z / (1)) % (32);
                        int __i0 = (blockIdx.z / (1)) % (1);
                        if (__i4 < 8) {
                            {
                                {
                                    if (__i1 >= 0 && __i1 < 32) {
                                        if (__i0 >= 0 && __i0 < 1) {
                                            {
                                                float __out;

                                                ///////////////////
                                                // Tasklet code (assign_75_4)
                                                __out = 0;
                                                ///////////////////

                                                gpu_Output[(((((262144 * __i0) + (8192 * __i1)) + (256 * __i2)) + (8 * __i3)) + __i4)] = __out;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_assign_75_4_map_0_1_4(dace_simple_0_t *__state, float * __restrict__ gpu_Output);
void __dace_runkernel_assign_75_4_map_0_1_4(dace_simple_0_t *__state, float * __restrict__ gpu_Output)
{

    void  *assign_75_4_map_0_1_4_args[] = { (void *)&gpu_Output };
    cudaLaunchKernel((void*)assign_75_4_map_0_1_4, dim3(int_ceil(int_ceil(8, 1), 32), int_ceil(int_ceil(32, 1), 1), int_ceil(((int_ceil(1, 1) * int_ceil(32, 1)) * int_ceil(32, 1)), 1)), dim3(32, 1, 1), assign_75_4_map_0_1_4_args, 0, __state->gpu_context->streams[0]);
}
__global__ void dace_simple_76_d_0_0_19(const float * __restrict__ gpu_Input, float * __restrict__ gpu_Output, const float * __restrict__ gpu_kernel) {
    {
        {
            {
                int w = blockIdx.x;
                int h = blockIdx.y;
                int d = blockIdx.z;
                {
                    {
                        float r_tmp[1]  DACE_ALIGN(64);
                        float trans_gpu_Input[108]  DACE_ALIGN(64);
                        int oc = threadIdx.x;
                        {
                            {
                                for (auto __i0 = 0; __i0 < 1; __i0 += 1) {
                                    {
                                        float __out;

                                        ///////////////////
                                        // Tasklet code (_numpy_full_)
                                        __out = 0.0;
                                        ///////////////////

                                        r_tmp[__i0] = __out;
                                    }
                                }
                            }

                            dace::CopyND<float, 1, false, 3, 3, 3, 4>::template ConstDst<36, 12, 4, 1>::Copy(
                            gpu_Input + (((4096 * d) + (128 * h)) + (4 * w)), trans_gpu_Input, 4096, 128, 4, 1);
                            {
                                for (auto kd = 0; kd < 3; kd += 1) {
                                    for (auto kh = 0; kh < 3; kh += 1) {
                                        for (auto kw = 0; kw < 3; kw += 1) {
                                            for (auto ic = 0; ic < 4; ic += 1) {
                                                float __tmp4;
                                                float __tmp5;
                                                {
                                                    float __in2 = gpu_kernel[(((((8 * ic) + (288 * kd)) + (96 * kh)) + (32 * kw)) + oc)];
                                                    float __in1 = trans_gpu_Input[(((ic + (36 * kd)) + (12 * kh)) + (4 * kw))];
                                                    float __out;

                                                    ///////////////////
                                                    // Tasklet code (_Mult_)
                                                    __out = (__in1 * __in2);
                                                    ///////////////////

                                                    __tmp4 = __out;
                                                }
                                                {
                                                    float __in2 = __tmp4;
                                                    float __in1 = r_tmp[0];
                                                    float __out;

                                                    ///////////////////
                                                    // Tasklet code (_Add_)
                                                    __out = (__in1 + __in2);
                                                    ///////////////////

                                                    __tmp5 = __out;
                                                }
                                                {
                                                    float __inp = __tmp5;
                                                    float __out;

                                                    ///////////////////
                                                    // Tasklet code (assign_80_12)
                                                    __out = __inp;
                                                    ///////////////////

                                                    r_tmp[0] = __out;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            {
                                float __inp = r_tmp[0];
                                float __out;

                                ///////////////////
                                // Tasklet code (assign_81_8)
                                __out = __inp;
                                ///////////////////

                                gpu_Output[((((8192 * d) + (256 * h)) + oc) + (8 * w))] = __out;
                            }
                        }
                    }
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_dace_simple_76_d_0_0_19(dace_simple_0_t *__state, const float * __restrict__ gpu_Input, float * __restrict__ gpu_Output, const float * __restrict__ gpu_kernel);
void __dace_runkernel_dace_simple_76_d_0_0_19(dace_simple_0_t *__state, const float * __restrict__ gpu_Input, float * __restrict__ gpu_Output, const float * __restrict__ gpu_kernel)
{

    void  *dace_simple_76_d_0_0_19_args[] = { (void *)&gpu_Input, (void *)&gpu_Output, (void *)&gpu_kernel };
    cudaLaunchKernel((void*)dace_simple_76_d_0_0_19, dim3(int_ceil(30, 1), int_ceil(30, 1), int_ceil(30, 1)), dim3(8, 1, 1), dace_simple_76_d_0_0_19_args, 0, __state->gpu_context->streams[0]);
}

