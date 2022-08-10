
#include <cuda_runtime.h>
#include <dace/dace.h>


struct dace_simple_t {
    dace::cuda::Context *gpu_context;
};



DACE_EXPORTED int __dace_init_cuda(dace_simple_t *__state, int d_inchannels, int d_indepth, int d_inheight, int d_inwidth, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth);
DACE_EXPORTED void __dace_exit_cuda(dace_simple_t *__state);



int __dace_init_cuda(dace_simple_t *__state, int d_inchannels, int d_indepth, int d_inheight, int d_inwidth, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth) {
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

    __state->gpu_context = new dace::cuda::Context(2, 1);

    // Create cuda streams and events
    for(int i = 0; i < 2; ++i) {
        cudaStreamCreateWithFlags(&__state->gpu_context->streams[i], cudaStreamNonBlocking);
    }
    for(int i = 0; i < 1; ++i) {
        cudaEventCreateWithFlags(&__state->gpu_context->events[i], cudaEventDisableTiming);
    }

    

    return 0;
}

void __dace_exit_cuda(dace_simple_t *__state) {
    

    // Destroy cuda streams and events
    for(int i = 0; i < 2; ++i) {
        cudaStreamDestroy(__state->gpu_context->streams[i]);
    }
    for(int i = 0; i < 1; ++i) {
        cudaEventDestroy(__state->gpu_context->events[i]);
    }

    delete __state->gpu_context;
}

__global__ void dace_simple_58_d_0_0_21(const float * __restrict__ gpu_Input, float * __restrict__ gpu_Output, const float * __restrict__ gpu_kernel, int d_inchannels, int d_indepth, int d_inheight, int d_inwidth, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth, cub::GridBarrier __gbar) {
    {
        {
            {
                int w = (blockIdx.x * 32 + threadIdx.x);
                int h = (blockIdx.y * 1 + threadIdx.y);
                int d = (blockIdx.z * 1 + threadIdx.z);
                if (w < d_outwidth) {
                    {
                        {
                            {
                                for (int oc = (blockIdx.x * 32 + threadIdx.x); oc < d_outchannels; oc += d_outwidth) {
                                    float r_tmp[1]  DACE_ALIGN(64);
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
                                    {
                                        for (auto kd = 0; kd < 3; kd += 1) {
                                            for (auto kh = 0; kh < 3; kh += 1) {
                                                for (auto kw = 0; kw < 3; kw += 1) {
                                                    for (auto ic = 0; ic < d_inchannels; ic += 1) {
                                                        float __tmp4;
                                                        float __tmp5;
                                                        {
                                                            float __in2 = gpu_kernel[(((((((9 * d_inchannels) * d_outchannels) * kd) + (((3 * d_inchannels) * d_outchannels) * kh)) + ((d_inchannels * d_outchannels) * kw)) + (d_outchannels * ic)) + oc)];
                                                            float __in1 = gpu_Input[((((((d_inchannels * d_inheight) * d_inwidth) * (d + kd)) + ((d_inchannels * d_inwidth) * (h + kh))) + (d_inchannels * (kw + w))) + ic)];
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
                                                            // Tasklet code (assign_62_12)
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
                                        // Tasklet code (assign_63_8)
                                        __out = __inp;
                                        ///////////////////

                                        gpu_Output[((((((d * d_outchannels) * d_outheight) * d_outwidth) + ((d_outchannels * d_outwidth) * h)) + (d_outchannels * w)) + oc)] = __out;
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


DACE_EXPORTED void __dace_runkernel_dace_simple_58_d_0_0_21(dace_simple_t *__state, const float * __restrict__ gpu_Input, float * __restrict__ gpu_Output, const float * __restrict__ gpu_kernel, int d_inchannels, int d_indepth, int d_inheight, int d_inwidth, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth);
void __dace_runkernel_dace_simple_58_d_0_0_21(dace_simple_t *__state, const float * __restrict__ gpu_Input, float * __restrict__ gpu_Output, const float * __restrict__ gpu_kernel, int d_inchannels, int d_indepth, int d_inheight, int d_inwidth, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth)
{
    cub::GridBarrierLifetime __gbar_dace_simple_58_d_0_0_21;
    __gbar_dace_simple_58_d_0_0_21.Setup(int_ceil(int_ceil(d_outwidth, 1), 32) * int_ceil(int_ceil(d_outheight, 1), 1) * int_ceil(int_ceil(d_outdepth, 1), 1));

    void  *dace_simple_58_d_0_0_21_args[] = { (void *)&gpu_Input, (void *)&gpu_Output, (void *)&gpu_kernel, (void *)&d_inchannels, (void *)&d_indepth, (void *)&d_inheight, (void *)&d_inwidth, (void *)&d_outchannels, (void *)&d_outdepth, (void *)&d_outheight, (void *)&d_outwidth, (void *)((cub::GridBarrier *)&__gbar_dace_simple_58_d_0_0_21) };
    cudaLaunchKernel((void*)dace_simple_58_d_0_0_21, dim3(int_ceil(int_ceil(d_outwidth, 1), 32), int_ceil(int_ceil(d_outheight, 1), 1), int_ceil(int_ceil(d_outdepth, 1), 1)), dim3(32, 1, 1), dace_simple_58_d_0_0_21_args, 0, __state->gpu_context->streams[0]);
}

