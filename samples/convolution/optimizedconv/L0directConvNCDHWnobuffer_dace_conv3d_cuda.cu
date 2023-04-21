
#include <cuda_runtime.h>
#include <dace/dace.h>


struct directConvNCDHWnobuffer_dace_conv3d_t {
    dace::cuda::Context *gpu_context;
};

const float r_DHW = (1.0f/(128*128*128));
const float r_HW = (1.0f/(128*128));
const float r_W = (1.0f/128);


DACE_EXPORTED int __dace_init_cuda(directConvNCDHWnobuffer_dace_conv3d_t *__state, int d_batchsize, int d_inchannels, int d_kdim, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth);
DACE_EXPORTED void __dace_exit_cuda(directConvNCDHWnobuffer_dace_conv3d_t *__state);

DACE_DFI void directConvNCDHWnobuffer_dace_conv3d_52_4_53_8_54_12_59_16_3_1_9(const float&  __tmp_60_26_r, const float* __restrict__ __tmp_60_32_r, const int&  __tmp_60_50_r, const int&  __tmp_60_56_r, const int&  __tmp_60_62_r, const float&  __tmp_60_68_r, float&  __tmp_60_20_w, int cta_n, int d_batchsize, int d_inchannels, int d_kdim, int d_outdepth, int d_outheight, int d_outwidth, int ic, int kd, int kh, int kw) {
    int __tmp13;
    int __tmp14;
    int __tmp15;
    int __sym___tmp13;
    int __sym___tmp14;
    int __sym___tmp15;

    {

        {
            int __in1 = __tmp_60_50_r;
            int __out;

            ///////////////////
            // Tasklet code (_Add_)
            __out = (int(__in1) + kd);
            ///////////////////

            __tmp13 = __out;
        }

    }
    __sym___tmp13 = __tmp13;
    {

        {
            int __in1 = __tmp_60_56_r;
            int __out;

            ///////////////////
            // Tasklet code (_Add_)
            __out = (int(__in1) + kh);
            ///////////////////

            __tmp14 = __out;
        }

    }
    __sym___tmp14 = __tmp14;
    {

        {
            int __in1 = __tmp_60_62_r;
            int __out;

            ///////////////////
            // Tasklet code (_Add_)
            __out = (int(__in1) + kw);
            ///////////////////

            __tmp15 = __out;
        }

    }
    __sym___tmp15 = __tmp15;
    {
        float __tmp17[1]  DACE_ALIGN(64);
        float __tmp18[1]  DACE_ALIGN(64);

        {
            float __in2 = __tmp_60_68_r;
            float __in1 = __tmp_60_32_r[((((((__sym___tmp13 * ((d_kdim + d_outheight) - 1)) * ((d_kdim + d_outwidth) - 1)) + (__sym___tmp14 * ((d_kdim + d_outwidth) - 1))) + __sym___tmp15) + ((((cta_n * d_inchannels) * ((d_kdim + d_outdepth) - 1)) * ((d_kdim + d_outheight) - 1)) * ((d_kdim + d_outwidth) - 1))) + (((ic * ((d_kdim + d_outdepth) - 1)) * ((d_kdim + d_outheight) - 1)) * ((d_kdim + d_outwidth) - 1)))];
            float __out;

            ///////////////////
            // Tasklet code (_Mult_)
            __out = (__in1 * __in2);
            ///////////////////

            __tmp17[0] = __out;
        }
        {
            float __in1 = __tmp_60_26_r;
            float __in2 = __tmp17[0];
            float __out;

            ///////////////////
            // Tasklet code (_Add_)
            __out = (__in1 + __in2);
            ///////////////////

            __tmp18[0] = __out;
        }
        {
            float __inp = __tmp18[0];
            float __out;

            ///////////////////
            // Tasklet code (assign_60_20)
            __out = __inp;
            ///////////////////

            __tmp_60_20_w = __out;
        }

    }
    
}

DACE_DFI void directConvNCDHWnobuffer_dace_conv3d_52_4_53_8_54_12_0_0_9(const float* __restrict__ __tmp_57_46_r, const float* __restrict__ __tmp_60_68_r_in_from_13_0, float* __restrict__ __tmp_61_16_w, int cta_dhw, int cta_n, int cta_oc, int d_HW, int d_batchsize, int d_inchannels, int d_kdim, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth, int dhw) {
    int __tmp7;
    int d;
    int dhw_residual;
    int h;
    int w;
    float tmp[1]  DACE_ALIGN(64);
    int __tmp9;
    int __sym___tmp5;
    int __sym___tmp8;
    int __sym___tmp10;

    {

        {
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32(((cta_dhw + dhw) / d_HW));
            ///////////////////

            d = __out;
        }
        {
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32(((cta_dhw + dhw) % d_HW));
            ///////////////////

            dhw_residual = __out;
        }
        {
            int __in1 = dhw_residual;
            int __out;

            ///////////////////
            // Tasklet code (_Div_)
            __out = int ((__in1) / (d_outheight));
            ///////////////////

            __tmp7 = __out;
        }

    }
    __tmp9 = (dhw_residual % d_outheight);
    __sym___tmp5 = d;
    {

        {
            int __inp = __tmp7;
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32(__inp);
            ///////////////////

            h = __out;
        }
        {
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32(__tmp9);
            ///////////////////

            w = __out;
        }
        {
            float __out;

            ///////////////////
            // Tasklet code (assign_58_16)
            __out = 0;
            ///////////////////

            tmp[0] = __out;
        }
        {
            for (auto ic = 0; ic < d_inchannels; ic += 1) {
                for (auto kd = 0; kd < d_kdim; kd += 1) {
                    for (auto kh = 0; kh < d_kdim; kh += 1) {
                        for (auto kw = 0; kw < d_kdim; kw += 1) {
                            directConvNCDHWnobuffer_dace_conv3d_52_4_53_8_54_12_59_16_3_1_9(tmp[0], &__tmp_57_46_r[0], d, h, w, __tmp_60_68_r_in_from_13_0[((((((d_kdim * d_kdim) * d_kdim) * ic) + ((d_kdim * d_kdim) * kd)) + (d_kdim * kh)) + kw)], tmp[0], cta_n, d_batchsize, d_inchannels, d_kdim, d_outdepth, d_outheight, d_outwidth, ic, kd, kh, kw);
                        }
                    }
                }
            }
        }

    }
    __sym___tmp8 = h;
    __sym___tmp10 = w;
    {

        {
            float __inp = tmp[0];
            float __out;

            ///////////////////
            // Tasklet code (assign_61_16)
            __out = __inp;
            ///////////////////

            __tmp_61_16_w[((((__sym___tmp10 + ((__sym___tmp5 * d_outheight) * d_outwidth)) + (__sym___tmp8 * d_outwidth)) + ((((cta_n * d_outchannels) * d_outdepth) * d_outheight) * d_outwidth)) + (((cta_oc * d_outdepth) * d_outheight) * d_outwidth))] = __out;
        }

    }
    
}



int __dace_init_cuda(directConvNCDHWnobuffer_dace_conv3d_t *__state, int d_batchsize, int d_inchannels, int d_kdim, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth) {
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

    

    __state->gpu_context = new dace::cuda::Context(3, 4);

    // Create cuda streams and events
    for(int i = 0; i < 3; ++i) {
        cudaStreamCreateWithFlags(&__state->gpu_context->streams[i], cudaStreamNonBlocking);
    }
    for(int i = 0; i < 4; ++i) {
        cudaEventCreateWithFlags(&__state->gpu_context->events[i], cudaEventDisableTiming);
    }

    

    return 0;
}

void __dace_exit_cuda(directConvNCDHWnobuffer_dace_conv3d_t *__state) {
    

    // Destroy cuda streams and events
    for(int i = 0; i < 3; ++i) {
        cudaStreamDestroy(__state->gpu_context->streams[i]);
    }
    for(int i = 0; i < 4; ++i) {
        cudaEventDestroy(__state->gpu_context->events[i]);
    }

    delete __state->gpu_context;
}

__global__ void directConvNCDHWnobuffer_dace_conv3d_52_0_0_0(const float * __restrict__ Input, float * __restrict__ Output, const float * __restrict__ kernel, int d_batchsize, int d_inchannels, int d_kdim, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth) {
    {
        {
            {
                int cta_oc = blockIdx.x;
                int cta_dhw = (64 * blockIdx.y);
                int cta_n = blockIdx.z;
                {
                    {
                        int dhw = (2 * threadIdx.x);
                        {
                            {
                                for (auto dhw = 0; dhw < 2; dhw += 1) {
                                    directConvNCDHWnobuffer_dace_conv3d_52_4_53_8_54_12_0_0_9(&Input[0], &kernel[((((cta_oc * d_inchannels) * d_kdim) * d_kdim) * d_kdim)], &Output[0], cta_dhw, cta_n, cta_oc, (d_outheight * d_outwidth), d_batchsize, d_inchannels, d_kdim, d_outchannels, d_outdepth, d_outheight, d_outwidth, dhw);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_directConvNCDHWnobuffer_dace_conv3d_52_0_0_0(directConvNCDHWnobuffer_dace_conv3d_t *__state, const float * __restrict__ Input, float * __restrict__ Output, const float * __restrict__ kernel, int d_batchsize, int d_inchannels, int d_kdim, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth);
void __dace_runkernel_directConvNCDHWnobuffer_dace_conv3d_52_0_0_0(directConvNCDHWnobuffer_dace_conv3d_t *__state, const float * __restrict__ Input, float * __restrict__ Output, const float * __restrict__ kernel, int d_batchsize, int d_inchannels, int d_kdim, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth)
{

    void  *directConvNCDHWnobuffer_dace_conv3d_52_0_0_0_args[] = { (void *)&Input, (void *)&Output, (void *)&kernel, (void *)&d_batchsize, (void *)&d_inchannels, (void *)&d_kdim, (void *)&d_outchannels, (void *)&d_outdepth, (void *)&d_outheight, (void *)&d_outwidth };
    cudaLaunchKernel((void*)directConvNCDHWnobuffer_dace_conv3d_52_0_0_0, dim3(int_ceil(d_outchannels, 1), int_ceil(((d_outdepth * d_outheight) * d_outwidth), 64), int_ceil(d_batchsize, 1)), dim3(32, 1, 1), directConvNCDHWnobuffer_dace_conv3d_52_0_0_0_args, 0, __state->gpu_context->streams[0]);
}

