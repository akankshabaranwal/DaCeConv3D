
#include <cuda_runtime.h>
#include <dace/dace.h>


struct directConvNCDHWmergeddace_dace_conv3d_t {
    dace::cuda::Context *gpu_context;
};



DACE_EXPORTED int __dace_init_cuda(directConvNCDHWmergeddace_dace_conv3d_t *__state, int d_batchsize, int d_inchannels, int d_kdim, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth);
DACE_EXPORTED void __dace_exit_cuda(directConvNCDHWmergeddace_dace_conv3d_t *__state);

DACE_DFI void directConvNCDHWmergeddace_dace_conv3d_37_4_38_8_41_12_1_1_4(const float&  __tmp_42_100_r, float*  __tmp_42_42_r, const int&  __tmp_42_56_r, const int&  __tmp_42_59_r, const int&  __tmp_42_62_r, const float* __restrict__ __tmp_42_68_r, float*  __tmp_42_16_w, int d_batchsize, int d_inchannels, int d_kdim, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth, int ic, int kd, int kh, int kw, int n, int oc) {
    float __tmp13[1]  DACE_ALIGN(64);
    int __tmp14;
    int __tmp15;
    int __tmp16;
    int __sym___tmp_42_56_r;
    int __sym___tmp_42_59_r;
    int __sym___tmp_42_62_r;
    int __sym___tmp14;
    int __sym___tmp15;
    int __sym___tmp16;


    __sym___tmp_42_56_r = __tmp_42_56_r;
    __sym___tmp_42_59_r = __tmp_42_59_r;
    __sym___tmp_42_62_r = __tmp_42_62_r;
    {


        dace::CopyND<float, 1, false, 1>::template ConstDst<1>::Copy(
        __tmp_42_42_r + ((((((__sym___tmp_42_56_r * d_outheight) * d_outwidth) + (__sym___tmp_42_59_r * d_outwidth)) + __sym___tmp_42_62_r) + ((((d_outchannels * d_outdepth) * d_outheight) * d_outwidth) * n)) + (((d_outdepth * d_outheight) * d_outwidth) * oc)), __tmp13, 1);
        {
            int __in1 = __tmp_42_56_r;
            int __out;

            ///////////////////
            // Tasklet code (_Add_)
            __out = (dace::int64(__in1) + kd);
            ///////////////////

            __tmp14 = __out;
        }

    }
    __sym___tmp14 = __tmp14;
    {

        {
            int __in1 = __tmp_42_59_r;
            int __out;

            ///////////////////
            // Tasklet code (_Add_)
            __out = (dace::int64(__in1) + kh);
            ///////////////////

            __tmp15 = __out;
        }

    }
    __sym___tmp15 = __tmp15;
    {

        {
            int __in1 = __tmp_42_62_r;
            int __out;

            ///////////////////
            // Tasklet code (_Add_)
            __out = (dace::int64(__in1) + kw);
            ///////////////////

            __tmp16 = __out;
        }

    }
    __sym___tmp16 = __tmp16;
    __sym___tmp_42_56_r = __tmp_42_56_r;
    __sym___tmp_42_59_r = __tmp_42_59_r;
    __sym___tmp_42_62_r = __tmp_42_62_r;
    {
        float __tmp18[1]  DACE_ALIGN(64);
        float __tmp19[1]  DACE_ALIGN(64);

        {
            float __in1 = __tmp_42_68_r[((((((__sym___tmp14 * ((d_kdim + d_outheight) - 1)) * ((d_kdim + d_outwidth) - 1)) + (__sym___tmp15 * ((d_kdim + d_outwidth) - 1))) + __sym___tmp16) + ((((d_inchannels * n) * ((d_kdim + d_outdepth) - 1)) * ((d_kdim + d_outheight) - 1)) * ((d_kdim + d_outwidth) - 1))) + (((ic * ((d_kdim + d_outdepth) - 1)) * ((d_kdim + d_outheight) - 1)) * ((d_kdim + d_outwidth) - 1)))];
            float __in2 = __tmp_42_100_r;
            float __out;

            ///////////////////
            // Tasklet code (_Mult_)
            __out = (__in1 * __in2);
            ///////////////////

            __tmp18[0] = __out;
        }
        {
            float __in1 = __tmp13[0];
            float __in2 = __tmp18[0];
            float __out;

            ///////////////////
            // Tasklet code (_Add_)
            __out = (__in1 + __in2);
            ///////////////////

            __tmp19[0] = __out;
        }
        {
            float __inp = __tmp19[0];
            float __out;

            ///////////////////
            // Tasklet code (assign_42_16)
            __out = __inp;
            ///////////////////

            __tmp_42_16_w[((((((__sym___tmp_42_56_r * d_outheight) * d_outwidth) + (__sym___tmp_42_59_r * d_outwidth)) + __sym___tmp_42_62_r) + ((((d_outchannels * d_outdepth) * d_outheight) * d_outwidth) * n)) + (((d_outdepth * d_outheight) * d_outwidth) * oc))] = __out;
        }

    }
    
}

DACE_DFI void directConvNCDHWmergeddace_dace_conv3d_37_4_38_8_0_0_8(const float* __restrict__ __tmp_42_100_r_in_from_9_0, float*  __tmp_42_42_r_in_from_9_0, const float* __restrict__ __tmp_42_68_r_in_from_9_0, float*  __tmp_42_16_w_out_of_9_1, int d_HW, int d_batchsize, int d_inchannels, int d_kdim, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth, int dhw, int ic, int n, int oc) {
    int d;
    int dhw_residual;
    int h;
    int __tmp10;

    {
        int __tmp4;
        int __tmp8;

        {
            int __out;

            ///////////////////
            // Tasklet code (_Div_)
            __out =int((dhw) / (d_HW));
            ///////////////////

            __tmp4 = __out;
        }
        {
            int __inp = __tmp4;
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32(__inp);
            ///////////////////

            d = __out;
        }
        {
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32((dhw % d_HW));
            ///////////////////

            dhw_residual = __out;
        }
        {
            int __in1 = dhw_residual;
            int __out;

            ///////////////////
            // Tasklet code (_Div_)
            __out =int((__in1) / (d_outheight));
            ///////////////////

            __tmp8 = __out;
        }
        {
            int __inp = __tmp8;
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32(__inp);
            ///////////////////

            h = __out;
        }

    }
    __tmp10 = (dhw_residual % d_outheight);
    {
        int w;

        {
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32(__tmp10);
            ///////////////////

            w = __out;
        }
        {
            for (auto kd = 0; kd < 3; kd += 1) {
                for (auto kh = 0; kh < 3; kh += 1) {
                    for (auto kw = 0; kw < 3; kw += 1) {
                        directConvNCDHWmergeddace_dace_conv3d_37_4_38_8_41_12_1_1_4(__tmp_42_100_r_in_from_9_0[((((d_kdim * d_kdim) * kd) + (d_kdim * kh)) + kw)], &__tmp_42_42_r_in_from_9_0[0], d, h, w, &__tmp_42_68_r_in_from_9_0[0], &__tmp_42_16_w_out_of_9_1[0], d_batchsize, d_inchannels, d_kdim, d_outchannels, d_outdepth, d_outheight, d_outwidth, ic, kd, kh, kw, n, oc);
                    }
                }
            }
        }

    }
    
}



int __dace_init_cuda(directConvNCDHWmergeddace_dace_conv3d_t *__state, int d_batchsize, int d_inchannels, int d_kdim, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth) {
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

    

    __state->gpu_context = new dace::cuda::Context(2, 2);

    // Create cuda streams and events
    for(int i = 0; i < 2; ++i) {
        cudaStreamCreateWithFlags(&__state->gpu_context->streams[i], cudaStreamNonBlocking);
    }
    for(int i = 0; i < 2; ++i) {
        cudaEventCreateWithFlags(&__state->gpu_context->events[i], cudaEventDisableTiming);
    }

    

    return 0;
}

void __dace_exit_cuda(directConvNCDHWmergeddace_dace_conv3d_t *__state) {
    

    // Destroy cuda streams and events
    for(int i = 0; i < 2; ++i) {
        cudaStreamDestroy(__state->gpu_context->streams[i]);
    }
    for(int i = 0; i < 2; ++i) {
        cudaEventDestroy(__state->gpu_context->events[i]);
    }

    delete __state->gpu_context;
}

__global__ void directConvNCDHWmergeddace_dace_conv3d_37_0_0_11(const float * __restrict__ Input, float * __restrict__ Output, const float * __restrict__ kernel, int d_batchsize, int d_inchannels, int d_kdim, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth) {
    {
        {
            {
                int tile_oc = blockIdx.x;
                int tile_dhw = (64 * blockIdx.y);
                int tile_n = blockIdx.z;
                {
                    for (auto tile_ic = 0; tile_ic < d_inchannels; tile_ic += 1) {
                        {
                            {
                                int tile1_dhw = ((2 * threadIdx.x) + tile_dhw);
                                if (tile1_dhw >= tile_dhw) {
                                    {
                                        for (auto ic = tile_ic; ic < (tile_ic + 1); ic += 1) {
                                            {
                                                for (auto n = tile_n; n < (tile_n + 1); n += 1) {
                                                    for (auto dhw = tile1_dhw; dhw < (tile1_dhw + 2); dhw += 1) {
                                                        for (auto oc = tile_oc; oc < (tile_oc + 1); oc += 1) {
                                                            directConvNCDHWmergeddace_dace_conv3d_37_4_38_8_0_0_8(&kernel[(((((d_inchannels * d_kdim) * d_kdim) * d_kdim) * oc) + (((d_kdim * d_kdim) * d_kdim) * ic))], &Output[0], &Input[0], &Output[0], (d_outheight * d_outwidth), d_batchsize, d_inchannels, d_kdim, d_outchannels, d_outdepth, d_outheight, d_outwidth, dhw, ic, n, oc);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        __syncthreads();
                    }
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_directConvNCDHWmergeddace_dace_conv3d_37_0_0_11(directConvNCDHWmergeddace_dace_conv3d_t *__state, const float * __restrict__ Input, float * __restrict__ Output, const float * __restrict__ kernel, int d_batchsize, int d_inchannels, int d_kdim, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth);
void __dace_runkernel_directConvNCDHWmergeddace_dace_conv3d_37_0_0_11(directConvNCDHWmergeddace_dace_conv3d_t *__state, const float * __restrict__ Input, float * __restrict__ Output, const float * __restrict__ kernel, int d_batchsize, int d_inchannels, int d_kdim, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth)
{

    void  *directConvNCDHWmergeddace_dace_conv3d_37_0_0_11_args[] = { (void *)&Input, (void *)&Output, (void *)&kernel, (void *)&d_batchsize, (void *)&d_inchannels, (void *)&d_kdim, (void *)&d_outchannels, (void *)&d_outdepth, (void *)&d_outheight, (void *)&d_outwidth };
    cudaLaunchKernel((void*)directConvNCDHWmergeddace_dace_conv3d_37_0_0_11, dim3(int_ceil(d_outchannels, 1), int_ceil(((d_outdepth * d_outheight) * d_outwidth), 64), int_ceil(d_batchsize, 1)), dim3(32, 1, 1), directConvNCDHWmergeddace_dace_conv3d_37_0_0_11_args, 0, __state->gpu_context->streams[0]);
}

