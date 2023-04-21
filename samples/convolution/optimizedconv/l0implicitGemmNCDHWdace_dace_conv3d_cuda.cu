
#include <cuda_runtime.h>
#include <dace/dace.h>


struct implicitGemmNCDHWdace_dace_conv3d_t {
    dace::cuda::Context *gpu_context;
};

const float rkdim = (1.0f/3);
const float rkdim2 = (1.0f/9);
const float rkdim3 = (1.0f/27);


DACE_EXPORTED int __dace_init_cuda(implicitGemmNCDHWdace_dace_conv3d_t *__state, int d_batchsize, int d_inchannels, int d_kdim, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth);
DACE_EXPORTED void __dace_exit_cuda(implicitGemmNCDHWdace_dace_conv3d_t *__state);

DACE_DFI void implicitGemmNCDHWdace_dace_conv3d_116_4_123_12_124_16_125_20_126_24_0_0_27(const float* __restrict__ __tmp_149_66_r, const float* __restrict__ __tmp_150_67_r, float&  __tmp_149_28_w, float&  __tmp_150_28_w, int cta_k, int cta_m, int cta_n, int d_DHW, int d_HW, int d_batchsize, int d_inchannels, int d_kdim, int d_kdim2, int d_kdim3, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth, int gemm_m, int gemm_n, int warp_k, int warp_m, int warp_n) {
    float __tmp38[1]  DACE_ALIGN(64);
    int n;
    int nopq_residual;
    int o;
    int opq_residual;
    int p;
    int q;
    int c;
    int ctrs_residual;
    int t;
    int trs_residual;
    int r;
    int s;
    int __tmp19;
    int __tmp23;
    int __tmp29;
    int __tmp33;
    int d;
    int h;
    int w;
    int __sym___tmp15;
    int __sym___tmp25;
    int __sym___tmp28;
    int __sym___tmp32;
    int __sym___tmp34;

    {
        int __tmp17;

        {
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32((((cta_m + gemm_m) + warp_m) / d_DHW));
            ///////////////////

            n = __out;
        }
        {
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32((((cta_m + gemm_m) + warp_m) % d_DHW));
            ///////////////////

            nopq_residual = __out;
        }
        {
            int __in1 = nopq_residual;
            int __out;

            ///////////////////
            // Tasklet code (_Div_)
            __out =int((__in1) / (d_HW));
            ///////////////////

            __tmp17 = __out;
        }
        {
            int __inp = __tmp17;
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32(__inp);
            ///////////////////

            o = __out;
        }

    }
    __tmp19 = (nopq_residual % d_HW);
    {
        int __tmp21;

        {
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32(__tmp19);
            ///////////////////

            opq_residual = __out;
        }
        {
            int __in1 = opq_residual;
            int __out;

            ///////////////////
            // Tasklet code (_Div_)
            __out =int((__in1) / (d_outwidth));
            ///////////////////

            __tmp21 = __out;
        }
        {
            int __inp = __tmp21;
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32(__inp);
            ///////////////////

            p = __out;
        }

    }
    __tmp23 = (opq_residual % d_outwidth);
    {

        {
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32(__tmp23);
            ///////////////////

            q = __out;
        }
        {
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32(((cta_k + warp_k) * rkdim3));
            ///////////////////

            c = __out;
        }
        {
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32(((cta_k + warp_k) % d_kdim3));
            ///////////////////

            ctrs_residual = __out;
        }

    }
    __tmp29 = (ctrs_residual % d_kdim2);
    {
        int __tmp27;
        int __tmp31;

        {
            int __in1 = ctrs_residual;
            int __out;

            ///////////////////
            // Tasklet code (_Div_)
            __out =int((__in1) * (rkdim2));
            ///////////////////

            __tmp27 = __out;
        }
        {
            int __inp = __tmp27;
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32(__inp);
            ///////////////////

            t = __out;
        }
        {
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32(__tmp29);
            ///////////////////

            trs_residual = __out;
        }
        {
            int __in1 = trs_residual;
            int __out;

            ///////////////////
            // Tasklet code (_Div_)
            __out =int((__in1) / (rkdim));
            ///////////////////

            __tmp31 = __out;
        }
        {
            int __inp = __tmp31;
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32(__inp);
            ///////////////////

            r = __out;
        }

    }
    __tmp33 = (trs_residual % d_kdim);
    d = (o + t);
    h = (p + r);
    {

        {
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32(__tmp33);
            ///////////////////

            s = __out;
        }

    }
    w = (q + s);
    __sym___tmp15 = n;
    __sym___tmp25 = c;
    {


        dace::CopyND<float, 1, false, 1>::template ConstDst<1>::Copy(
        __tmp_149_66_r + ((((((((__sym___tmp15 * d_inchannels) * ((d_kdim + d_outdepth) - 1)) * ((d_kdim + d_outheight) - 1)) * ((d_kdim + d_outwidth) - 1)) + (((__sym___tmp25 * ((d_kdim + d_outdepth) - 1)) * ((d_kdim + d_outheight) - 1)) * ((d_kdim + d_outwidth) - 1))) + ((d * ((d_kdim + d_outheight) - 1)) * ((d_kdim + d_outwidth) - 1))) + (h * ((d_kdim + d_outwidth) - 1))) + w), __tmp38, 1);

    }
    __sym___tmp25 = c;
    __sym___tmp28 = t;
    __sym___tmp32 = r;
    __sym___tmp34 = s;
    {

        {
            float __inp = __tmp38[0];
            float __out;

            ///////////////////
            // Tasklet code (assign_149_28)
            __out = __inp;
            ///////////////////

            __tmp_149_28_w = __out;
        }
        {
            float __inp = __tmp_150_67_r[(((((((__sym___tmp25 * d_kdim) * d_kdim) * d_kdim) + ((__sym___tmp28 * d_kdim) * d_kdim)) + (__sym___tmp32 * d_kdim)) + __sym___tmp34) + ((((d_inchannels * d_kdim) * d_kdim) * d_kdim) * ((cta_n + gemm_n) + warp_n)))];
            float __out;

            ///////////////////
            // Tasklet code (assign_150_28)
            __out = __inp;
            ///////////////////

            __tmp_150_28_w = __out;
        }

    }
    
}

DACE_DFI void implicitGemmNCDHWdace_dace_conv3d_116_4_123_12_152_16_0_0_22(const float* __restrict__ __tmp_153_69_r, float * __restrict__ __tmp_157_53_r_in_from_1_0_in_from_3_0, float * __restrict__ __tmp_158_54_r_in_from_1_0_in_from_3_0, float *  __tmp_161_77_r_in_from_2_0_in_from_3_0, float *  __tmp_161_32_w_out_of_2_1_out_of_3_1, int d_batchsize, int d_inchannels, int d_kdim, int d_outdepth, int d_outheight, int d_outwidth) {

    {

        {
            for (auto warp_k = 0; warp_k < 1; warp_k += 1) {
                float __tmp_157_32_w_out_of_1_1[2]  DACE_ALIGN(64);
                float __tmp_158_32_w_out_of_1_1[16]  DACE_ALIGN(64);
                {
                    for (auto gemm_n = 0; gemm_n < 16; gemm_n += 1) {
                        for (auto gemm_m = 0; gemm_m < 2; gemm_m += 1) {
                            {
                                float __inp = __tmp_157_53_r_in_from_1_0_in_from_3_0[(gemm_m + (64 * warp_k))];
                                float __out;

                                ///////////////////
                                // Tasklet code (assign_157_32)
                                __out = __inp;
                                ///////////////////

                                __tmp_157_32_w_out_of_1_1[gemm_m] = __out;
                            }
                            {
                                float __inp = __tmp_158_54_r_in_from_1_0_in_from_3_0[(gemm_n + (16 * warp_k))];
                                float __out;

                                ///////////////////
                                // Tasklet code (assign_158_32)
                                __out = __inp;
                                ///////////////////

                                __tmp_158_32_w_out_of_1_1[gemm_n] = __out;
                            }
                        }
                    }
                }
                {
                    for (auto gemm_n = 0; gemm_n < 16; gemm_n += 1) {
                        for (auto gemm_m = 0; gemm_m < 2; gemm_m += 1) {
                            float __tmp17;
                            float __tmp18;
                            {
                                float __in1 = __tmp_157_32_w_out_of_1_1[gemm_m];
                                float __in2 = __tmp_158_32_w_out_of_1_1[gemm_n];
                                float __out;

                                ///////////////////
                                // Tasklet code (_Mult_)
                                __out = (__in1 * __in2);
                                ///////////////////

                                __tmp17 = __out;
                            }
                            {
                                float __in2 = __tmp17;
                                float __in1 = __tmp_161_77_r_in_from_2_0_in_from_3_0[(gemm_m + (64 * gemm_n))];
                                float __out;

                                ///////////////////
                                // Tasklet code (_Add_)
                                __out = (__in1 + __in2);
                                ///////////////////

                                __tmp18 = __out;
                            }
                            {
                                float __inp = __tmp18;
                                float __out;

                                ///////////////////
                                // Tasklet code (assign_161_32)
                                __out = __inp;
                                ///////////////////

                                __tmp_161_32_w_out_of_2_1_out_of_3_1[(gemm_m + (64 * gemm_n))] = __out;
                            }
                        }
                    }
                }
            }
        }

    }
    
}

DACE_DFI void implicitGemmNCDHWdace_dace_conv3d_116_4_163_12_164_16_0_0_15(const float&  __tmp_175_64_r, float* __restrict__ __tmp_175_20_w, int cta_m, int cta_n, int d_DHW, int d_HW, int d_batchsize, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth, int gemm_m, int gemm_n, int warp_m, int warp_n) {
    int n;
    int nopq_residual;
    int o;
    int opq_residual;
    int p;
    int q;
    int __tmp17;
    int __tmp21;
    int __sym___tmp13;
    int __sym___tmp16;
    int __sym___tmp20;
    int __sym___tmp22;

    {
        int __tmp15;

        {
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32((((cta_m + gemm_m) + warp_m) / d_DHW));
            ///////////////////

            n = __out;
        }
        {
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32((((cta_m + gemm_m) + warp_m) % d_DHW));
            ///////////////////

            nopq_residual = __out;
        }
        {
            int __in1 = nopq_residual;
            int __out;

            ///////////////////
            // Tasklet code (_Div_)
            __out =int((__in1) / (d_HW));
            ///////////////////

            __tmp15 = __out;
        }
        {
            int __inp = __tmp15;
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32(__inp);
            ///////////////////

            o = __out;
        }

    }
    __tmp17 = (nopq_residual % d_HW);
    {
        int __tmp19;

        {
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32(__tmp17);
            ///////////////////

            opq_residual = __out;
        }
        {
            int __in1 = opq_residual;
            int __out;

            ///////////////////
            // Tasklet code (_Div_)
            __out =int((__in1) / (d_outwidth));
            ///////////////////

            __tmp19 = __out;
        }
        {
            int __inp = __tmp19;
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32(__inp);
            ///////////////////

            p = __out;
        }

    }
    __tmp21 = (opq_residual % d_outwidth);
    __sym___tmp13 = n;
    __sym___tmp16 = o;
    {

        {
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32(__tmp21);
            ///////////////////

            q = __out;
        }

    }
    __sym___tmp20 = p;
    __sym___tmp22 = q;
    {

        {
            float __inp = __tmp_175_64_r;
            float __out;

            ///////////////////
            // Tasklet code (assign_175_20)
            __out = __inp;
            ///////////////////

            __tmp_175_20_w[((((((((__sym___tmp13 * d_outchannels) * d_outdepth) * d_outheight) * d_outwidth) + ((__sym___tmp16 * d_outheight) * d_outwidth)) + (__sym___tmp20 * d_outwidth)) + __sym___tmp22) + (((d_outdepth * d_outheight) * d_outwidth) * ((cta_n + gemm_n) + warp_n)))] = __out;
        }

    }
    
}



int __dace_init_cuda(implicitGemmNCDHWdace_dace_conv3d_t *__state, int d_batchsize, int d_inchannels, int d_kdim, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth) {
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

void __dace_exit_cuda(implicitGemmNCDHWdace_dace_conv3d_t *__state) {
    

    // Destroy cuda streams and events
    for(int i = 0; i < 2; ++i) {
        cudaStreamDestroy(__state->gpu_context->streams[i]);
    }
    for(int i = 0; i < 2; ++i) {
        cudaEventDestroy(__state->gpu_context->events[i]);
    }

    delete __state->gpu_context;
}

__global__ void implicitGemmNCDHWdace_dace_conv3d_116_0_0_0(const float * __restrict__ Input, float * __restrict__ Output, const float * __restrict__ kernel, int d_batchsize, int d_inchannels, int d_kdim, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth) {
    {
        {
            int cta_m = (64 * blockIdx.x);
            int cta_n = (16 * blockIdx.y);
            __shared__ float cta_reducedk[1024];
            {
                {
                    {
                        int warp_m = (2 * threadIdx.x);
                        int warp_n = (16 * threadIdx.y);
                        {
                            {
                                {
                                    for (auto gemm_n = 0; gemm_n < 16; gemm_n += 1) {
                                        for (auto gemm_m = 0; gemm_m < 2; gemm_m += 1) {
                                            {
                                                float __out;

                                                ///////////////////
                                                // Tasklet code (assign_120_24)
                                                __out = 0;
                                                ///////////////////

                                                cta_reducedk[(((gemm_m + (64 * gemm_n)) + warp_m) + (64 * warp_n))] = __out;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            //__syncthreads();
            {
                for (auto cta_k = 0; cta_k < (((d_inchannels * d_kdim) * d_kdim) * d_kdim); cta_k += 1) {
                    __shared__ float __tmp_149_28_w_out_of_1_1_out_of_1_1_out_of_1_1[64];
                    __shared__ float __tmp_150_28_w_out_of_1_1_out_of_1_1_out_of_1_1[16];
                    {
                        {
                            {
                                int warp_m = (2 * threadIdx.x);
                                int warp_n = (16 * threadIdx.y);
                                {
                                    {
                                        {
                                            for (auto warp_k = 0; warp_k < 1; warp_k += 1) {
                                                {
                                                    for (auto gemm_n = 0; gemm_n < 16; gemm_n += 1) {
                                                        for (auto gemm_m = 0; gemm_m < 2; gemm_m += 1) {
                                                            implicitGemmNCDHWdace_dace_conv3d_116_4_123_12_124_16_125_20_126_24_0_0_27(&Input[0], &kernel[0], __tmp_149_28_w_out_of_1_1_out_of_1_1_out_of_1_1[((gemm_m + (64 * warp_k)) + warp_m)], __tmp_150_28_w_out_of_1_1_out_of_1_1_out_of_1_1[((gemm_n + (16 * warp_k)) + warp_n)], cta_k, cta_m, cta_n, ((d_outdepth * d_outheight) * d_outwidth), (d_outheight * d_outwidth), d_batchsize, d_inchannels, d_kdim, (d_kdim * d_kdim), ((d_kdim * d_kdim) * d_kdim), d_outchannels, d_outdepth, d_outheight, d_outwidth, gemm_m, gemm_n, warp_k, warp_m, warp_n);
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
              //      __syncthreads();
                    {
                        {
                            {
                                int warp_m = (2 * threadIdx.x);
                                int warp_n = (16 * threadIdx.y);
                                {
                                    {
                                        implicitGemmNCDHWdace_dace_conv3d_116_4_123_12_152_16_0_0_22(&Input[0], &__tmp_149_28_w_out_of_1_1_out_of_1_1_out_of_1_1[warp_m], &__tmp_150_28_w_out_of_1_1_out_of_1_1_out_of_1_1[warp_n], &cta_reducedk[(warp_m + (64 * warp_n))], &cta_reducedk[(warp_m + (64 * warp_n))], d_batchsize, d_inchannels, d_kdim, d_outdepth, d_outheight, d_outwidth);
                                    }
                                }
                            }
                        }
                    }
                    __syncthreads();
                }
            }
            {
                {
                    {
                        int warp_m = (2 * threadIdx.x);
                        int warp_n = (16 * threadIdx.y);
                        {
                            {
                                {
                                    for (auto gemm_n = 0; gemm_n < 16; gemm_n += 1) {
                                        for (auto gemm_m = 0; gemm_m < 2; gemm_m += 1) {
                                            implicitGemmNCDHWdace_dace_conv3d_116_4_163_12_164_16_0_0_15(cta_reducedk[(((gemm_m + (64 * gemm_n)) + warp_m) + (64 * warp_n))], &Output[0], cta_m, cta_n, ((d_outdepth * d_outheight) * d_outwidth), (d_outheight * d_outwidth), d_batchsize, d_outchannels, d_outdepth, d_outheight, d_outwidth, gemm_m, gemm_n, warp_m, warp_n);
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


DACE_EXPORTED void __dace_runkernel_implicitGemmNCDHWdace_dace_conv3d_116_0_0_0(implicitGemmNCDHWdace_dace_conv3d_t *__state, const float * __restrict__ Input, float * __restrict__ Output, const float * __restrict__ kernel, int d_batchsize, int d_inchannels, int d_kdim, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth);
void __dace_runkernel_implicitGemmNCDHWdace_dace_conv3d_116_0_0_0(implicitGemmNCDHWdace_dace_conv3d_t *__state, const float * __restrict__ Input, float * __restrict__ Output, const float * __restrict__ kernel, int d_batchsize, int d_inchannels, int d_kdim, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth)
{

    void  *implicitGemmNCDHWdace_dace_conv3d_116_0_0_0_args[] = { (void *)&Input, (void *)&Output, (void *)&kernel, (void *)&d_batchsize, (void *)&d_inchannels, (void *)&d_kdim, (void *)&d_outchannels, (void *)&d_outdepth, (void *)&d_outheight, (void *)&d_outwidth };
    cudaLaunchKernel((void*)implicitGemmNCDHWdace_dace_conv3d_116_0_0_0, dim3(int_ceil((((d_batchsize * d_outdepth) * d_outheight) * d_outwidth), 64), int_ceil(d_outchannels, 16), 1), dim3(32, 1, 1), implicitGemmNCDHWdace_dace_conv3d_116_0_0_0_args, 0, __state->gpu_context->streams[0]);
    DACE_CUDA_CHECK(cudaGetLastError());
    DACE_CUDA_CHECK(cudaDeviceSynchronize());
}

