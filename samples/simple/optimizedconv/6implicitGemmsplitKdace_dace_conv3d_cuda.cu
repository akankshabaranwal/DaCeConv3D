
#include <cuda_runtime.h>
#include <dace/dace.h>


struct implicitGemmsplitKdace_dace_conv3d_t {
    dace::cuda::Context *gpu_context;
};

const float r_kdim = (1.0f/3);
const float r_kdim2 = (1.0f/9);
const float r_kdim3 = (1.0f/27);


DACE_EXPORTED int __dace_init_cuda(implicitGemmsplitKdace_dace_conv3d_t *__state, int d_batchsize, int d_inchannels, int d_kdim, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth);
DACE_EXPORTED void __dace_exit_cuda(implicitGemmsplitKdace_dace_conv3d_t *__state);

DACE_DFI void implicitGemmsplitKdace_dace_conv3d_118_4_125_12_128_16_129_20_130_24_0_0_49(const int&  __tmp_141_46_r, const float* __restrict__ __tmp_154_65_r, const float* __restrict__ __tmp_155_66_r, int&  __tmp_140_28_w, float&  __tmp_154_28_w, float&  __tmp_155_28_w, int cta_k, int cta_m, int cta_n, int d_DHW, int d_GEMM_K, int d_HW, int d_batchsize, int d_inchannels, int d_kdim, int d_kdim2, int d_kdim3, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth, int gemm_m, int gemm_n, int isplit_k, int warp_k, int warp_m, int warp_n) {
    int __tmp28;
    int __tmp33;
    int __tmp38;
    float __tmp49[1]  DACE_ALIGN(64);
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
    int __tmp21;
    int __tmp25;
    int __tmp29;
    int __tmp34;
    int __tmp40;
    int __tmp44;
    int d;
    int h;
    int w;
    int __sym___tmp17;
    int __sym___tmp32;
    int __sym___tmp39;
    int __sym___tmp43;
    int __sym___tmp45;

    {
        int __tmp19;

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

            __tmp19 = __out;
        }
        {
            int __inp = __tmp19;
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32(__inp);
            ///////////////////

            o = __out;
        }

    }
    __tmp21 = (nopq_residual % d_HW);
    {
        int __tmp23;

        {
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32(__tmp21);
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

            __tmp23 = __out;
        }
        {
            int __inp = __tmp23;
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32(__inp);
            ///////////////////

            p = __out;
        }

    }
    __tmp25 = (opq_residual % d_outwidth);
    {
        int __tmp27;

        {
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32(__tmp25);
            ///////////////////

            q = __out;
        }
        {
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32((0.125 * d_GEMM_K));
            ///////////////////

            __tmp27 = __out;
        }
        {
            int __inp = __tmp27;
            int __out;

            ///////////////////
            // Tasklet code (assign_140_28)
            __out = __inp;
            ///////////////////

            __tmp_140_28_w = __out;
        }
        {
            int __in1 = __tmp_141_46_r;
            int __out;

            ///////////////////
            // Tasklet code (_Mult_)
            __out = (int(__in1) * isplit_k);
            ///////////////////

            __tmp28 = __out;
        }

    }
    __tmp29 = (__tmp28 + cta_k);
    {
        int __tmp31;

        {
            int __out;

            ///////////////////
            // Tasklet code (_Div_)
            __out =int(((__tmp29 + warp_k)) * (r_kdim3));
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

            c = __out;
        }
        {
            int __in1 = __tmp_141_46_r;
            int __out;

            ///////////////////
            // Tasklet code (_Mult_)
            __out = (int(__in1) * isplit_k);
            ///////////////////

            __tmp33 = __out;
        }

    }
    __tmp34 = (__tmp33 + cta_k);
    {
        int __tmp36;

        {
            int __out;

            ///////////////////
            // Tasklet code (_Mod_)
            __out = ((__tmp34 + warp_k) % int(d_kdim3));
            ///////////////////

            __tmp36 = __out;
        }
        {
            int __inp = __tmp36;
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32(__inp);
            ///////////////////

            ctrs_residual = __out;
        }
        {
            int __in1 = ctrs_residual;
            int __out;

            ///////////////////
            // Tasklet code (_Div_)
            __out =int((__in1) * (r_kdim2));
            ///////////////////

            __tmp38 = __out;
        }

    }
    __tmp40 = (ctrs_residual % d_kdim2);
    {
        int __tmp42;

        {
            int __inp = __tmp38;
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
            __out = dace::int32(__tmp40);
            ///////////////////

            trs_residual = __out;
        }
        {
            int __in1 = trs_residual;
            int __out;

            ///////////////////
            // Tasklet code (_Div_)
            __out =int((__in1) * (r_kdim));
            ///////////////////

            __tmp42 = __out;
        }
        {
            int __inp = __tmp42;
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32(__inp);
            ///////////////////

            r = __out;
        }

    }
    __tmp44 = (trs_residual % d_kdim);
    d = (o + t);
    h = (p + r);
    {

        {
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32(__tmp44);
            ///////////////////

            s = __out;
        }

    }
    w = (q + s);
    __sym___tmp17 = n;
    __sym___tmp32 = c;
    {


        dace::CopyND<float, 1, false, 1>::template ConstDst<1>::Copy(
        __tmp_154_65_r + ((((((((__sym___tmp17 * d_inchannels) * ((d_kdim + d_outdepth) - 1)) * ((d_kdim + d_outheight) - 1)) * ((d_kdim + d_outwidth) - 1)) + (((__sym___tmp32 * ((d_kdim + d_outdepth) - 1)) * ((d_kdim + d_outheight) - 1)) * ((d_kdim + d_outwidth) - 1))) + ((d * ((d_kdim + d_outheight) - 1)) * ((d_kdim + d_outwidth) - 1))) + (h * ((d_kdim + d_outwidth) - 1))) + w), __tmp49, 1);

    }
    __sym___tmp32 = c;
    __sym___tmp39 = t;
    __sym___tmp43 = r;
    __sym___tmp45 = s;
    {

        {
            float __inp = __tmp49[0];
            float __out;

            ///////////////////
            // Tasklet code (assign_154_28)
            __out = __inp;
            ///////////////////

            __tmp_154_28_w = __out;
        }
        {
            float __inp = __tmp_155_66_r[(((((((__sym___tmp32 * d_kdim) * d_kdim) * d_kdim) + ((__sym___tmp39 * d_kdim) * d_kdim)) + (__sym___tmp43 * d_kdim)) + __sym___tmp45) + ((((d_inchannels * d_kdim) * d_kdim) * d_kdim) * ((cta_n + gemm_n) + warp_n)))];
            float __out;

            ///////////////////
            // Tasklet code (assign_155_28)
            __out = __inp;
            ///////////////////

            __tmp_155_28_w = __out;
        }

    }
    
}

DACE_DFI void implicitGemmsplitKdace_dace_conv3d_118_4_125_12_157_16_161_24_0_0_44(const float* __restrict__ __tmp_162_73_r, float * __restrict__ __tmp_165_53_r_in_from_3_0, float * __restrict__ __tmp_166_54_r_in_from_3_0, float *  __tmp_170_77_r_in_from_4_0, float *  __tmp_170_32_w_out_of_4_1, int d_batchsize, int d_inchannels, int d_kdim, int d_outdepth, int d_outheight, int d_outwidth) {

    {
        float warp_input[2]  DACE_ALIGN(64);
        float warp_kernel[8]  DACE_ALIGN(64);

        {
            for (auto gemm_n = 0; gemm_n < 8; gemm_n += 1) {
                for (auto gemm_m = 0; gemm_m < 2; gemm_m += 1) {
                    {
                        float __inp = __tmp_165_53_r_in_from_3_0[gemm_m];
                        float __out;

                        ///////////////////
                        // Tasklet code (assign_165_32)
                        __out = __inp;
                        ///////////////////

                        warp_input[gemm_m] = __out;
                    }
                    {
                        float __inp = __tmp_166_54_r_in_from_3_0[gemm_n];
                        float __out;

                        ///////////////////
                        // Tasklet code (assign_166_32)
                        __out = __inp;
                        ///////////////////

                        warp_kernel[gemm_n] = __out;
                    }
                }
            }
        }
        {
            for (auto gemm_n = 0; gemm_n < 8; gemm_n += 1) {
                for (auto gemm_m = 0; gemm_m < 2; gemm_m += 1) {
                    float __tmp19;
                    float __tmp20;
                    {
                        float __in1 = warp_input[gemm_m];
                        float __in2 = warp_kernel[gemm_n];
                        float __out;

                        ///////////////////
                        // Tasklet code (_Mult_)
                        __out = (__in1 * __in2);
                        ///////////////////

                        __tmp19 = __out;
                    }
                    {
                        float __in2 = __tmp19;
                        float __in1 = __tmp_170_77_r_in_from_4_0[(gemm_m + (64 * gemm_n))];
                        float __out;

                        ///////////////////
                        // Tasklet code (_Add_)
                        __out = (__in1 + __in2);
                        ///////////////////

                        __tmp20 = __out;
                    }
                    {
                        float __inp = __tmp20;
                        float __out;

                        ///////////////////
                        // Tasklet code (assign_170_32)
                        __out = __inp;
                        ///////////////////

                        __tmp_170_32_w_out_of_4_1[(gemm_m + (64 * gemm_n))] = __out;
                    }
                }
            }
        }

    }
    
}

DACE_DFI void implicitGemmsplitKdace_dace_conv3d_179_4_180_8_186_16_0_0_15(const float* __restrict__ __tmp_187_54_r, float* __restrict__ __tmp_198_40_r_in_from_19_0, float* __restrict__ __tmp_199_24_w, int cta_m, int cta_n, int d_DHW, int d_HW, int d_batchsize, int d_inchannels, int d_kdim, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth, int gemm_m, int gemm_n, int warp_m, int warp_n) {
    int __tmp15;
    float tmp[1]  DACE_ALIGN(64);
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

        {
            float __out;

            ///////////////////
            // Tasklet code (assign_188_24)
            __out = 0;
            ///////////////////

            tmp[0] = __out;
        }
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

    }
    __tmp17 = (nopq_residual % d_HW);
    {
        int __tmp19;

        {
            int __inp = __tmp15;
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32(__inp);
            ///////////////////

            o = __out;
        }
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
    __sym___tmp20 = p;
    {

        {
            int __out;

            ///////////////////
            // Tasklet code (_convert_to_int32_)
            __out = dace::int32(__tmp21);
            ///////////////////

            q = __out;
        }
        {
            for (auto isplit_k = 0; isplit_k < 8; isplit_k += 1) {
                float __tmp24;
                {
                    float __in1 = tmp[0];
                    float __in2 = __tmp_198_40_r_in_from_19_0[(16 * isplit_k)];
                    float __out;

                    ///////////////////
                    // Tasklet code (_Add_)
                    __out = (__in1 + __in2);
                    ///////////////////

                    __tmp24 = __out;
                }
                {
                    float __inp = __tmp24;
                    float __out;

                    ///////////////////
                    // Tasklet code (assign_198_28)
                    __out = __inp;
                    ///////////////////

                    tmp[0] = __out;
                }
            }
        }

    }
    __sym___tmp22 = q;
    {

        {
            float __inp = tmp[0];
            float __out;

            ///////////////////
            // Tasklet code (assign_199_24)
            __out = __inp;
            ///////////////////

            __tmp_199_24_w[((((((((__sym___tmp13 * d_outchannels) * d_outdepth) * d_outheight) * d_outwidth) + ((__sym___tmp16 * d_outheight) * d_outwidth)) + (__sym___tmp20 * d_outwidth)) + __sym___tmp22) + (((d_outdepth * d_outheight) * d_outwidth) * ((cta_n + gemm_n) + warp_n)))] = __out;
        }

    }
    
}



int __dace_init_cuda(implicitGemmsplitKdace_dace_conv3d_t *__state, int d_batchsize, int d_inchannels, int d_kdim, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth) {
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

    

    __state->gpu_context = new dace::cuda::Context(4, 9);

    // Create cuda streams and events
    for(int i = 0; i < 4; ++i) {
        cudaStreamCreateWithFlags(&__state->gpu_context->streams[i], cudaStreamNonBlocking);
    }
    for(int i = 0; i < 9; ++i) {
        cudaEventCreateWithFlags(&__state->gpu_context->events[i], cudaEventDisableTiming);
    }

    

    return 0;
}

void __dace_exit_cuda(implicitGemmsplitKdace_dace_conv3d_t *__state) {
    

    // Destroy cuda streams and events
    for(int i = 0; i < 4; ++i) {
        cudaStreamDestroy(__state->gpu_context->streams[i]);
    }
    for(int i = 0; i < 9; ++i) {
        cudaEventDestroy(__state->gpu_context->events[i]);
    }

    delete __state->gpu_context;
}

__global__ void implicitGemmsplitKdace_dace_conv3d_118_0_0_0(const float * __restrict__ Input, const float * __restrict__ kernel, float * __restrict__ splitGemm, int d_batchsize, int d_inchannels, int d_kdim, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth) {
    {
        {
            {
                int cta_m = (64 * blockIdx.x);
                int cta_n = (32 * blockIdx.y);
                int isplit_k = blockIdx.z;
                __shared__ float cta_reducedk[2048];
                __shared__ int splitPartK[1];
                {
                    {
                        {
                            int warp_m = (2 * threadIdx.x);
                            int warp_n = (8 * threadIdx.y);
                            {
                                {
                                    {
                                        for (auto gemm_n = 0; gemm_n < 8; gemm_n += 1) {
                                            for (auto gemm_m = 0; gemm_m < 2; gemm_m += 1) {
                                                {
                                                    float __out;

                                                    ///////////////////
                                                    // Tasklet code (assign_122_24)
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
                __syncthreads();
                {
                    int __out;

                    ///////////////////
                    // Tasklet code (_convert_to_int32_)
                    __out = dace::int32((0.125 * (d_inchannels * (dace::math::ipow(d_kdim, 3)))));
                    ///////////////////

                    splitPartK[0] = __out;
                }
                {
                    int __map_125_e0 = splitPartK[0];
                    for (auto cta_k = 0; cta_k < __map_125_e0; cta_k += 8) {
                        __shared__ float cta_input[512];
                        __shared__ float cta_kernel[256];
                        {
                            {
                                {
                                    int warp_m = (2 * threadIdx.x);
                                    int warp_n = (8 * threadIdx.y);
                                    {
                                        {
                                            {
                                                for (auto warp_k = 0; warp_k < 8; warp_k += 1) {
                                                    {
                                                        for (auto gemm_n = 0; gemm_n < 8; gemm_n += 1) {
                                                            for (auto gemm_m = 0; gemm_m < 2; gemm_m += 1) {
                                                                implicitGemmsplitKdace_dace_conv3d_118_4_125_12_128_16_129_20_130_24_0_0_49(splitPartK[0], &Input[0], &kernel[0], splitPartK[0], cta_input[((gemm_m + (64 * warp_k)) + warp_m)], cta_kernel[((gemm_n + (32 * warp_k)) + warp_n)], cta_k, cta_m, cta_n, ((d_outdepth * d_outheight) * d_outwidth), (((d_inchannels * d_kdim) * d_kdim) * d_kdim), (d_outheight * d_outwidth), d_batchsize, d_inchannels, d_kdim, (d_kdim * d_kdim), ((d_kdim * d_kdim) * d_kdim), d_outchannels, d_outdepth, d_outheight, d_outwidth, gemm_m, gemm_n, isplit_k, warp_k, warp_m, warp_n);
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
                        __syncthreads();
                        {
                            {
                                {
                                    int warp_m = (2 * threadIdx.x);
                                    int warp_n = (8 * threadIdx.y);
                                    {
                                        {
                                            {
                                                for (auto warp_k = 0; warp_k < 8; warp_k += 1) {
                                                    implicitGemmsplitKdace_dace_conv3d_118_4_125_12_157_16_161_24_0_0_44(&Input[0], &cta_input[((64 * warp_k) + warp_m)], &cta_kernel[((32 * warp_k) + warp_n)], &cta_reducedk[(warp_m + (64 * warp_n))], &cta_reducedk[(warp_m + (64 * warp_n))], d_batchsize, d_inchannels, d_kdim, d_outdepth, d_outheight, d_outwidth);
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
                {
                    {
                        {
                            int warp_m = (2 * threadIdx.x);
                            int warp_n = (8 * threadIdx.y);
                            {
                                {
                                    {
                                        for (auto gemm_n = 0; gemm_n < 8; gemm_n += 1) {
                                            for (auto gemm_m = 0; gemm_m < 2; gemm_m += 1) {
                                                {
                                                    float __inp = cta_reducedk[(((gemm_m + (64 * gemm_n)) + warp_m) + (64 * warp_n))];
                                                    float __out;

                                                    ///////////////////
                                                    // Tasklet code (assign_177_20)
                                                    __out = __inp;
                                                    ///////////////////

                                                    splitGemm[((((cta_m + (((((d_batchsize * d_outchannels) * d_outdepth) * d_outheight) * d_outwidth) * isplit_k)) + ((((d_batchsize * d_outdepth) * d_outheight) * d_outwidth) * ((cta_n + gemm_n) + warp_n))) + gemm_m) + warp_m)] = __out;
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
}


DACE_EXPORTED void __dace_runkernel_implicitGemmsplitKdace_dace_conv3d_118_0_0_0(implicitGemmsplitKdace_dace_conv3d_t *__state, const float * __restrict__ Input, const float * __restrict__ kernel, float * __restrict__ splitGemm, int d_batchsize, int d_inchannels, int d_kdim, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth);
void __dace_runkernel_implicitGemmsplitKdace_dace_conv3d_118_0_0_0(implicitGemmsplitKdace_dace_conv3d_t *__state, const float * __restrict__ Input, const float * __restrict__ kernel, float * __restrict__ splitGemm, int d_batchsize, int d_inchannels, int d_kdim, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth)
{

    void  *implicitGemmsplitKdace_dace_conv3d_118_0_0_0_args[] = { (void *)&Input, (void *)&kernel, (void *)&splitGemm, (void *)&d_batchsize, (void *)&d_inchannels, (void *)&d_kdim, (void *)&d_outchannels, (void *)&d_outdepth, (void *)&d_outheight, (void *)&d_outwidth };
    cudaLaunchKernel((void*)implicitGemmsplitKdace_dace_conv3d_118_0_0_0, dim3(int_ceil((((d_batchsize * d_outdepth) * d_outheight) * d_outwidth), 64), int_ceil(d_outchannels, 32), 8), dim3(32, 4, 1), implicitGemmsplitKdace_dace_conv3d_118_0_0_0_args, 0, __state->gpu_context->streams[0]);
    DACE_CUDA_CHECK(cudaGetLastError());
    DACE_CUDA_CHECK(cudaDeviceSynchronize());
}
__global__ void implicitGemmsplitKdace_dace_conv3d_179_0_0_5(const float * __restrict__ Input, float * __restrict__ Output, const float * __restrict__ splitGemm, int d_batchsize, int d_inchannels, int d_kdim, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth) {
    {
        {
            int cta_m = (64 * blockIdx.x);
            int cta_n = (32 * blockIdx.y);
            {
                {
                    {
                        float warp_readk[128]  DACE_ALIGN(64);
                        int warp_m = (2 * threadIdx.x);
                        int warp_n = (8 * threadIdx.y);
                        {
                            {
                                {
                                    for (auto gemm_n = 0; gemm_n < 8; gemm_n += 1) {
                                        for (auto gemm_m = 0; gemm_m < 2; gemm_m += 1) {
                                            {
                                                for (auto isplit_k = 0; isplit_k < 8; isplit_k += 1) {
                                                    {
                                                        float __inp = splitGemm[((((cta_m + (((((d_batchsize * d_outchannels) * d_outdepth) * d_outheight) * d_outwidth) * isplit_k)) + ((((d_batchsize * d_outdepth) * d_outheight) * d_outwidth) * ((cta_n + gemm_n) + warp_n))) + gemm_m) + warp_m)];
                                                        float __out;

                                                        ///////////////////
                                                        // Tasklet code (assign_184_24)
                                                        __out = __inp;
                                                        ///////////////////

                                                        warp_readk[((gemm_m + (2 * gemm_n)) + (16 * isplit_k))] = __out;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                {
                                    for (auto gemm_n = 0; gemm_n < 8; gemm_n += 1) {
                                        for (auto gemm_m = 0; gemm_m < 2; gemm_m += 1) {
                                            implicitGemmsplitKdace_dace_conv3d_179_4_180_8_186_16_0_0_15(&Input[0], &warp_readk[(gemm_m + (2 * gemm_n))], &Output[0], cta_m, cta_n, ((d_outdepth * d_outheight) * d_outwidth), (d_outheight * d_outwidth), d_batchsize, d_inchannels, d_kdim, d_outchannels, d_outdepth, d_outheight, d_outwidth, gemm_m, gemm_n, warp_m, warp_n);
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


DACE_EXPORTED void __dace_runkernel_implicitGemmsplitKdace_dace_conv3d_179_0_0_5(implicitGemmsplitKdace_dace_conv3d_t *__state, const float * __restrict__ Input, float * __restrict__ Output, const float * __restrict__ splitGemm, int d_batchsize, int d_inchannels, int d_kdim, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth);
void __dace_runkernel_implicitGemmsplitKdace_dace_conv3d_179_0_0_5(implicitGemmsplitKdace_dace_conv3d_t *__state, const float * __restrict__ Input, float * __restrict__ Output, const float * __restrict__ splitGemm, int d_batchsize, int d_inchannels, int d_kdim, int d_outchannels, int d_outdepth, int d_outheight, int d_outwidth)
{

    void  *implicitGemmsplitKdace_dace_conv3d_179_0_0_5_args[] = { (void *)&Input, (void *)&Output, (void *)&splitGemm, (void *)&d_batchsize, (void *)&d_inchannels, (void *)&d_kdim, (void *)&d_outchannels, (void *)&d_outdepth, (void *)&d_outheight, (void *)&d_outwidth };
    cudaLaunchKernel((void*)implicitGemmsplitKdace_dace_conv3d_179_0_0_5, dim3(int_ceil((((d_batchsize * d_outdepth) * d_outheight) * d_outwidth), 64), int_ceil(d_outchannels, 32), 1), dim3(32, 4, 1), implicitGemmsplitKdace_dace_conv3d_179_0_0_5_args, 0, __state->gpu_context->streams[0]);
    DACE_CUDA_CHECK(cudaGetLastError());
    DACE_CUDA_CHECK(cudaDeviceSynchronize());
}

