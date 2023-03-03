#include <iomanip>
#include <cstdlib>
#include <miopen/miopen.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "hip/hip_runtime.h"
#include <vector>

#define HIP_ASSERT(x) (assert(x==hipSuccess))
using namespace std;
// To compile code:  
// hipcc miopen_conv.cpp 
// -I /apps/ault/spack/opt/spack/linux-centos8-zen/gcc-8.4.1/miopen-hip-5.4.3-ow45oosheu4sig6ribgawitfphdis47x/include/ 
// -L /apps/ault/spack/opt/spack/linux-centos8-zen/gcc-8.4.1/miopen-hip-5.4.3-ow45oosheu4sig6ribgawitfphdis47x/lib/ -lMIOpen

void throw_miopen_err(miopenStatus_t status, int line, const char* filename) {
    if (status != miopenStatusSuccess) {
        std::stringstream ss;
        ss << "MIOPEN failure: " << status <<
              " in " << filename << " at line: " << line << std::endl;
        throw std::runtime_error(ss.str());
    }
}

#define CHECK_MIOPEN_ERROR(status) throw_miopen_err(status, __LINE__, __FILE__)

int main()
{
    // Create handle
    miopenHandle_t miopen_h;
    CHECK_MIOPEN_ERROR(miopenCreate(&miopen_h));
    std::cout<<" Created miopen handle" <<std::endl;

    // input
    const int in_n = 4, in_c = 1, in_d = 8, in_h = 8, in_w = 8;
    std::cout << "in_n: " << in_n << ", in_c: " << in_c << ", in_d: " << in_d << ", in_h: " << in_h << ", in_w: " << in_w << std::endl;
    miopenTensorDescriptor_t in_desc;
    CHECK_MIOPEN_ERROR(miopenCreateTensorDescriptor(&in_desc));
    int dims[5];
    dims[0]=in_n; dims[1]=in_c; dims[2]=in_d;dims[3]=in_h;dims[4]=in_w;
    int strides[5];
    strides[0]=in_c*in_d*in_h*in_w; strides[1] =in_d*in_h*in_w; strides[2]= in_h*in_w; strides[3]= in_w; strides[4] =1;
    CHECK_MIOPEN_ERROR(miopenSetTensorDescriptor(in_desc, miopenFloat, 5, dims, strides));
    float *in_data;
    int bytes = in_n * in_c * in_d * in_h * in_w * sizeof(float);
    HIP_ASSERT(hipMalloc( &in_data, bytes));

    // filter
    const int filt_k = 1, filt_c = 1, filt_d = 3, filt_h = 3, filt_w = 3;
    std::cout << "filt_k: " << filt_k << ", filt_c: " << filt_c << ", filt_d: " << filt_d << ", filt_h: " << filt_h << ", filt_w: " << filt_w << std::endl;
    miopenTensorDescriptor_t filt_desc;
    CHECK_MIOPEN_ERROR(miopenCreateTensorDescriptor(&filt_desc));
    int filtdims[5];
    filtdims[0]=filt_k; filtdims[1]=filt_c; filtdims[2]=filt_d; filtdims[3]=filt_h; filtdims[4]=filt_w;
    int filtstrides[5];
    filtstrides[0]=filt_c*filt_d*filt_h*filt_w; filtstrides[1] =filt_d*filt_h*filt_w; filtstrides[2]= filt_h*filt_w; filtstrides[3]= filt_w; filtstrides[4] =1;
    CHECK_MIOPEN_ERROR(miopenSetTensorDescriptor(filt_desc, miopenFloat, 5, filtdims, filtstrides));
    float *filt_data;
    int filtbytes = filt_k * filt_c * filt_d * filt_h * filt_w * sizeof(float);
    HIP_ASSERT(hipMalloc( &filt_data, filtbytes));

    // convolution
    const int pad_d = 0, pad_h = 0, pad_w = 0, str_d=1, str_h = 1, str_w = 1, dil_d=1, dil_h = 1, dil_w = 1;
    std::cout << "pad_d: " << pad_d << ", pad_h: " << pad_h << ", pad_w: " << pad_w << ", str_d: " << str_d << ", str_h: " << str_h << ", str_w: " << str_w << ", dil_d: " << dil_d <<", dil_h: " << dil_h << ", dil_w: " << dil_w << std::endl;
    miopenConvolutionDescriptor_t conv_desc;
    CHECK_MIOPEN_ERROR(miopenCreateConvolutionDescriptor(&conv_desc));
    miopenConvolutionMode_t c_mode = miopenConvolution;
    vector<int> convpad = {pad_d, pad_h, pad_w};
    vector<int> filtstr = {str_d, str_h, str_w};
    vector<int> convdil = {dil_d, dil_h, dil_w};
    CHECK_MIOPEN_ERROR(miopenInitConvolutionNdDescriptor(conv_desc, 
                                                        3, 
                                                        convpad.data(), 
                                                        filtstr.data(), 
                                                        convdil.data(), 
                                                        c_mode));

    // Get convolution output dimension
    miopenTensorDescriptor_t out_desc;
    CHECK_MIOPEN_ERROR(miopenCreateTensorDescriptor(&out_desc));
    std::vector<int> outdims{0,0,0,0,0};
    int ndim=3;
    CHECK_MIOPEN_ERROR(miopenGetConvolutionNdForwardOutputDim(conv_desc, 
                                                            in_desc, 
                                                            filt_desc, 
                                                            &ndim, 
                                                            outdims.data()));
    int out_n = outdims[0];
    int out_c = outdims[1];
    int out_d = outdims[2];
    int out_h = outdims[3];
    int out_w = outdims[4];
    std::cout << "out_n: " << out_n << ", out_c: " << out_c << ", out_d: "<< out_d<< ", out_h: " << out_h << ", out_w: " << out_w << std::endl;
    vector<int> outstrides = {out_c*out_d*out_h*out_w, out_d*out_h*out_w, out_h*out_w, out_w, 1};

    CHECK_MIOPEN_ERROR(miopenSetTensorDescriptor(out_desc, miopenFloat, 5, outdims.data(), outstrides.data()));
    float *out_data;
    int outbytes = out_n * out_c * out_d * out_h * out_w * sizeof(float);
    HIP_ASSERT(hipMalloc( &out_data, outbytes));

    size_t ws_size=3355443200;
    
    CHECK_MIOPEN_ERROR(miopenConvolutionForwardGetWorkSpaceSize(miopen_h, 
                                                            filt_desc, 
                                                            in_desc, 
                                                            conv_desc, 
                                                            out_desc, 
                                                            &ws_size));
    std::cerr << "Workspace size: " << (ws_size ) << "bytes"<< std::endl;
    
    // Run the find algorithm to get the most optimal algorithm to run.
    void *search_ws;
    HIP_ASSERT(hipMalloc( &search_ws, ws_size));
    int returnedAlgoCount;
    miopenConvFwdAlgorithm_t selectedAlgo;
    miopenConvAlgoPerf_t perfResults[1];
    CHECK_MIOPEN_ERROR(miopenFindConvolutionForwardAlgorithm(miopen_h, 
                                                            in_desc, in_data,
                                                            filt_desc, filt_data, 
                                                            conv_desc, 
                                                            out_desc, out_data, 
                                                            1, 
                                                            &returnedAlgoCount,
                                                            perfResults, search_ws, 
                                                            ws_size, true));
    
    void* d_workspace{nullptr};
    HIP_ASSERT(hipMalloc(&d_workspace, ws_size));
    const float alpha = 1.0f, beta = 0.0f;
    selectedAlgo = perfResults->fwd_algo;
    selectedAlgo = miopenConvolutionFwdAlgoImplicitGEMM;
    std::cout<<perfResults->fwd_algo;
    CHECK_MIOPEN_ERROR(miopenConvolutionForward(miopen_h, &alpha, 
                                                in_desc, in_data, 
                                                filt_desc, filt_data,
                                                conv_desc, selectedAlgo, 
                                                &beta, out_desc, out_data,
                                                d_workspace, ws_size));
    return 0;
}