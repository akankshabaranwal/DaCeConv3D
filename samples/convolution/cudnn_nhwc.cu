// References:
// http://courses.cms.caltech.edu/cs101gpu/2022_lectures/cs179_2022_lec17.pdf
// https://gist.github.com/odashi/1c20ba90388cf02330e1b95963d78039
// https://medium.com/@rohitdwivedula/minimal-cudnn-c-hello-world-example-47d3c6b60b73
// API Reference: https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionForward
// https://gist.github.com/goldsborough/865e6717e64fbae75cdaf6c9914a130d

#include <iomanip>
#include <cstdlib>
#include <vector>
#include <cuda.h>
#include <cudnn.h>
#include <cassert>
#include <iostream>
#include <algorithm>

using namespace std;

#define CUDA_CALL(f) { \
  cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    std::cout \
        << "    Error occurred: " << err << ' ' << cudaGetErrorString(err) <<' '<< __LINE__ << std::endl; \
    std::exit(1); \
  } \
}

#define CUDNN_CALL(f) { \
  cudnnStatus_t err = (f); \
  if (err != CUDNN_STATUS_SUCCESS) { \
    std::cout \
        << "    Error occurred: " << err << ' '<< cudnnGetErrorString(err)<<' '<<__LINE__<< std::endl; \
    std::exit(1); \
  } \
}

int main()
{
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);
        std::cout << "Found " << numGPUs << " GPUs." << std::endl;
        cudaSetDevice(0); // use GPU0
        int device; 
        struct cudaDeviceProp devProp;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&devProp, device);
        std::cout << "Compute capability:" << devProp.major << "." << devProp.minor << std::endl;

        cudnnHandle_t cudnn;
        CUDNN_CALL(cudnnCreate(&cudnn));
        std::cout << "Created cuDNN handle" << std::endl;

        // input
        const int in_n = 4, in_c = 4, in_d = 128, in_h = 128, in_w = 128;
        std::cout << "in_n: " << in_n << ", in_c: " << in_c << ", in_d: " << in_d << ", in_h: " << in_h << ", in_w: " << in_w << std::endl;
        cudnnTensorDescriptor_t in_desc;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
        int dims[5];
        dims[0]=in_n; dims[1]=in_c; dims[2]=in_d; dims[3]=in_h; dims[4]=in_w; 
        //dims[0]=in_n; dims[1]=in_d; dims[2]=in_h; dims[3]=in_w; dims[4]=in_c; 
        //int strides[5];
        //strides[0]=in_c*in_d*in_h*in_w; strides[1]=in_d*in_h*in_w; strides[2]=in_h*in_w; strides[3]=in_w; strides[4]=1;
        //strides[0]=in_d*in_h*in_w*in_c; strides[1]=in_h*in_w*in_c; strides[2]=in_w*in_c; strides[3]=in_c; strides[4]=1;
        CUDNN_CALL(cudnnSetTensorNdDescriptorEx(in_desc,
                                        CUDNN_TENSOR_NHWC,
                                        CUDNN_DATA_FLOAT,
                                        5,
                                        dims));
        float *in_data;
        CUDA_CALL(cudaMalloc( &in_data, in_n * in_c * in_d * in_h * in_w * sizeof(float)));

        // filter
        const int filt_k = 1, filt_c = 1, filt_d = 3, filt_h = 3, filt_w = 3;
        std::cout << "filt_k: " << filt_k << ", filt_c: " << filt_c << ", filt_d: " << filt_d << ", filt_h: " << filt_h << ", filt_w: " << filt_w << std::endl;
        cudnnFilterDescriptor_t filt_desc;
        CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
        vector<int> filtdims = {filt_k, in_c, filt_d, filt_h, filt_w};
        //vector<int> filtdims = {filt_k, filt_d, filt_h, filt_w, in_c};
        CUDNN_CALL(cudnnSetFilterNdDescriptor(filt_desc, 
                                              CUDNN_DATA_FLOAT, 
                                              CUDNN_TENSOR_NHWC,
                                              5, 
                                              filtdims.data()));
        float *filt_data;
        CUDA_CALL(cudaMalloc(&filt_data, filt_k * filt_c * filt_d * filt_h * filt_w * sizeof(float)));

        // convolution
        const int pad_d = 0, pad_h = 0, pad_w = 0, str_d=1, str_h = 1, str_w = 1, dil_d=1, dil_h = 1, dil_w = 1;
        std::cout << "pad_d: " << pad_d << ", pad_h: " << pad_h << ", pad_w: " << pad_w << ", str_d: " << str_d << ", str_h: " << str_h << ", str_w: " << str_w << ", dil_d: " << dil_d <<", dil_h: " << dil_h << ", dil_w: " << dil_w << std::endl;
        cudnnConvolutionDescriptor_t conv_desc;
        CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
        vector<int> convpad = {pad_d, pad_h, pad_w};
        vector<int> filtstr = {str_d, str_h, str_w};
        vector<int> convdil = {dil_d, dil_h, dil_w};
        CUDNN_CALL(cudnnSetConvolutionNdDescriptor(conv_desc, 
                                                  3, 
                                                  convpad.data(), 
                                                  filtstr.data(), 
                                                  convdil.data(), 
                                                  CUDNN_CROSS_CORRELATION, 
                                                  CUDNN_DATA_FLOAT));

        // output
        int outdims[5];
        CUDNN_CALL(cudnnGetConvolutionNdForwardOutputDim( conv_desc,
                                                        in_desc,
                                                        filt_desc,
                                                        5,
                                                        outdims));

        cudnnTensorDescriptor_t out_desc;        
        CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
        int out_n = outdims[0]; int out_c = outdims[1]; int out_d = outdims[2]; int out_h = outdims[3]; int out_w = outdims[4];
        //int out_n = outdims[0]; int out_d = outdims[1]; int out_h = outdims[2]; int out_w = outdims[3]; int out_c = outdims[4];

        std::cout << "out_n: " << out_n << ", out_c: " << out_c << ", out_d: "<< out_d<< ", out_h: " << out_h << ", out_w: " << out_w << std::endl;
        //vector<int> outstrides = {out_c*out_d*out_h*out_w, out_d*out_h*out_w, out_h*out_w, out_w, 1};
        vector<int> outstrides = { out_d*out_h*out_w*out_c, out_h*out_w*out_c, out_w*out_c, out_c, 1};
        
        CUDNN_CALL(cudnnSetTensorNdDescriptorEx(out_desc, 
                                              CUDNN_TENSOR_NHWC,
                                              CUDNN_DATA_FLOAT,
                                              5,
                                              outdims));
        float *out_data;
        CUDA_CALL(cudaMalloc(&out_data, out_n * out_c * out_d * out_h * out_w * sizeof(float)));

        void *search_ws;
        cudaMalloc(&search_ws, 33554432);        
        cudnnConvolutionFwdAlgoPerf_t perfResults[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
        int returnedAlgoCount;
        CUDNN_CALL(cudnnFindConvolutionForwardAlgorithmEx(cudnn, 
                                                        in_desc,
                                                        in_data,
                                                        filt_desc, 
                                                        filt_data, 
                                                        conv_desc, 
                                                        out_desc, 
                                                        out_data, 
                                                        CUDNN_CONVOLUTION_FWD_ALGO_COUNT, 
                                                        &returnedAlgoCount, 
                                                        perfResults, 
                                                        search_ws, 
                                                        33554432));
      for (int i=0; i<returnedAlgoCount; i++)
        std::cout<<perfResults[i].status<<' '<<perfResults[i].algo<<std::endl;


      // Till here the code works.
        assert(in_desc!=nullptr);
        assert(filt_desc!=nullptr);
        assert(out_desc!=nullptr);
        assert(conv_desc!=nullptr);

        cudaFree(search_ws);
        
        cudnnConvolutionFwdAlgo_t selectedAlgo;
        //selectedAlgo = perfResults.algo;
        selectedAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
        std::cout<<selectedAlgo;

        //return 0;
        
        size_t ws_size=3355443200;
        CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(cudnn, 
                                                          in_desc, 
                                                          filt_desc, 
                                                          conv_desc, 
                                                          out_desc, 
                                                          selectedAlgo, 
                                                          &ws_size));
        std::cerr << "Workspace size: " << (ws_size ) << "bytes"<< std::endl;
        
        void* d_workspace{nullptr};
        cudaMalloc(&d_workspace, ws_size);
        const float alpha = 1.0f, beta = 0.0f;
        CUDNN_CALL(cudnnConvolutionForward(cudnn, 
                                        &alpha, 
                                        in_desc, 
                                        in_data, 
                                        filt_desc, 
                                        filt_data, 
                                        conv_desc, 
                                        selectedAlgo, 
                                        d_workspace, 
                                        ws_size, 
                                        &beta, 
                                        out_desc, 
                                        out_data));
        cudaFree(in_data);
        cudaFree(out_data);
        cudaFree(filt_data);
        cudaFree(d_workspace);
        cudnnDestroyTensorDescriptor(in_desc);
        cudnnDestroyTensorDescriptor(out_desc);
        cudnnDestroyFilterDescriptor(filt_desc);
        cudnnDestroyConvolutionDescriptor(conv_desc);
}

// Command line to compile: nvcc cudnn_conv.cu -I /users/abaranwa/cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive/include/ -L /users/abaranwa/cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive/lib64/ -lcudnn