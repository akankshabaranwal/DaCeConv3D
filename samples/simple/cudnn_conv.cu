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

using namespace std;

#define CUDA_CALL(f) { \
  cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

#define CUDNN_CALL(f) { \
  cudnnStatus_t err = (f); \
  if (err != CUDNN_STATUS_SUCCESS) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
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
        const int in_n = 1, in_c = 1, in_h = 5, in_w = 5;
        std::cout << "in_n: " << in_n << ", in_c: " << in_c << ", in_h: " << in_h << ", in_w: " << in_w << std::endl;
        cudnnTensorDescriptor_t in_desc;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in_n, in_c, in_h, in_w));
        float *in_data;
        CUDA_CALL(cudaMalloc( &in_data, in_n * in_c * in_h * in_w * sizeof(float)));

        // filter
        const int filt_k = 1, filt_c = 1, filt_h = 2, filt_w = 2;
        std::cout << "filt_k: " << filt_k << ", filt_c: " << filt_c << ", filt_h: " << filt_h << ", filt_w: " << filt_w << std::endl;
        cudnnFilterDescriptor_t filt_desc;
        CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
        CUDNN_CALL(cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filt_k, filt_c, filt_h, filt_w));

        float *filt_data;
        CUDA_CALL(cudaMalloc(&filt_data, filt_k * filt_c * filt_h * filt_w * sizeof(float)));

        // convolution
        const int pad_h = 1, pad_w = 1, str_h = 1, str_w = 1, dil_h = 1, dil_w = 1;
        std::cout << "pad_h: " << pad_h << ", pad_w: " << pad_w << ", str_h: " << str_h << ", str_w: " << str_w << ", dil_h: " << dil_h << ", dil_w: " << dil_w << std::endl;
        cudnnConvolutionDescriptor_t conv_desc;
        CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
        CUDNN_CALL(cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w, str_h, str_w, dil_h, dil_w, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

        // output
        int out_n, out_c, out_h, out_w;
        CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim( conv_desc, in_desc, filt_desc, &out_n, &out_c, &out_h, &out_w));
        std::cout << "out_n: " << out_n << ", out_c: " << out_c << ", out_h: " << out_h << ", out_w: " << out_w << std::endl;
        cudnnTensorDescriptor_t out_desc;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
        CUDNN_CALL(cudnnSetTensor4dDescriptor( out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w));
        float *out_data;
        CUDA_CALL(cudaMalloc(&out_data, out_n * out_c * out_h * out_w * sizeof(float)));

        void *search_ws;
        cudaMalloc(&search_ws, 33554432);        
        cudnnConvolutionFwdAlgoPerf_t perfResults;
        int requestedAlgoCount = 1;
        int returnedAlgoCount = 1;
        CUDNN_CALL(cudnnFindConvolutionForwardAlgorithmEx(cudnn, in_desc, in_data, filt_desc, filt_data, conv_desc, 
                                                        out_desc, out_data, requestedAlgoCount, &returnedAlgoCount, &perfResults, search_ws, 33554432));

        cudnnConvolutionFwdAlgo_t selectedAlgo;
        selectedAlgo = perfResults.algo;
        size_t ws_size;
        CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(cudnn, in_desc, filt_desc, conv_desc, out_desc, selectedAlgo, &ws_size));
        std::cerr << "Workspace size: " << (ws_size ) << "bytes"<< std::endl;
        void* d_workspace{nullptr};
        cudaMalloc(&d_workspace, ws_size);
        const float alpha = 1.0f, beta = 0.0f;
        CUDNN_CALL(cudnnConvolutionForward(cudnn, &alpha, in_desc, in_data, filt_desc, filt_data, conv_desc, selectedAlgo, d_workspace, ws_size, &beta, out_desc, out_data));
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