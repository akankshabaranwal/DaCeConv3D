import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
import libcudnn, ctypes
import numpy as np

# https://github.com/hannes-brt/cudnn-python-wrappers/blob/master/example.py
# Create a cuDNN context
cudnn_context = libcudnn.cudnnCreate()
print("Created cudnn context")

# Set some options and tensor dimensions
tensor_format = libcudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW']
data_type = libcudnn.cudnnDataType['CUDNN_DATA_FLOAT']
tensor_dim = 5
conv_dim = tensor_dim-2
convolution_mode = libcudnn.cudnnConvolutionMode['CUDNN_CROSS_CORRELATION']
convolution_algo = libcudnn.cudnnConvolutionFwdAlgo['CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM']

# Setting Input tensor
'''        
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));        
vector<int> dims = {in_n, in_c, in_d, in_h, in_w};
vector<int> strides = {in_c*in_d*in_h*in_w, in_d*in_h*in_w, in_h*in_w, in_w, 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(in_desc, 
                                CUDNN_DATA_FLOAT, 
                                5, 
                                dims.data(), 
                                strides.data()));
'''
in_n = 1
in_c = 1
in_d = 5
in_h = 5
in_w = 5
c_int_p = ctypes.POINTER(ctypes.c_int)
dims = [in_n, in_c, in_d, in_h, in_w]
strides = [in_c*in_d*in_h*in_w, in_d*in_h*in_w, in_h*in_w, in_w, 1]
in_desc = libcudnn.cudnnCreateTensorDescriptor()
libcudnn.cudnnSetTensorNdDescriptor(in_desc,
                                    data_type, 
                                    tensor_dim,
                                    dims, 
                                    strides)
'''
float *in_data;
CUDA_CALL(cudaMalloc( &in_data, in_n * in_c * in_d * in_h * in_w * sizeof(float)));
'''
in_data_g = gpuarray.to_gpu(np.random.rand(in_n, in_c, in_d, in_h, in_w).astype(np.float32))                                    
# Get pointers to GPU memory
in_data = ctypes.c_void_p(int(in_data_g.gpudata))
print("Input tensor created")

# Setting filter tensor
filt_k = 1
filt_d = 3
filt_h = 3
filt_w = 3
filt_desc = libcudnn.cudnnCreateFilterDescriptor()
filt_dims = [filt_k, in_c, filt_d, filt_h, filt_w]
'''
CUDNN_CALL(cudnnSetFilterNdDescriptor(filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5, filtdims.data()));
'''
libcudnn.cudnnSetFilterNdDescriptor(filt_desc,
                                    data_type,
                                    tensor_format,
                                    tensor_dim,
                                    filt_dims)
'''
float *filt_data;
CUDA_CALL(cudaMalloc(&filt_data, filt_k * filt_c * filt_d * filt_h * filt_w * sizeof(float)));
'''
filt_data_g = gpuarray.to_gpu(np.random.rand(filt_k, in_c, filt_d, filt_h, filt_w).astype(np.float32))                                    
filt_data = ctypes.c_void_p(int(filt_data_g.gpudata))

print("Filter tensor created")

# Setting convolution tensor
pad_d = 1 
pad_h = 1 
pad_w = 1 
str_d = 1
str_h = 1
str_w = 1
dil_d = 1
dil_h = 1 
dil_w = 1
alpha = 1.0
beta = 0

'''
cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
vector<int> convpad = {pad_d, pad_h, pad_w};
vector<int> filtstr = {str_d, str_h, str_w};
vector<int> convdil = {dil_d, dil_h, dil_w};
CUDNN_CALL(cudnnSetConvolutionNdDescriptor(conv_desc, 3, convpad.data(), filtstr.data(), convdil.data(), CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
'''
conv_desc = libcudnn.cudnnCreateConvolutionDescriptor()
convpad = [pad_d, pad_h, pad_w]
filtstr = [str_d, str_h, str_w]
convdil = [dil_d, dil_h, dil_w]
libcudnn.cudnnSetConvolutionNdDescriptor(conv_desc,
                                        conv_dim,
                                        convpad,
                                        filtstr,
                                        convdil,
                                        convolution_mode,
                                        data_type)

print("Convolution descriptor created")

# Set output dimensions
'''        
int outdims[5];
CUDNN_CALL(cudnnGetConvolutionNdForwardOutputDim( conv_desc, 
                                                in_desc, 
                                                filt_desc, 
                                                5, 
                                                outdims));
cudnnTensorDescriptor_t out_desc;    
'''
outdimsinit = [0, 0, 0, 0, 0]
outdims = libcudnn.cudnnGetConvolutionNdForwardOutputDim(conv_desc, 
                                            in_desc, 
                                            filt_desc, 
                                            tensor_dim, 
                                            outdimsinit)
print(f"Output dimensions have been found as {outdims}")
'''        
vector<int> outstrides = {out_c*out_d*out_h*out_w, out_d*out_h*out_w, out_h*out_w, out_w, 1};        
CUDNN_CALL(cudnnSetTensorNdDescriptor(out_desc, CUDNN_DATA_FLOAT, 5, outdims, outstrides.data()));
float *out_data;
CUDA_CALL(cudaMalloc(&out_data, out_n * out_c * out_d * out_h * out_w * sizeof(float)));
'''
out_n = outdims[0]
out_c = outdims[1]
out_d = outdims[2]
out_h = outdims[3]
out_w = outdims[4]
outstrides = [ outdims[1]*outdims[2]*outdims[3]*outdims[4], outdims[2]*outdims[3]*outdims[4], outdims[3]*outdims[4], outdims[4], 1]
out_desc = libcudnn.cudnnCreateTensorDescriptor()
libcudnn.cudnnSetTensorNdDescriptor(out_desc,
                                    data_type, 
                                    tensor_dim,
                                    outdims, 
                                    outstrides)
'''
float *out_data;
CUDA_CALL(cudaMalloc(&out_data, out_n * out_c * out_d * out_h * out_w * sizeof(float)));
'''
out_data_g = gpuarray.to_gpu(np.random.rand(out_n, out_c, out_d, out_h, out_w).astype(np.float32))                                    
out_data = ctypes.c_void_p(int(out_data_g.gpudata))
print("Output tensor created")

# Not implemented find convolution algorithm

# Find required workspace size.
'''
        size_t ws_size=33554432;
        CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(cudnn, in_desc, filt_desc, conv_desc, out_desc, selectedAlgo, &ws_size));
        std::cerr << "Workspace size: " << (ws_size ) << "bytes"<< std::endl;
        
        void* d_workspace{nullptr};
        cudaMalloc(&d_workspace, ws_size);
'''
print("Cudnn algorithm = %d" % convolution_algo)
ws_size = libcudnn.cudnnGetConvolutionForwardWorkspaceSize(cudnn_context, 
                                                        in_desc,
                                                        filt_desc, 
                                                        conv_desc, 
                                                        out_desc, 
                                                        convolution_algo)
ws_ptr  = drv.mem_alloc(ws_size.value) if ws_size.value > 0 else 0
ws_data = ctypes.c_void_p(int(ws_ptr))

# Perform the actual convolution
'''
CUDNN_CALL(cudnnConvolutionForward(cudnn, &alpha, in_desc, in_data, filt_desc, filt_data, conv_desc, selectedAlgo, d_workspace, ws_size, &beta, out_desc, out_data));
'''
libcudnn.cudnnConvolutionForward(cudnn_context, 
                                alpha, 
                                in_desc, 
                                in_data,
                                filt_desc, 
                                filt_data, 
                                conv_desc, 
                                convolution_algo,
                                ws_data, 
                                ws_size.value, 
                                beta,
                                out_desc, 
                                out_data)
print(type(out_data_g))
out_data_np = np.array(out_data_g)
print(out_data_np)

#TODO: Verification for cudnn convolution call
#TODO: Integrate this with the existing benchmarking code.
# TODO: Code to free up memory and cuda handles.
ws_ptr = None
libcudnn.cudnnDestroyTensorDescriptor(in_desc)
libcudnn.cudnnDestroyTensorDescriptor(out_desc)
libcudnn.cudnnDestroyFilterDescriptor(filt_desc)
libcudnn.cudnnDestroyConvolutionDescriptor(conv_desc)
libcudnn.cudnnDestroy(cudnn_context)