import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
import libcudnn, ctypes
import numpy as np

# Create a cuDNN context
cudnn_context = libcudnn.cudnnCreate()
print("Created cudnn context")

# Set some options and tensor dimensions
tensor_format = libcudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW']
data_type = libcudnn.cudnnDataType['CUDNN_DATA_FLOAT']
tensor_dim = 5
convolution_mode = libcudnn.cudnnConvolutionMode['CUDNN_CROSS_CORRELATION']

# Set convolution parameters
in_n = 1
in_c = 1
in_d = 5
in_h = 5
in_w = 5

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
c_int_p = ctypes.POINTER(ctypes.c_int)

dims = [in_n, in_c, in_d, in_h, in_w]
strides = [in_c*in_d*in_h*in_w, in_d*in_h*in_w, in_h*in_w, in_w, 1]

in_desc = libcudnn.cudnnCreateTensorDescriptor()
libcudnn.cudnnSetTensorNdDescriptor(in_desc,
                                    data_type, 
                                    tensor_dim,
                                    dims, 
                                    strides)

depth_filter = 7
height_filter = 7
width_filter = 7
pad_d = 3
pad_h = 3
pad_w = 3
stride_d = 1
stride_h = 1
stride_w = 1
upscaled = 1
upscaleh = 1
upscalew = 1
alpha = 1.0
beta = 1.0