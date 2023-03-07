import ctypes
import libmiopen
import libhip
import numpy as np
import torch

miopen_context = libmiopen.miopenCreate()
print("Created miopen context")

# Set some options and tensor dimensions
data_type = libmiopen.miopenDatatype['miopenFloat']
tensor_dim = 5
conv_dim = tensor_dim-2
convolution_mode = libmiopen.miopenConvolutionMode['miopenConvolution']
convolution_algo = libmiopen.miopenConvFwdAlgo['miopenConvolutionFwdAlgoDirect']

# Initializing input data pointer
in_n = 4
in_c = 3
in_d = 8
in_h = 8
in_w = 8
in_dims = [in_n, in_c, in_d, in_h, in_w]
in_strides = [in_c*in_d*in_h*in_w, in_d*in_h*in_w, in_h*in_w, in_w, 1]
in_desc = libmiopen.miopenCreateTensorDescriptor()
libmiopen.miopenSetTensorDescriptor(in_desc,
                                    data_type, 
                                    tensor_dim,
                                    in_dims, 
                                    in_strides)
in_bytes = in_n*in_c*in_d*in_h*in_w*ctypes.sizeof(ctypes.c_float)
in_data = libhip.hipMalloc(in_bytes)
tmp_input = torch.rand(in_n, in_c, in_d, in_h, in_w).cuda()
input_data_ptr = tmp_input.data_ptr()
libhip.hipMemcpyDtoD(in_data, input_data_ptr, in_bytes)
print("Input tensor created")

# Initializing filter pointer
filt_k = 1
filt_c = 3
filt_d = 3
filt_h = 3
filt_w = 3
filt_dims = [filt_k, filt_c, filt_d, filt_h, filt_w]
filt_strides = [filt_c*filt_d*filt_h*filt_w, filt_d*filt_h*filt_w, filt_h*filt_w, filt_w, 1]
filt_desc = libmiopen.miopenCreateTensorDescriptor()
libmiopen.miopenSetTensorDescriptor(filt_desc,
                                    data_type, 
                                    tensor_dim,
                                    filt_dims, 
                                    filt_strides)
data_ptr = tmp_input.data_ptr()
libhip.hipMemcpyDtoD(in_data, data_ptr, in_bytes)
filt_bytes = filt_k*filt_c*filt_d*filt_h*filt_w*ctypes.sizeof(ctypes.c_float)
filt_data = libhip.hipMalloc(filt_bytes)
tmp_filt = torch.rand(filt_k, filt_c, filt_d, filt_h, filt_w).cuda()
filt_data_ptr = tmp_filt.data_ptr()
libhip.hipMemcpyDtoD(filt_data, filt_data_ptr, filt_bytes)
print("Filter tensor created")

# Convolution descriptor
pad_d = 0
pad_h = 0
pad_w = 0
str_d = 1
str_h = 1
str_w = 1
dil_d = 1
dil_h = 1
dil_w = 1
conv_desc = libmiopen.miopenCreateConvolutionDescriptor()
convdim = 3
convpad = [pad_d, pad_h, pad_w]
filtstr = [str_d, str_h, str_w]
convdil = [dil_d, dil_h, dil_w]
libmiopen.miopenInitConvolutionNdDescriptor(conv_desc, 
                                            convdim,
                                            convpad,
                                            filtstr,
                                            convdil,
                                            convolution_mode,
                                            data_type)
print("Convolution descriptor created")

# Output descriptor
outdimsinit = [0, 0, 0, 0, 0]
tensor_dim = [5]
outdims = libmiopen.miopenGetConvolutionNdForwardOutputDim(conv_desc, 
                                            in_desc,
                                            filt_desc,
                                            tensor_dim,
                                            outdimsinit)                                            
out_n = outdims[0]
out_c = outdims[1]
out_d = outdims[2]
out_h = outdims[3]
out_w = outdims[4]
outstrides = [outdims[1]*outdims[2]*outdims[3]*outdims[4], outdims[2]*outdims[3]*outdims[4], outdims[3]*outdims[4], outdims[4], 1]
outconvdim = 5
out_desc = libmiopen.miopenCreateTensorDescriptor()
libmiopen.miopenSetTensorDescriptor(out_desc,
                                    data_type,
                                    outconvdim,
                                    outdims, 
                                    outstrides)
out_bytes = int(out_n*out_c*out_d*out_h*out_w*ctypes.sizeof(ctypes.c_float))
out_data = libhip.hipMalloc(out_bytes)
tmp_output = torch.zeros(out_n, out_c, out_d, out_h, out_w).cuda()
out_data_ptr = tmp_output.data_ptr()

print(outdims)
print("Output tensor created")

# Find workspace size
ws_size =  libmiopen.miopenConvolutionForwardGetWorkSpaceSize(miopen_context, 
                                                        filt_desc,
                                                        in_desc,
                                                        conv_desc,
                                                        out_desc)

#Find algorithm to be able to run convolution
search_ws = libhip.hipMalloc(ws_size)
requestedalgocount = 1
perfresult = libmiopen.miopenFindConvolutionForwardAlgorithm(miopen_context, 
                                                in_desc, in_data,
                                                filt_desc, filt_data,
                                                conv_desc, 
                                                out_desc, out_data,
                                                requestedalgocount,
                                                search_ws, ws_size.value
                                                )
alpha = 1.0
beta = 0
# Run the particular algorithm
libmiopen.miopenConvolutionForward(miopen_context, alpha,
                                    in_desc, in_data,
                                    filt_desc, filt_data,
                                    conv_desc, convolution_algo,
                                    beta, out_desc, out_data,
                                    search_ws, ws_size.value
                                   )
libhip.hipMemcpyDtoD(out_data_ptr, out_data, out_bytes)
import torch.nn.functional as F
ref_op = F.conv3d(tmp_input, tmp_filt, stride=1, padding='valid')
diff = np.linalg.norm((ref_op.cpu() - tmp_output.cpu())) / (out_n * out_c * out_d * out_h * out_w )

print(diff)
libmiopen.miopenDestroyConvolutionDescriptor(conv_desc)
libmiopen.miopenDestroyTensorDescriptor(in_desc)
libmiopen.miopenDestroyTensorDescriptor(out_desc)
libmiopen.miopenDestroyTensorDescriptor(filt_desc)
libmiopen.miopenDestroy(miopen_context)