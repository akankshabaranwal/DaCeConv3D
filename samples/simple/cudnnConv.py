import libcudnn, ctypes
import numpy as np
from pycuda import gpuarray
import pycuda.driver as drv


def cudnn_init(pad, stride, dil, layout):## cudnn fixed parameters init
    cudnn_context = libcudnn.cudnnCreate()
    if(layout=='NCDHW'):
        tensor_format = libcudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW']
    else:
        tensor_format = libcudnn.cudnnTensorFormat['CUDNN_TENSOR_NHWC']
    data_type = libcudnn.cudnnDataType['CUDNN_DATA_FLOAT']
    tensor_dim = 5
    conv_dim = tensor_dim-2
    convolution_mode = libcudnn.cudnnConvolutionMode['CUDNN_CROSS_CORRELATION']
    convolution_algo = libcudnn.cudnnConvolutionFwdAlgo['CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM']
    ''' Available algorithms for 3d convolution cudnn are: 
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING
    '''
    alpha = 1.0
    beta = 0
    c_int_p = ctypes.POINTER(ctypes.c_int)
    outdimsinit = [0, 0, 0, 0, 0]
    # cudnn convolution descriptor
    conv_desc = libcudnn.cudnnCreateConvolutionDescriptor()
    convpad = [pad, pad, pad]
    filtstr = [stride, stride, stride]
    convdil = [dil, dil, dil]
    libcudnn.cudnnSetConvolutionNdDescriptor(conv_desc, conv_dim, convpad, filtstr, convdil, convolution_mode, data_type)
    return conv_desc, cudnn_context, tensor_format, convolution_mode, convolution_algo, alpha, beta, c_int_p, outdimsinit, data_type, tensor_dim, conv_dim


def cudnnsetlayerdesc(cudnn_context, outdimsinit, conv_desc, convolution_algo, d_input,  d_kernel, d_output, batchsize, kdim, inchannels, indepth, inheight, inwidth, outchannels, data_type, tensor_dim, tensor_format):
    dims = [batchsize, inchannels, indepth, inheight, inwidth]
    strides = [inchannels*indepth*inheight*inwidth, indepth*inheight*inwidth, inheight*inwidth, inwidth, 1]
    in_desc = libcudnn.cudnnCreateTensorDescriptor()
    libcudnn.cudnnSetTensorNdDescriptorEx(in_desc, tensor_format, data_type, tensor_dim, dims)
   
    # TODO: Maybe simplify this conversion to gpuarray ??
    cudnn_input = d_input.detach().clone()
    cudnn_kernel = d_kernel.detach().clone()
    cudnn_output = d_output.detach().clone()
    in_data_g = gpuarray.to_gpu(cudnn_input.cpu().numpy().astype(np.float32))
    in_data = ctypes.c_void_p(int(in_data_g.gpudata))

    # cudnn filter input
    filt_desc = libcudnn.cudnnCreateFilterDescriptor()
    filt_dims = [outchannels, inchannels, kdim, kdim, kdim]
    libcudnn.cudnnSetFilterNdDescriptor(filt_desc, data_type, tensor_format, tensor_dim, filt_dims)
    filt_data_g = gpuarray.to_gpu(cudnn_kernel.cpu().numpy().astype(np.float32))                                    
    filt_data = ctypes.c_void_p(int(filt_data_g.gpudata))

    # cudnn output
    outdims = libcudnn.cudnnGetConvolutionNdForwardOutputDim(conv_desc, in_desc, filt_desc, tensor_dim, outdimsinit)
    out_n, out_c, out_d, out_h, out_w = outdims[0], outdims[1], outdims[2], outdims[3], outdims[4]
    outstrides = [ out_c*out_d*out_h*out_w, out_d*out_h*out_w, out_h*out_w, out_w, 1]
    out_desc = libcudnn.cudnnCreateTensorDescriptor()
    libcudnn.cudnnSetTensorNdDescriptorEx(out_desc,  tensor_format, data_type, tensor_dim, outdims)

    out_data_g = gpuarray.to_gpu(cudnn_output.cpu().numpy().astype(np.float32))                               
    out_data = ctypes.c_void_p(int(out_data_g.gpudata))
    
    # Compute cudnn workspace size
    ws_size = libcudnn.cudnnGetConvolutionForwardWorkspaceSize(cudnn_context, in_desc, filt_desc, conv_desc, out_desc, convolution_algo)
    ws_ptr  = drv.mem_alloc(ws_size.value) if ws_size.value > 0 else 0
    ws_data = ctypes.c_void_p(int(ws_ptr))
    
    return cudnn_input, cudnn_kernel, cudnn_output, in_desc, in_data, in_data_g, out_desc, out_data, out_data_g, outdims, filt_desc, filt_data, filt_data_g, ws_ptr, ws_data, ws_size


def destroydescinoutfilt(in_desc, out_desc, filt_desc, ws_ptr):
    ws_ptr = None
    libcudnn.cudnnDestroyTensorDescriptor(in_desc)
    libcudnn.cudnnDestroyTensorDescriptor(out_desc)
    libcudnn.cudnnDestroyFilterDescriptor(filt_desc)
    return in_desc, out_desc, filt_desc, ws_ptr