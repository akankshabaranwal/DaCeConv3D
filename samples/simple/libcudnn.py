import sys
import ctypes

# Most functions taken from: https://github.com/hannes-brt/cudnn-python-wrappers/blob/master/libcudnn.py
# https://github.com/hannes-brt/cudnn-python-wrappers/
# Not all functions from the original source code have been implemented.
# TODO: Fix the data types of alpha beta, hardcoded it to float for now. 
# TODO: Write the cuda free function calls

_libcudnn_libname_list = ['libcudnn.so', 'libcudnn.so.8', 'libcudnn.so.8.4.1']
_libcudnn = None
for _libcudnn_libname in _libcudnn_libname_list:
    try:
        _libcudnn = ctypes.cdll.LoadLibrary(_libcudnn_libname)
    except OSError:
        pass
    else:
        break
if _libcudnn is None:
    raise OSError('cuDNN library not found')

# cuDNN error
_libcudnn.cudnnGetErrorString.restype = ctypes.c_char_p
_libcudnn.cudnnGetErrorString.argtypes = [ctypes.c_int]
class cudnnError(Exception):
    def __init__(self, status):
        self.status = status
    def __str__(self):
        error = _libcudnn.cudnnGetErrorString(self.status)
        return '%s' % (error)

# Data layout specification
# cudnnTensorFormat_t is an enumerated type used by
cudnnTensorFormat = {
     'CUDNN_TENSOR_NCHW': 0,       # This tensor format specifies that the data
                                   # is laid out in the following order: image,
                                   # features map, rows, columns. The strides
                                   # are implicitly defined in such a way that
                                   # the data are contiguous in memory with no
                                   # padding between images, feature maps,
                                   # rows, and columns; the columns are the
                                   # inner dimension and the images are the
                                   # outermost dimension.
     'CUDNN_TENSOR_NHWC': 1,       # This tensor format specifies that the data
                                   # is laid out in the following order: image,
                                   # rows, columns, features maps. The strides
                                   # are implicitly defined in such a way that
                                   # the data are contiguous in memory with no
                                   # padding between images, rows, columns, and
                                   # features maps; the feature maps are the
                                   # inner dimension and the images are the
                                   # outermost dimension.
     'CUDNN_TENSOR_NCHW_VECT_C': 2 # This tensor format specifies that the data
                                   # is laid out in the following order: batch
                                   # size, feature maps, rows, columns. However,
                                   # each element of the tensor is a vector of
                                   # multiple feature maps. The length of the
                                   # vector is carried by the data type of the
                                   # tensor. The strides are implicitly defined
                                   # in such a way that the data are contiguous
                                   # in memory with no padding between images,
                                   # feature maps, rows, and columns; the
                                   # columns are the inner dimension and the
                                   # images are the outermost dimension. This
                                   # format is only supported with tensor data
                                   # type CUDNN_DATA_INT8x4.
}

# Data type
# cudnnDataType_t is an enumerated type indicating the data type to which a tensor
# descriptor or filter descriptor refers.
cudnnDataType = {
    'CUDNN_DATA_FLOAT': 0,  # The data is 32-bit single-precision floating point
                            # ( float ).
    'CUDNN_DATA_DOUBLE': 1, # The data is 64-bit double-precision floating point
                            # ( double ).
    'CUDNN_DATA_HALF': 2,   # The data is 16-bit half-precision floating point
                            # ( half ).
    'CUDNN_DATA_INT8': 3,   # The data is 8-bit signed integer.
    'CUDNN_DATA_INT32': 4,  # The data is 32-bit signed integer.
    'CUDNN_DATA_INT8x4': 5  # The data is 32-bit element composed of 4 8-bit
                            # signed integer. This data type is only supported
                            # with tensor format CUDNN_TENSOR_NCHW_VECT_C.
}

# cudnnConvolutionMode_t is an enumerated type used by
# cudnnSetConvolutionDescriptor() to configure a convolution descriptor. The
# filter used for the convolution can be applied in two different ways, corresponding
# mathematically to a convolution or to a cross-correlation. (A cross-correlation is
# equivalent to a convolution with its filter rotated by 180 degrees.)
cudnnConvolutionMode = {
    'CUDNN_CONVOLUTION': 0, # In this mode, a convolution operation will be done
                            # when applying the filter to the images.
    'CUDNN_CROSS_CORRELATION': 1 # In this mode, a cross-correlation operation will
                            # be done when applying the filter to the images.
}

# cudnnConvolutionFwdAlgo_t is an enumerated type that exposes the different algorithm
# available to execute the forward convolution operation.
cudnnConvolutionFwdAlgo = {
    'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM': 0, # This algorithm expresses the convolution
                        # as a matrix product without actually explicitly forming the matrix
                        # that holds the input tensor data.
    'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM': 1, # This algorithm expresses the convolution
                        # as a matrix product without actually explicitly forming the matrix
                        # that holds the input tensor data, but still needs some memory
                        # workspace to precompute some indices in order to facilitate the
                        # implicit construction of the matrix that holds the input tensor data.
    'CUDNN_CONVOLUTION_FWD_ALGO_GEMM': 2, # This algorithm expresses the convolution as an
                        # explicit matrix product. A significant memory workspace is needed to
                        # store the matrix that holds the input tensor data.
    'CUDNN_CONVOLUTION_FWD_ALGO_DIRECT': 3, # This algorithm expresses the convolution as a
                        # direct convolution (e.g without implicitly or explicitly doing a
                        # matrix multiplication).
    'CUDNN_CONVOLUTION_FWD_ALGO_FFT': 4,
    'CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING': 5,
    'CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD': 6,
    'CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED': 7,
    'CUDNN_CONVOLUTION_FWD_ALGO_COUNT': 8
}

def cudnnCheckStatus(status):
    """
    Raise cuDNN exception
    Raise an exception corresponding to the specified cuDNN error code.
    Parameters
    ----------
    status : int
        cuDNN error code
    """

    if status != 0:
        raise cudnnError(status)



# Helper functions

_libcudnn.cudnnGetVersion.restype = ctypes.c_size_t
_libcudnn.cudnnGetVersion.argtypes = []
def cudnnGetVersion():
    """
    Get cuDNN Version.
    """
    return _libcudnn.cudnnGetVersion()

_libcudnn.cudnnCreate.restype = int
_libcudnn.cudnnCreate.argtypes = [ctypes.c_void_p]

def cudnnCreate():
    """
    Initialize cuDNN.
    Initializes cuDNN and returns a handle to the cuDNN context.
    Returns
    -------
    handle : cudnnHandle
        cuDNN context
    """
    handle = ctypes.c_void_p()
    status = _libcudnn.cudnnCreate(ctypes.byref(handle))
    cudnnCheckStatus(status)
    return handle.value


_libcudnn.cudnnDestroy.restype = int
_libcudnn.cudnnDestroy.argtypes = [ctypes.c_void_p]
def cudnnDestroy(handle):
    """
    Release cuDNN resources.
    Release hardware resources used by cuDNN.
    Parameters
    ----------
    handle : cudnnHandle
        cuDNN context.
    """
    status = _libcudnn.cudnnDestroy(ctypes.c_void_p(handle))
    cudnnCheckStatus(status)

_libcudnn.cudnnCreateTensorDescriptor.restype = int
_libcudnn.cudnnCreateTensorDescriptor.argtypes = [ctypes.c_void_p]
def cudnnCreateTensorDescriptor():
    """
    Create a Tensor descriptor object.
    Allocates a cudnnTensorDescriptor_t structure and returns a pointer to it.
    Returns
    -------
    tensor_descriptor : int
        Tensor descriptor.
    """

    tensor = ctypes.c_void_p()
    status = _libcudnn.cudnnCreateTensorDescriptor(ctypes.byref(tensor))
    cudnnCheckStatus(status)
    return tensor.value


_libcudnn.cudnnSetTensorNdDescriptor.restype = int
_libcudnn.cudnnSetTensorNdDescriptor.argtypes = [ctypes.c_void_p, # Descriptor
                                                ctypes.c_int, # Datatype
                                                ctypes.c_int,  # Dimension
                                                ctypes.POINTER(ctypes.c_int), #dims.data()
                                                ctypes.POINTER(ctypes.c_int) #strides.data()
                                                ]
def cudnnSetTensorNdDescriptor(tensorDesc, dataType, tensor_dim, dims, strides):
    """
    """
    dims_arr = (ctypes.c_int * len(dims))(*dims) # Convert list to array
    strides_arr = (ctypes.c_int * len(strides))(*strides)
    status = _libcudnn.cudnnSetTensorNdDescriptor(tensorDesc, 
                                                dataType, 
                                                tensor_dim, 
                                                dims_arr,
                                                strides_arr)                             
    cudnnCheckStatus(status)

_libcudnn.cudnnCreateFilterDescriptor.restype = int
_libcudnn.cudnnCreateFilterDescriptor.argtypes = [ctypes.c_void_p]
def cudnnCreateFilterDescriptor():
    """"
    Create a filter descriptor.
    This function creates a filter descriptor object by allocating the memory needed
    to hold its opaque structure.
    Parameters
    ----------
    Returns
    -------
    wDesc : cudnnFilterDescriptor
        Handle to a newly allocated filter descriptor.
    """

    wDesc = ctypes.c_void_p()
    status = _libcudnn.cudnnCreateFilterDescriptor(ctypes.byref(wDesc))
    cudnnCheckStatus(status)

    return wDesc.value


_libcudnn.cudnnSetFilterNdDescriptor.restype = int
_libcudnn.cudnnSetFilterNdDescriptor.argtypes = [ctypes.c_void_p, #desc
                                                ctypes.c_int, #datatype
                                                ctypes.c_int, #layout
                                                ctypes.c_int, #tensordim
                                                ctypes.POINTER(ctypes.c_int)] #filtdim
def cudnnSetFilterNdDescriptor(wDesc, dataType, format, tensor_dim, filterdims):
    filterdims_arr = (ctypes.c_int * len(filterdims))(*filterdims) # Convert list to array
    status = _libcudnn.cudnnSetFilterNdDescriptor(wDesc, 
                                                dataType, 
                                                format, 
                                                tensor_dim,
                                                filterdims_arr)
    cudnnCheckStatus(status)


_libcudnn.cudnnCreateConvolutionDescriptor.restype = int
_libcudnn.cudnnCreateConvolutionDescriptor.argtypes = [ctypes.c_void_p]
def cudnnCreateConvolutionDescriptor():
    """"
    Create a convolution descriptor.
    This function creates a convolution descriptor object by allocating the memory needed to
    hold its opaque structure.
    Returns
    -------
    convDesc : cudnnConvolutionDescriptor
        Handle to newly allocated convolution descriptor.
    """

    convDesc = ctypes.c_void_p()

    status = _libcudnn.cudnnCreateConvolutionDescriptor(ctypes.byref(convDesc))
    cudnnCheckStatus(status)

    return convDesc.value


_libcudnn.cudnnSetConvolutionNdDescriptor.restype = int
_libcudnn.cudnnSetConvolutionNdDescriptor.argtypes = [ctypes.c_void_p, # convDesc
                                                      ctypes.c_int, # arrayLength
                                                      ctypes.POINTER(ctypes.c_int), # padA[]
                                                      ctypes.POINTER(ctypes.c_int), # filterStrideA[]
                                                      ctypes.POINTER(ctypes.c_int), # dilationA[]
                                                      ctypes.c_int, # mode
                                                      ctypes.c_int] # dataType
def cudnnSetConvolutionNdDescriptor(convDesc, dim, padA, filterStrideA, dilationA, mode, dataType):
    padA_arr = (ctypes.c_int * len(padA))(*padA) # Convert list to array
    filterStrideA_arr = (ctypes.c_int * len(filterStrideA))(*filterStrideA) # Convert list to array
    dilationA_arr = (ctypes.c_int * len(dilationA))(*dilationA) # Convert list to array

    status = _libcudnn.cudnnSetConvolutionNdDescriptor(convDesc,
                                                       dim,
                                                       padA_arr,
                                                       filterStrideA_arr,
                                                       dilationA_arr,
                                                       mode,
                                                       dataType)
    cudnnCheckStatus(status)


import numpy as np
_libcudnn.cudnnGetConvolutionNdForwardOutputDim.restype = int
_libcudnn.cudnnGetConvolutionNdForwardOutputDim.argtypes = [ctypes.c_void_p, 
                                                            ctypes.c_void_p, 
                                                            ctypes.c_void_p,
                                                            ctypes.c_int,
                                                            ctypes.POINTER(ctypes.c_int)]
def cudnnGetConvolutionNdForwardOutputDim(convDesc, inputTensorDesc, wDesc, tensor_dim, outdims):
    """"
    """    
    outdims_arr = (ctypes.c_int * len(outdims))(*outdims) # Convert list to array

    status = _libcudnn.cudnnGetConvolutionNdForwardOutputDim(convDesc, 
                                                            inputTensorDesc,
                                                            wDesc,
                                                            tensor_dim,
                                                            outdims_arr)
    outdims = np.ctypeslib.as_array(outdims_arr)
    cudnnCheckStatus(status)
    return outdims

class cudnnConvolutionFwdAlgoPerf(ctypes.Structure):
    _fields_ = [("algo", ctypes.c_int),
                ("status", ctypes.c_int),
                ("time", ctypes.c_float),
                ("memory", ctypes.c_size_t)]

    def __str__(self):
        return '(algo=%d, status=%d, time=%f, memory=%d)' % (self.algo,
                                                             self.status,
                                                             self.time,
                                                             self.memory)
    def __repr__(self):
        return self.__str__()
        

_libcudnn.cudnnGetConvolutionForwardWorkspaceSize.restype = int
_libcudnn.cudnnGetConvolutionForwardWorkspaceSize.argtypes = [ctypes.c_void_p,
                                                              ctypes.c_void_p,
                                                              ctypes.c_void_p,
                                                              ctypes.c_void_p,
                                                              ctypes.c_void_p,
                                                              ctypes.c_int]
def cudnnGetConvolutionForwardWorkspaceSize(handle, srcDesc, wDesc,
                                            convDesc, destDesc, algo):
    """"
    This function returns the amount of GPU memory workspace the user needs
    to allocate to be able to call cudnnConvolutionForward with the specified algorithm.
    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    srcDesc : cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.
    wDesc : cudnnFilterDescriptor
        Handle to a previously initialized filter descriptor.
    convDesc : cudnnConvolutionDescriptor
        Previously initialized convolution descriptor.
    destDesc : cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.
    algo : cudnnConvolutionFwdAlgo
        Enumerant that specifies the chosen convolution algorithm.
    Returns
    -------
    sizeInBytes: c_size_t
        Amount of GPU memory needed as workspace to be able to execute a
        forward convolution with the sepcified algo.
    """
    sizeInBytes = ctypes.c_size_t()
    conv_algo = ctypes.c_int(algo)

    status = _libcudnn.cudnnGetConvolutionForwardWorkspaceSize(handle, 
                                                            srcDesc, 
                                                            wDesc,
                                                            convDesc, 
                                                            destDesc,
                                                            conv_algo,
                                                            ctypes.byref(sizeInBytes))
    cudnnCheckStatus(status)
    return sizeInBytes


_libcudnn.cudnnConvolutionForward.restype = int
_libcudnn.cudnnConvolutionForward.argtypes = [ctypes.c_void_p, 
                                            ctypes.c_void_p, 
                                            ctypes.c_void_p,
                                            ctypes.c_void_p, 
                                            ctypes.c_void_p, 
                                            ctypes.c_void_p,
                                            ctypes.c_void_p, 
                                            ctypes.c_int,
                                            ctypes.c_void_p, 
                                            ctypes.c_size_t,
                                            ctypes.c_void_p, 
                                            ctypes.c_void_p,
                                            ctypes.c_void_p]

def cudnnConvolutionForward(handle, alpha, srcDesc, srcData, wDesc, w,
                            convDesc, algo, workspace, workSpaceSizeInBytes, beta,
                            destDesc, destData):
    """
    """
    alphaRef = ctypes.byref(ctypes.c_float(alpha))
    betaRef = ctypes.byref(ctypes.c_float(beta))

    status = _libcudnn.cudnnConvolutionForward(handle, 
                                            alphaRef, 
                                            srcDesc, 
                                            srcData,
                                            wDesc, 
                                            w,
                                            convDesc, 
                                            algo, 
                                            workspace,
                                            ctypes.c_size_t(workSpaceSizeInBytes),
                                            betaRef, 
                                            destDesc, 
                                            destData)
    cudnnCheckStatus(status)


_libcudnn.cudnnDestroyTensorDescriptor.restype = int
_libcudnn.cudnnDestroyTensorDescriptor.argtypes = [ctypes.c_void_p]
def cudnnDestroyTensorDescriptor(tensorDesc):
    """"
    Destroy a Tensor descriptor.
    This function destroys a previously created Tensor descriptor object.
    Parameters
    ----------
    tensorDesc : cudnnTensorDescriptor
        Previously allocated Tensor descriptor object.
    """

    status = _libcudnn.cudnnDestroyTensorDescriptor(tensorDesc)
    cudnnCheckStatus(status)

_libcudnn.cudnnDestroyFilterDescriptor.restype = int
_libcudnn.cudnnDestroyFilterDescriptor.argtypes = [ctypes.c_void_p]
def cudnnDestroyFilterDescriptor(wDesc):
    """"
    Destroy filter descriptor.
    This function destroys a previously created Tensor4D descriptor object.
    Parameters
    ----------
    wDesc : cudnnFilterDescriptor
    """

    status = _libcudnn.cudnnDestroyFilterDescriptor(wDesc)
    cudnnCheckStatus(status)


_libcudnn.cudnnDestroyConvolutionDescriptor.restype = int
_libcudnn.cudnnDestroyConvolutionDescriptor.argtypes = [ctypes.c_void_p]
def cudnnDestroyConvolutionDescriptor(convDesc):
    """"
    Destroy a convolution descriptor.
    This function destroys a previously created convolution descriptor object.
    Parameters
    ----------
    convDesc : int
        Previously created convolution descriptor.
    """

    status = _libcudnn.cudnnDestroyConvolutionDescriptor(convDesc)
    cudnnCheckStatus(status)


_libcudnn.cudnnDestroy.restype = int
_libcudnn.cudnnDestroy.argtypes = [ctypes.c_void_p]
def cudnnDestroy(handle):
    """
    Release cuDNN resources.
    Release hardware resources used by cuDNN.
    Parameters
    ----------
    handle : cudnnHandle
        cuDNN context.
    """

    status = _libcudnn.cudnnDestroy(ctypes.c_void_p(handle))
    cudnnCheckStatus(status)