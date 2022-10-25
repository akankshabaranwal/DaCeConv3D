import sys
import ctypes

# Most functions taken from: https://github.com/hannes-brt/cudnn-python-wrappers/blob/master/libcudnn.py
# Not all functions from the original source code have been implemented.

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
