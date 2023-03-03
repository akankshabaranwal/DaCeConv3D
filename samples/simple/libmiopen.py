import sys
import ctypes
import numpy as np

_libmiopen_libname_list = ['libMIOpen.so', 'libMIOpen.so.1', 'libMIOpen.so.1.0']
_libmiopen = None

for _libmiopen_libname in _libmiopen_libname_list:
    try:
        _libmiopen = ctypes.cdll.LoadLibrary(_libmiopen_libname)
    except OSError:
        pass
    else:
        break
if _libmiopen is None:
    raise OSError('miopen library not found')

# cuDNN error
_libmiopen.miopenGetErrorString.restype = ctypes.c_char_p
_libmiopen.miopenGetErrorString.argtypes = [ctypes.c_int]
class miopenError(Exception):
    def __init__(self, status):
        self.status = status
    def __str__(self):
        error = _libmiopen.miopenGetErrorString(self.status)
        return '%s' % (error)


def miopenCheckStatus(status):
    """
    Raise miopen exception
    Raise an exception corresponding to the specified miopen error code.
    Parameters
    ----------
    status : int
        miopen error code
    """

    if status != 0:
        raise miopenError(status)
    
def miopenCreate():
    """
    Initialize miopen.
    Initializes miopen and returns a handle to the miopen context.
    Returns
    -------
    handle : miopenHandle
        miopen context
    """
    handle = ctypes.c_void_p()
    status = _libmiopen.miopenCreate(ctypes.byref(handle))
    miopenCheckStatus(status)
    return handle.value

miopenConvolutionMode = {    
    'miopenConvolution': 0,
    'miopenTranspose': 1,
    'miopenGroupConv': 2,
    'miopenDepthwise': 3
}

miopenConvFwdAlgo = {
    'miopenConvolutionFwdAlgoGEMM': 0, 
    'miopenConvolutionFwdAlgoDirect': 1, 
    'miopenConvolutionFwdAlgoFFT': 2, 
    'miopenConvolutionFwdAlgoWinograd': 3, 
    'miopenConvolutionFwdAlgoImplicitGEMM': 4
}

miopenDatatype = {
    'miopenHalf' : 0,
    'miopenFloat': 1,
    'miopenInt32': 2,
    'miopenInt8': 3,
    'miopenInt8x4': 4,
    'miopenBFloat16': 5
}


_libmiopen.miopenCreateTensorDescriptor.restype = int
_libmiopen.miopenCreateTensorDescriptor.argtypes = [ctypes.c_void_p]
def miopenCreateTensorDescriptor():
    """
    Create a Tensor descriptor object.
    Allocates a miopenTensorDescriptor_t structure and returns a pointer to it.
    Returns
    -------
    tensor_descriptor : int
        Tensor descriptor.
    """

    tensor = ctypes.c_void_p()
    status = _libmiopen.miopenCreateTensorDescriptor(ctypes.byref(tensor))
    miopenCheckStatus(status)
    return tensor.value


_libmiopen.miopenSetTensorDescriptor.restype = int
_libmiopen.miopenSetTensorDescriptor.argtypes = [ctypes.c_void_p, # Descriptor
                                                ctypes.c_int, # Datatype
                                                ctypes.c_int,  # Dimension
                                                ctypes.POINTER(ctypes.c_int), #dims.data()
                                                ctypes.POINTER(ctypes.c_int) #strides.data()
                                                ]
def miopenSetTensorDescriptor(tensorDesc, dataType, tensor_dim, dims, strides):
    """
    """
    dims_arr = (ctypes.c_int * len(dims))(*dims) # Convert list to array
    strides_arr = (ctypes.c_int * len(strides))(*strides)
    status = _libmiopen.miopenSetTensorDescriptor(tensorDesc, 
                                                dataType, 
                                                tensor_dim, 
                                                dims_arr,
                                                strides_arr)                             
    miopenCheckStatus(status)


_libmiopen.miopenCreateConvolutionDescriptor.restype = int
_libmiopen.miopenCreateConvolutionDescriptor.argtypes = [ctypes.c_void_p]
def miopenCreateConvolutionDescriptor():
    """"
    Create a convolution descriptor.
    This function creates a convolution descriptor object by allocating the memory needed to
    hold its opaque structure.
    Returns
    -------
    convDesc : miopenCreateConvolutionDescriptor
        Handle to newly allocated convolution descriptor.
    """

    convDesc = ctypes.c_void_p()

    status = _libmiopen.miopenCreateConvolutionDescriptor(ctypes.byref(convDesc))
    miopenCheckStatus(status)
    return convDesc.value


_libmiopen.miopenInitConvolutionNdDescriptor.restype = int
_libmiopen.miopenInitConvolutionNdDescriptor.argtypes = [ctypes.c_void_p, # convDesc
                                                      ctypes.c_int, # arrayLength
                                                      ctypes.POINTER(ctypes.c_int), # padA[]
                                                      ctypes.POINTER(ctypes.c_int), # filterStrideA[]
                                                      ctypes.POINTER(ctypes.c_int), # dilationA[]
                                                      ctypes.c_int, # mode
                                                      ctypes.c_int] # dataType
def miopenInitConvolutionNdDescriptor(convDesc, dim, padA, filterStrideA, dilationA, mode, dataType):
    padA_arr = (ctypes.c_int * len(padA))(*padA) # Convert list to array
    filterStrideA_arr = (ctypes.c_int * len(filterStrideA))(*filterStrideA) # Convert list to array
    dilationA_arr = (ctypes.c_int * len(dilationA))(*dilationA) # Convert list to array

    status = _libmiopen.miopenInitConvolutionNdDescriptor(convDesc,
                                                       dim,
                                                       padA_arr,
                                                       filterStrideA_arr,
                                                       dilationA_arr,
                                                       mode,
                                                       dataType)
    miopenCheckStatus(status)


_libmiopen.miopenGetConvolutionNdForwardOutputDim.restype = int
_libmiopen.miopenGetConvolutionNdForwardOutputDim.argtypes = [ctypes.c_void_p, 
                                                            ctypes.c_void_p, 
                                                            ctypes.c_void_p,
                                                            ctypes.POINTER(ctypes.c_int),
                                                            ctypes.POINTER(ctypes.c_int)]
def miopenGetConvolutionNdForwardOutputDim(convDesc, inputTensorDesc, wDesc, tensor_dim, outdims):
    """"
    """    
    outdims_arr = (ctypes.c_int * len(outdims))(*outdims) # Convert list to array
    dim_arr = (ctypes.c_int)(*tensor_dim) # Convert list to array
    status = _libmiopen.miopenGetConvolutionNdForwardOutputDim(convDesc, 
                                                            inputTensorDesc,
                                                            wDesc,
                                                            dim_arr,
                                                            outdims_arr)
    outdims = np.ctypeslib.as_array(outdims_arr)
    miopenCheckStatus(status)
    return outdims