import sys
import ctypes

# https://github.com/jatinx/PyHIP/blob/main/pyhip/hip.py 
# Use this for hipmalloc etc. 

_libhip_libname_list = ['libamdhip64.so', 'libamdhip64.so.5', 'libamdhip64.so.5.4.22804']
_libhip = None
for _libhip_libname in _libhip_libname_list:
    try:
        _libhip = ctypes.cdll.LoadLibrary(_libhip_libname)
    except OSError:
        pass
    else:
        break

    if _libhip is None:
        raise OSError('hip library not found')

# Generic hip error


class hipError(Exception):
    """hip error"""
    pass


class hipErrorInvalidValue(hipError):
    __doc__ = _libhip.hipGetErrorString(1)
    pass


class hipErrorOutOfMemory(hipError):
    __doc__ = _libhip.hipGetErrorString(2)
    pass


class hipErrorNotInitialized(hipError):
    __doc__ = _libhip.hipGetErrorString(3)
    pass


class hipErrorDeinitialized(hipError):
    __doc__ = _libhip.hipGetErrorString(4)
    pass


class hipErrorProfilerDisabled(hipError):
    __doc__ = _libhip.hipGetErrorString(5)
    pass


class hipErrorProfilerNotInitialized(hipError):
    __doc__ = _libhip.hipGetErrorString(6)
    pass


class hipErrorProfilerAlreadyStarted(hipError):
    __doc__ = _libhip.hipGetErrorString(7)
    pass


class hipErrorProfilerAlreadyStopped(hipError):
    __doc__ = _libhip.hipGetErrorString(8)
    pass


class hipErrorInvalidConfiguration(hipError):
    __doc__ = _libhip.hipGetErrorString(9)
    pass


class hipErrorInvalidSymbol(hipError):
    __doc__ = _libhip.hipGetErrorString(13)
    pass


class hipErrorInvalidDevicePointer(hipError):
    __doc__ = _libhip.hipGetErrorString(17)
    pass


class hipErrorInvalidMemcpyDirection(hipError):
    __doc__ = _libhip.hipGetErrorString(21)
    pass


class hipErrorInsufficientDriver(hipError):
    __doc__ = _libhip.hipGetErrorString(35)
    pass


class hipErrorMissingConfiguration(hipError):
    __doc__ = _libhip.hipGetErrorString(52)
    pass


class hipErrorPriorLaunchFailure(hipError):
    __doc__ = _libhip.hipGetErrorString(53)
    pass


class hipErrorInvalidDeviceFunction(hipError):
    __doc__ = _libhip.hipGetErrorString(98)
    pass


class hipErrorNoDevice(hipError):
    __doc__ = _libhip.hipGetErrorString(100)
    pass


class hipErrorInvalidDevice(hipError):
    __doc__ = _libhip.hipGetErrorString(101)
    pass


class hipErrorInvalidImage(hipError):
    __doc__ = _libhip.hipGetErrorString(200)
    pass


class hipErrorInvalidContext(hipError):
    __doc__ = _libhip.hipGetErrorString(201)
    pass


class hipErrorContextAlreadyCurrent(hipError):
    __doc__ = _libhip.hipGetErrorString(202)
    pass


class hipErrorMapFailed(hipError):
    __doc__ = _libhip.hipGetErrorString(205)
    pass


class hipErrorUnmapFailed(hipError):
    __doc__ = _libhip.hipGetErrorString(206)
    pass


class hipErrorArrayIsMapped(hipError):
    __doc__ = _libhip.hipGetErrorString(207)
    pass


class hipErrorAlreadyMapped(hipError):
    __doc__ = _libhip.hipGetErrorString(208)
    pass


class hipErrorNoBinaryForGpu(hipError):
    __doc__ = _libhip.hipGetErrorString(209)
    pass


class hipErrorAlreadyAcquired(hipError):
    __doc__ = _libhip.hipGetErrorString(210)
    pass


class hipErrorNotMapped(hipError):
    __doc__ = _libhip.hipGetErrorString(211)
    pass


class hipErrorNotMappedAsArray(hipError):
    __doc__ = _libhip.hipGetErrorString(212)
    pass


class hipErrorNotMappedAsPointer(hipError):
    __doc__ = _libhip.hipGetErrorString(213)
    pass


class hipErrorECCNotCorrectable(hipError):
    __doc__ = _libhip.hipGetErrorString(214)
    pass


class hipErrorUnsupportedLimit(hipError):
    __doc__ = _libhip.hipGetErrorString(215)
    pass


class hipErrorContextAlreadyInUse(hipError):
    __doc__ = _libhip.hipGetErrorString(216)
    pass


class hipErrorPeerAccessUnsupported(hipError):
    __doc__ = _libhip.hipGetErrorString(217)
    pass


class hipErrorInvalidKernelFile(hipError):
    __doc__ = _libhip.hipGetErrorString(218)
    pass


class hipErrorInvalidGraphicsContext(hipError):
    __doc__ = _libhip.hipGetErrorString(219)
    pass


class hipErrorInvalidSource(hipError):
    __doc__ = _libhip.hipGetErrorString(300)
    pass


class hipErrorFileNotFound(hipError):
    __doc__ = _libhip.hipGetErrorString(301)
    pass


class hipErrorSharedObjectSymbolNotFound(hipError):
    __doc__ = _libhip.hipGetErrorString(302)
    pass


class hipErrorSharedObjectInitFailed(hipError):
    __doc__ = _libhip.hipGetErrorString(303)
    pass


class hipErrorOperatingSystem(hipError):
    __doc__ = _libhip.hipGetErrorString(304)
    pass


class hipErrorInvalidHandle(hipError):
    __doc__ = _libhip.hipGetErrorString(400)
    pass


class hipErrorNotFound(hipError):
    __doc__ = _libhip.hipGetErrorString(500)
    pass


class hipErrorNotReady(hipError):
    __doc__ = _libhip.hipGetErrorString(600)
    pass


class hipErrorIllegalAddress(hipError):
    __doc__ = _libhip.hipGetErrorString(700)
    pass


class hipErrorLaunchOutOfResources(hipError):
    __doc__ = _libhip.hipGetErrorString(701)
    pass


class hipErrorLaunchTimeOut(hipError):
    __doc__ = _libhip.hipGetErrorString(702)
    pass


class hipErrorPeerAccessAlreadyEnabled(hipError):
    __doc__ = _libhip.hipGetErrorString(704)
    pass


class hipErrorPeerAccessNotEnabled(hipError):
    __doc__ = _libhip.hipGetErrorString(705)
    pass


class hipErrorSetOnActiveProcess(hipError):
    __doc__ = _libhip.hipGetErrorString(708)
    pass


class hipErrorAssert(hipError):
    __doc__ = _libhip.hipGetErrorString(710)
    pass


class hipErrorHostMemoryAlreadyRegistered(hipError):
    __doc__ = _libhip.hipGetErrorString(712)
    pass


class hipErrorHostMemoryNotRegistered(hipError):
    __doc__ = _libhip.hipGetErrorString(713)
    pass


class hipErrorLaunchFailure(hipError):
    __doc__ = _libhip.hipGetErrorString(719)
    pass


class hipErrorCooperativeLaunchTooLarge(hipError):
    __doc__ = _libhip.hipGetErrorString(720)
    pass


class hipErrorNotSupported(hipError):
    __doc__ = _libhip.hipGetErrorString(801)
    pass


class hipErrorUnknown(hipError):
    __doc__ = _libhip.hipGetErrorString(999)
    pass


class hipErrorRuntimeMemory(hipError):
    __doc__ = _libhip.hipGetErrorString(1052)
    pass


class hipErrorRuntimeOther(hipError):
    __doc__ = _libhip.hipGetErrorString(1053)
    pass

hipExceptions = {
    1: hipErrorInvalidValue,
    2: hipErrorOutOfMemory,
    3: hipErrorNotInitialized,
    4: hipErrorDeinitialized,
    5: hipErrorProfilerDisabled,
    6: hipErrorProfilerNotInitialized,
    7: hipErrorProfilerAlreadyStarted,
    8: hipErrorProfilerAlreadyStopped,
    9: hipErrorInvalidConfiguration,
    13: hipErrorInvalidSymbol,
    17: hipErrorInvalidDevicePointer,
    21: hipErrorInvalidMemcpyDirection,
    35: hipErrorInsufficientDriver,
    52: hipErrorMissingConfiguration,
    53: hipErrorPriorLaunchFailure,
    98: hipErrorInvalidDeviceFunction,
    100: hipErrorNoDevice,
    101: hipErrorInvalidDevice,
    200: hipErrorInvalidImage,
    201: hipErrorInvalidContext,
    202: hipErrorContextAlreadyCurrent,
    205: hipErrorMapFailed,
    206: hipErrorUnmapFailed,
    207: hipErrorArrayIsMapped,
    208: hipErrorAlreadyMapped,
    209: hipErrorNoBinaryForGpu,
    210: hipErrorAlreadyAcquired,
    211: hipErrorNotMapped,
    212: hipErrorNotMappedAsArray,
    213: hipErrorNotMappedAsPointer,
    214: hipErrorECCNotCorrectable,
    215: hipErrorUnsupportedLimit,
    216: hipErrorContextAlreadyInUse,
    217: hipErrorPeerAccessUnsupported,
    218: hipErrorInvalidKernelFile,
    219: hipErrorInvalidGraphicsContext,
    300: hipErrorInvalidSource,
    301: hipErrorFileNotFound,
    302: hipErrorSharedObjectSymbolNotFound,
    303: hipErrorSharedObjectInitFailed,
    304: hipErrorOperatingSystem,
    400: hipErrorInvalidHandle,
    500: hipErrorNotFound,
    600: hipErrorNotReady,
    700: hipErrorIllegalAddress,
    701: hipErrorLaunchOutOfResources,
    702: hipErrorLaunchTimeOut,
    704: hipErrorPeerAccessAlreadyEnabled,
    705: hipErrorPeerAccessNotEnabled,
    708: hipErrorSetOnActiveProcess,
    710: hipErrorAssert,
    712: hipErrorHostMemoryAlreadyRegistered,
    713: hipErrorHostMemoryNotRegistered,
    719: hipErrorLaunchFailure,
    720: hipErrorCooperativeLaunchTooLarge,
    801: hipErrorNotSupported,
    999: hipErrorUnknown,
    1052: hipErrorRuntimeMemory,
    1053: hipErrorRuntimeOther
}


def hipCheckStatus(status):
    """
    Raise hip exception.
    Raise an exception corresponding to the specified hip runtime error
    code.
    Parameters
    ----------
    status : int
        hip runtime error code.
    See Also
    --------
    hipExceptions
    """

    if status != 0:
        try:
            e = hipExceptions[status]
        except KeyError:
            raise hipError('unknown hip error %s' % status)
        else:
            raise e
        
def POINTER(obj):
    """
    ctype pointer to object
    """
    p = ctypes.POINTER(obj)
    if not isinstance(p.from_param, classmethod):
        def from_param(cls, x):
            if x is None:
                return cls()
            else:
                return x
        p.from_param = classmethod(from_param)
    return p


def hipMalloc(count, ctype=None):
    """
    Allocate device memory.
    Allocate memory on the device associated with the current active
    context.
    Parameters
    ----------
    count : int
        Number of bytes of memory to allocate
    ctype : _ctypes.SimpleType, optional
        ctypes type to cast returned pointer.
    Returns
    -------
    ptr : ctypes pointer
        Pointer to allocated device memory.
    """

    ptr = ctypes.c_void_p()
    status = _libhip.hipMalloc(ctypes.byref(ptr), count)
    hipCheckStatus(status)
    if ctype is not None:
        ptr = ctypes.cast(ptr, ctypes.POINTER(ctype))
    return ptr


_libhip.hipFree.restype = int
_libhip.hipFree.argtypes = [ctypes.c_void_p]

def hipFree(ptr):
    """
    Free device memory.
    Free allocated memory on the device associated with the current active
    context.
    Parameters
    ----------
    ptr : ctypes pointer
        Pointer to allocated device memory.
    """

    status = _libhip.hipFree(ptr)
    hipCheckStatus(status)