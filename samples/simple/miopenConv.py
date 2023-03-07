import libmiopen, libhip, ctypes

def miopen_init(pad, stride, dil):
    miopen_context = libmiopen.miopenCreate()
    print("Created miopen context")

    data_type = libmiopen.miopenDatatype['miopenFloat']
    tensor_dim = 5
    convdim = tensor_dim-2
    convolution_mode = libmiopen.miopenConvolutionMode['miopenConvolution']
    convolution_algo = libmiopen.miopenConvFwdAlgo['miopenConvolutionFwdAlgoDirect']

    alpha = 1.0
    beta = 0
    c_int_p = ctypes.POINTER(ctypes.c_int)
    outdimsinit = [0, 0, 0, 0, 0]
    convpad = [pad, pad, pad]
    filtstr = [stride, stride, stride]
    convdil = [dil, dil, dil]
    conv_desc = libmiopen.miopenCreateConvolutionDescriptor()

    libmiopen.miopenInitConvolutionNdDescriptor(conv_desc, 
                                                convdim,
                                                convpad,
                                                filtstr,
                                                convdil,
                                                convolution_mode,
                                                data_type)
    return conv_desc, miopen_context, convolution_mode, convolution_algo, alpha, beta, c_int_p, outdimsinit, data_type, tensor_dim, convdim


def miopensetlayerdesc(miopen_context, outdimsinit, conv_desc, d_input, d_kernel, batchsize, kdim, inchannels, indepth, inheight, inwidth, outchannels, data_type, tensor_dim):

    #miopen_input = d_input.detach().clone()
    #miopen_kernel = d_kernel.detach().clone()
    
    # miopen kernel
    dims = [batchsize, inchannels, indepth, inheight, inwidth]
    strides = [inchannels*indepth*inheight*inwidth, indepth*inheight*inwidth, inheight*inwidth, inwidth, 1]
    in_desc = libmiopen.miopenCreateTensorDescriptor()
    libmiopen.miopenSetTensorDescriptor(in_desc,
                                        data_type, 
                                        tensor_dim,
                                        dims, 
                                        strides)
    in_bytes = int(batchsize*inchannels*indepth*inheight*inwidth*ctypes.sizeof(ctypes.c_float))
    in_data = libhip.hipMalloc(in_bytes)
    import torch
    test_in_data = torch.zeros(d_input.shape).cuda()
    in_data_ptr = d_input.data_ptr()
    libhip.hipMemcpyDtoD(in_data, in_data_ptr, in_bytes)
    test_in_data_ptr = test_in_data.data_ptr()
    libhip.hipMemcpyDtoD(test_in_data_ptr, in_data, in_bytes)
    import numpy as np
    diff = np.linalg.norm((d_input.cpu() - test_in_data.cpu())) / (batchsize * inchannels * indepth * inheight * inwidth )
    print(f" difference in input is: {diff}")
    # miopen filter 
    filt_dims = [outchannels, inchannels, kdim, kdim, kdim]
    filt_strides = [inchannels*kdim*kdim*kdim, kdim*kdim, kdim, 1]
    filt_desc = libmiopen.miopenCreateTensorDescriptor()
    libmiopen.miopenSetTensorDescriptor(filt_desc,
                                        data_type, 
                                        tensor_dim,
                                        filt_dims, 
                                        filt_strides)
    filt_bytes = int(outchannels*inchannels*kdim*kdim*kdim)
    filt_data = libhip.hipMalloc(filt_bytes)
    filt_data_ptr = d_kernel.data_ptr()
    libhip.hipMemcpyDtoD(filt_data, filt_data_ptr, filt_bytes)

    # miopen output
    outdimsinit = [0, 0, 0, 0, 0]
    tensor_dim = [5]
    outdims = libmiopen.miopenGetConvolutionNdForwardOutputDim(conv_desc, 
                                            in_desc,
                                            filt_desc,
                                            tensor_dim,
                                            outdimsinit)  
    
    outstrides = [outdims[1]*outdims[2]*outdims[3]*outdims[4], outdims[2]*outdims[3]*outdims[4], outdims[3]*outdims[4], outdims[4], 1]
    outconvdim = 5
    out_desc = libmiopen.miopenCreateTensorDescriptor()
    libmiopen.miopenSetTensorDescriptor(out_desc,
                                        data_type,
                                        outconvdim,
                                        outdims, 
                                        outstrides)
    out_bytes = int(outdims[0]*outdims[1]*outdims[2]*outdims[3]*outdims[4]*ctypes.sizeof(ctypes.c_float))
    out_data = libhip.hipMalloc(out_bytes)
    print(out_bytes)
    ws_size =  libmiopen.miopenConvolutionForwardGetWorkSpaceSize(miopen_context, 
                                                        filt_desc,
                                                        in_desc,
                                                        conv_desc,
                                                        out_desc)

    #Find algorithm to be able to run convolution
    search_ws = libhip.hipMalloc(ws_size)
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
    
    # Compute miopen workspace size
    return in_desc, in_data, out_desc, out_data, outdims, out_bytes, filt_desc, filt_data, ws_size, search_ws, perfresult

def miopendestroydescinoutfilt(in_desc, out_desc, filt_desc, ws_ptr):
    ws_ptr = None
    libmiopen.miopenDestroyTensorDescriptor(in_desc)
    libmiopen.miopenDestroyTensorDescriptor(out_desc)
    libmiopen.miopenDestroyTensorDescriptor(filt_desc)
    return in_desc, out_desc, filt_desc, ws_ptr