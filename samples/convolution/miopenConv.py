import libmiopen, libhip, ctypes

def miopen_init(pad, stride, dil):
    miopen_context = libmiopen.miopenCreate()
    data_type = libmiopen.miopenDatatype['miopenFloat']
    tensor_dim = 5
    conv_dim = tensor_dim-2
    convolution_mode = libmiopen.miopenConvolutionMode['miopenConvolution']
    convpad = [pad, pad, pad]
    convstr = [stride, stride, stride]
    convdil = [dil, dil, dil]
    conv_desc = libmiopen.miopenCreateConvolutionDescriptor()    
    libmiopen.miopenInitConvolutionNdDescriptor(conv_desc,
                                                conv_dim,
                                                convpad,
                                                convstr,
                                                convdil,
                                                convolution_mode,
                                                data_type)
    return miopen_context, data_type, tensor_dim, conv_dim, convolution_mode, conv_desc

def miopensetlayerdesc(miopen_context, conv_desc, d_input, d_kernel, batchsize, kdim, inchannels, indepth, inheight, inwidth, outchannels, data_type, tensor_dim):
    in_n = batchsize
    in_c = inchannels
    in_d = indepth
    in_h = inheight
    in_w = inwidth
    in_dims = [in_n, in_c, in_d, in_h, in_w]
    in_strides = [in_c*in_d*in_h*in_w, in_d*in_h*in_w, in_h*in_w, in_w, 1]
    in_desc = libmiopen.miopenCreateTensorDescriptor()
    libmiopen.miopenSetTensorDescriptor(in_desc,
                                        data_type, 
                                        tensor_dim,
                                        in_dims, 
                                        in_strides)
    in_bytes = int(in_n*in_c*in_d*in_h*in_w*ctypes.sizeof(ctypes.c_float))
    in_data = libhip.hipMalloc(in_bytes)
    input_data_ptr = d_input.data_ptr()
    libhip.hipMemcpyDtoD(in_data, input_data_ptr, in_bytes)

    filt_k = outchannels
    filt_c = inchannels
    filt_d = kdim
    filt_h = kdim
    filt_w = kdim
    filt_dims = [filt_k, filt_c, filt_d, filt_h, filt_w]
    filt_strides = [filt_c*filt_d*filt_h*filt_w, filt_d*filt_h*filt_w, filt_h*filt_w, filt_w, 1]
    filt_desc = libmiopen.miopenCreateTensorDescriptor()
    libmiopen.miopenSetTensorDescriptor(filt_desc,
                                        data_type, 
                                        tensor_dim,
                                        filt_dims, 
                                        filt_strides)
    filt_bytes = int(filt_k*filt_c*filt_d*filt_h*filt_w*ctypes.sizeof(ctypes.c_float))
    filt_data = libhip.hipMalloc(filt_bytes)
    
    filt_data_ptr = d_kernel.data_ptr()
    libhip.hipMemcpyDtoD(filt_data, filt_data_ptr, filt_bytes)
    print("Filter tensor created")

    outdimsinit = [0, 0, 0, 0, 0]
    tensor_dim_array = [5]
    outdims = libmiopen.miopenGetConvolutionNdForwardOutputDim(conv_desc, 
                                                in_desc,
                                                filt_desc,
                                                tensor_dim_array,
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
    import torch
    out_data_verify = torch.zeros(out_n, out_c, out_d, out_h, out_w).cuda()
    out_data_ptr2 = out_data_verify.data_ptr()
    ws_size =  libmiopen.miopenConvolutionForwardGetWorkSpaceSize(miopen_context, 
                                                            filt_desc,
                                                            in_desc,
                                                            conv_desc,
                                                            out_desc)
    workspace = libhip.hipMalloc(ws_size)
    requestedalgocount = 5
    print("INFO: Launching MIOpen find it takes a few seconds")
    perfresult = libmiopen.miopenFindConvolutionForwardAlgorithm(miopen_context, 
                                                    in_desc, in_data,
                                                    filt_desc, filt_data,
                                                    conv_desc, 
                                                    out_desc, out_data,
                                                    requestedalgocount,
                                                    workspace, ws_size.value
                                                    )
    for convresult in perfresult:
        print(f"For algorithm {convresult.fwd_algo}, time is {convresult.time}, memory is {convresult.memory}.")
    convolution_algo = perfresult[0].fwd_algo
    print(f"INFO: Done launching MIOpen find, optimal convolution algo is: {convolution_algo}. Change the MIOPEN_FIND_MODE to 1 for exhaustive finding the algorithm")
    return in_desc, in_data, filt_desc, filt_data, out_desc, out_data, out_data_ptr2, outdims, out_bytes, out_data_verify, ws_size, workspace, convolution_algo

def miopendestroydescinoutfilt(in_desc, out_desc, filt_desc, ws_ptr):
    return in_desc, out_desc, filt_desc, ws_ptr
    ws_ptr = None
    libmiopen.miopenDestroyTensorDescriptor(in_desc)
    libmiopen.miopenDestroyTensorDescriptor(out_desc)
    libmiopen.miopenDestroyTensorDescriptor(filt_desc)
    return in_desc, out_desc, filt_desc, ws_ptr