import torch
import torch.nn.functional as F
import libmiopen, libhip
from miopenConv import miopen_init, miopensetlayerdesc, miopendestroydescinoutfilt


batchsize = 16
inchannels = 1
indepth = 130
inheight = 130
inwidth = 130
outchannels = 16
kdim = 3
outdepth = indepth - kdim + 1
outheight = inheight - kdim + 1
outwidth = inheight - kdim + 1

Input = torch.rand(batchsize, inchannels, indepth, inheight, inwidth).cuda()
kernel = torch.rand(outchannels, inchannels, kdim, kdim, kdim).cuda()
Output = torch.zeros(batchsize, outchannels, outdepth, outheight, outwidth).cuda()

Output = F.conv3d(Input, kernel, stride=1, padding='valid')
pad = 0
dil = 1
stride = 1
miopen_context, data_type, tensor_dim, conv_dim, convolution_mode, conv_desc = miopen_init(pad, stride, dil)
in_desc, in_data, filt_desc, filt_data, out_desc, out_data, out_data_ptr, outdims, out_bytes, out_data_verify, ws_size, workspace, convolution_algo = miopensetlayerdesc(miopen_context, conv_desc, Input, kernel, batchsize, kdim, inchannels, indepth, inheight, inwidth, outchannels, libmiopen.miopenDatatype['miopenFloat'], 5)
