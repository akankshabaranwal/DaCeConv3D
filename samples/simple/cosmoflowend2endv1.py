import torch
import torch.nn.functional as F

from daceml.testing.profiling import time_funcs, print_time_statistics

# Source: https://github.com/mlcommons/hpc/blob/main/cosmoflow/models/cosmoflow.py 
# Source: https://github.com/mlcommons/hpc/blob/main/cosmoflow/configs/cosmo.yaml 
# Pytorch variant
poolF = torch.nn.MaxPool3d(2, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
reluF =  torch.nn.LeakyReLU(0.1)

# Maybe create a plot with this for different batch sizes.
batchsize = 8
indepth = 128
numfilters = 16

fc1_size = 2048
fc2_size = 256

target_size = 3

# Layer definitions
l0_input = torch.rand(batchsize, 1, 128, 128, 128).cuda().float()
l0_kernel = torch.rand(numfilters, 1, 3, 3, 3).cuda().float()
l1_kernel = torch.rand(numfilters*2, numfilters, 3, 3, 3).cuda().float()
l2_kernel = torch.rand(numfilters*4, numfilters*2, 3, 3, 3).cuda().float()
l3_kernel = torch.rand(numfilters*8, numfilters*4, 3, 3, 3).cuda().float()
l4_kernel = torch.rand(numfilters*16, numfilters*8, 3, 3, 3).cuda().float()
l5_kernel = torch.rand(numfilters*32, numfilters*16, 3, 3, 3).cuda().float()
l6_kernel = torch.rand(numfilters*64, numfilters*32, 3, 3, 3).cuda().float()

l7_fclayer = torch.rand(fc1_size, 1024*batchsize).cuda().float()
l8_fclayer = torch.rand(fc2_size, fc1_size).cuda().float()
l9_fclayer = torch.rand(target_size, fc2_size).cuda().float()

l0_output = F.conv3d(l0_input, l0_kernel, stride=1, padding='valid')
l1_input = poolF(reluF(l0_output))
l1_output = F.conv3d(l1_input, l1_kernel, stride=1, padding='same')
l2_input = poolF(reluF(l1_output))
l2_output = F.conv3d(l2_input, l2_kernel, stride=1, padding='same')
l3_input = poolF(reluF(l2_output))
l3_output = F.conv3d(l3_input, l3_kernel, stride=1, padding='same')
l4_input = poolF(reluF(l3_output))
l4_output = F.conv3d(l4_input, l4_kernel, stride=1, padding='same')
l5_input = poolF(reluF(l4_output))
l5_output = F.conv3d(l5_input, l5_kernel, stride=1, padding='same')
l6_input = poolF(reluF(l5_output))
l6_output = F.conv3d(l6_input, l6_kernel, stride=1, padding='same')

l7_input = l6_output.flatten()
l7_output = F.linear(l7_input, l7_fclayer)
l8_output = F.linear(l7_output, l8_fclayer)
l9_output = F.linear(l8_output, l9_fclayer)

def run_end2end():
    l0_output = F.conv3d(l0_input, l0_kernel, stride=1, padding='valid')
    l1_input = poolF(reluF(l0_output))
    l1_output = F.conv3d(l1_input, l1_kernel, stride=1, padding='same')
    l2_input = poolF(reluF(l1_output))
    l2_output = F.conv3d(l2_input, l2_kernel, stride=1, padding='same')
    l3_input = poolF(reluF(l2_output))
    l3_output = F.conv3d(l3_input, l3_kernel, stride=1, padding='same')
    l4_input = poolF(reluF(l3_output))
    l4_output = F.conv3d(l4_input, l4_kernel, stride=1, padding='same')
    l5_input = poolF(reluF(l4_output))
    l5_output = F.conv3d(l5_input, l5_kernel, stride=1, padding='same')
    l6_input = poolF(reluF(l5_output))
    l6_output = F.conv3d(l6_input, l6_kernel, stride=1, padding='same')

    l7_input = l6_output.flatten()
    l7_output = F.linear(l7_input, l7_fclayer)
    l8_output = F.linear(l7_output, l8_fclayer)
    l9_output = F.linear(l8_output, l9_fclayer)
    return l9_output

def run_onlyconv():
    l0_output = F.conv3d(l0_input, l0_kernel, stride=1, padding='same')
    l1_output = F.conv3d(l1_input, l1_kernel, stride=1, padding='same')
    l2_output = F.conv3d(l2_input, l2_kernel, stride=1, padding='same')
    l3_output = F.conv3d(l3_input, l3_kernel, stride=1, padding='same')
    l4_output = F.conv3d(l4_input, l4_kernel, stride=1, padding='same')
    l5_output = F.conv3d(l5_input, l5_kernel, stride=1, padding='same')
    l6_output = F.conv3d(l6_input, l6_kernel, stride=1, padding='same')

    return l0_output, l1_output, l2_output, l3_output, l4_output, l5_output, l6_output

x = run_end2end()
print(x.shape)

times = time_funcs([run_end2end],
                func_names=["end2end"],
                warmups=10,
                num_iters=100,
                launch_wait=False)
print_time_statistics(times, [ "end2end"])

times = time_funcs([run_onlyconv],
                func_names=["onlyconv"],
                warmups=10,
                num_iters=10,
                launch_wait=False)
print_time_statistics(times, [ "onlyconv"])
