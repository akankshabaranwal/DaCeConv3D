import torch
import torch.nn.functional as F

from daceml.testing.profiling import time_funcs, print_time_statistics

# Source: https://github.com/mlcommons/hpc/blob/main/cosmoflow/models/cosmoflow.py 
# Source: https://github.com/mlcommons/hpc/blob/main/cosmoflow/configs/cosmo.yaml 
# Pytorch variant
poolF = torch.nn.MaxPool3d(2, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
reluF =  torch.nn.LeakyReLU(0.1)

# Maybe create a plot with this for different batch sizes.
batchsize = 4
indepth = 128
numfilters = 32

fc1_size = 128  
fc2_size = 64

target_size = 4

# Layer definitions
l0_input = torch.rand(batchsize, 4, 128, 128, 128).cuda().float()
l0_kernel = torch.rand(numfilters, 4, 3, 3, 3).cuda().float()
l1_kernel = torch.rand(numfilters*2, numfilters, 3, 3, 3).cuda().float()
l2_kernel = torch.rand(numfilters*4, numfilters*2, 3, 3, 3).cuda().float()
l3_kernel = torch.rand(numfilters*8, numfilters*4, 3, 3, 3).cuda().float()
l4_kernel = torch.rand(numfilters*16, numfilters*8, 3, 3, 3).cuda().float()
l5_fclayer = torch.rand(fc1_size, 262144*batchsize).cuda().float()
l6_fclayer = torch.rand(fc2_size, fc1_size).cuda().float()
l7_fclayer = torch.rand(target_size, fc2_size).cuda().float()

l0_output = F.conv3d(l0_input, l0_kernel, stride=1, padding='same')
l1_input = poolF(reluF(l0_output))
l1_output = F.conv3d(l1_input, l1_kernel, stride=1, padding='same')
l2_input = poolF(reluF(l1_output))
l2_output = F.conv3d(l2_input, l2_kernel, stride=1, padding='same')
l3_input = poolF(reluF(l2_output))
l3_output = F.conv3d(l3_input, l3_kernel, stride=1, padding='same')
l4_input = poolF(reluF(l3_output))
l4_output = F.conv3d(l4_input, l4_kernel, stride=1, padding='same')
l5_input = l4_output.flatten()
l5_output = F.linear(l5_input, l5_fclayer)
l6_output = F.linear(l5_output, l6_fclayer)
l7_output = F.linear(l6_output, l7_fclayer)

def run_end2end():
    l0_output = F.conv3d(l0_input, l0_kernel, stride=1, padding='same')
    l1_input = poolF(reluF(l0_output))
    l1_output = F.conv3d(l1_input, l1_kernel, stride=1, padding='same')
    l2_input = poolF(reluF(l1_output))
    l2_output = F.conv3d(l2_input, l2_kernel, stride=1, padding='same')
    l3_input = poolF(reluF(l2_output))
    l3_output = F.conv3d(l3_input, l3_kernel, stride=1, padding='same')
    l4_input = poolF(reluF(l3_output))
    l4_output = F.conv3d(l4_input, l4_kernel, stride=1, padding='same')
    l5_input = l4_output.flatten()
    l5_output = F.linear(l5_input, l5_fclayer)
    l6_output = F.linear(l5_output, l6_fclayer)
    l7_output = F.linear(l6_output, l7_fclayer)
    return l7_output

def run_onlyconv():
    l0_output = F.conv3d(l0_input, l0_kernel, stride=1, padding='same')
    l1_output = F.conv3d(l1_input, l1_kernel, stride=1, padding='same')
    l2_output = F.conv3d(l2_input, l2_kernel, stride=1, padding='same')
    l3_output = F.conv3d(l3_input, l3_kernel, stride=1, padding='same')
    l4_output = F.conv3d(l4_input, l4_kernel, stride=1, padding='same')
    return l0_output, l1_output, l2_output, l3_output, l4_output

x = run_end2end()
print(x.shape)

times = time_funcs([run_end2end],
                func_names=["end2end"],
                warmups=5,
                num_iters=10,
                launch_wait=False)
print_time_statistics(times, [ "end2end"])

times = time_funcs([run_onlyconv],
                func_names=["onlyconv"],
                warmups=5,
                num_iters=10,
                launch_wait=False)
print_time_statistics(times, [ "onlyconv"])
