import torch
import torch.nn.functional as F

# https://catalog.ngc.nvidia.com/orgs/nvidia/resources/unet3d_medical_for_tensorflow/files
# Version 21.10.0

# Needs dace implementation for stride 2 if the goal is to compare against dace.

batchsize = 1
l0_input = torch.rand(batchsize, 3, 128, 128, 128).cuda().float()
