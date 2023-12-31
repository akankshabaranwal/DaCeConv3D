import matplotlib.pyplot as plt
import numpy as np

# https://crd.lbl.gov/assets/Uploads/cug19-roofline-final.pdf
# Set style
plt.style.use('seaborn-whitegrid')
plt.xlabel("Arithmetic Intensity: FLOPs/Byte")
plt.ylabel("Performance: GFLOPs/s")

# Peak values picked from here: https://crd.lbl.gov/assets/Uploads/cug19-roofline-final.pdf 
#plt.title("Roofline log log plot for cudnn implicit gemm with batch size = 16")
#plt.suptitle("Tesla GV100, ault23, cosmoflow layers")
x = np.linspace(0.01, 10000000, 10)

# Max performance metrics
fp32Max = 12819 # Value updated using nsight compute
#tensorPerf = 118500
L2roofline = 4199.4 # GB/s
DRAMroofline = 900 # GB/s

# Plotting the maximum floating point performance
plt.axhline(y=fp32Max, color='r', linestyle='--')
plt.text( 1000, fp32Max*0.6, 'FP32 Maximum 12.8 TFLOPS', color='r')

# # Plotting tensor core performance
# plt.axhline(y=tensorPerf, linestyle='--', label='Tensor Core maximum 118.5 TFLOPS')

# L2 roofline
y = L2roofline*x
plt.loglog(x, y, '--g')
plt.ylim(100,1e5*0.3)
plt.text( 0.1, 200, 'L2 4198.4 GB/s', color='g')

# DRAM roofline
y = DRAMroofline*x
plt.loglog(x, y, '--b')
plt.ylim(100,1e5*0.3)
plt.text( 1, 400, 'DRAM 900 GB/s', color='b')

# https://matplotlib.org/stable/api/markers_api.html

layer0_AI = 15.74
layer0_Perf = 5205
L0 = plt.loglog(layer0_AI, layer0_Perf,'o', label='implicit_convolveNd_sgemm')
plt.annotate('0', (layer0_AI, layer0_Perf))
 
layer1_AI = 28.23
layer1_Perf = 9655
L1 = plt.loglog(layer1_AI, layer1_Perf,'v', label='volta_128X32_3dconv')
plt.annotate('1', (layer1_AI, layer1_Perf))

layer2_AI = 55.60
layer2_Perf = 11596
L2 = plt.loglog(layer2_AI, layer2_Perf,'^', label='volta_128X64_3dconv')
plt.annotate('2', (layer2_AI, layer2_Perf))

layer3_AI = 108.81
layer3_Perf = 11710
L3 = plt.loglog(layer3_AI, layer3_Perf,'D', label='volta_128X128_3dconv')
plt.annotate('3', (layer3_AI, layer3_Perf))

layer4_AI = 188.75
layer4_Perf = 11882
L4 = plt.loglog(layer4_AI, layer4_Perf,'>', label='volta_128X128_3dconv')
plt.annotate('4', (layer4_AI, layer4_Perf))

layer5_AI = 247.96
layer5_Perf = 6857
L5 = plt.loglog(layer5_AI, layer5_Perf,'<', label='volta_128X64_3dconv')
plt.annotate('5', (layer5_AI, layer5_Perf))

layer6_AI = 54.86
layer6_Perf = 1438
L6 = plt.loglog(layer6_AI, layer6_Perf,'s', label='volta_128X32_3dconv')
plt.annotate('6', (layer6_AI, layer6_Perf))

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('outputplots/roofline_batch16.png', bbox_inches='tight')

exit()