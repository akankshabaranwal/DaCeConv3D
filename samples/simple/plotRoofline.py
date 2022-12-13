import matplotlib.pyplot as plt
import numpy as np

# https://crd.lbl.gov/assets/Uploads/cug19-roofline-final.pdf
# Set style
plt.style.use('seaborn-whitegrid')
plt.xlabel("Arithmetic Intensity: FLOPs/Byte")
plt.ylabel("Performance: GFLOPs/s")

# Peak values picked from here: https://crd.lbl.gov/assets/Uploads/cug19-roofline-final.pdf 
plt.title("Roofline Log Log Plot for cutlass on GV100 for different cosmoflow layers")
plt.suptitle("Device: Tesla V100")
x = np.linspace(0.01, 100000000, 10)

# Max performance metrics
fp32Max = 14800
tensorPerf = 118500
L2roofline = 4198.4 # GB/s
DRAMroofline = 900 # GB/s

# Plotting the maximum floating point performance
plt.axhline(y=fp32Max, color='m', linestyle='--', label='FP32 Maximum 14.8 TFLOPS')

# Plotting tensor core performance
plt.axhline(y=tensorPerf, linestyle='--', label='Tensor Core maximum 118.5 TFLOPS')

# L2 roofline
y = L2roofline*x
plt.loglog(x, y, '-.g', label='L2 4198.4 GB/s')
plt.ylim(100,1e6*0.3)

# DRAM roofline
y = DRAMroofline*x
plt.loglog(x,y,'-.b',label='DRAM 900 GB/s')
plt.ylim(100,1e6*0.3)


# Function points
layer2_AI = 27
layer2_Perf = 10001.6
plt.loglog(layer2_AI, layer2_Perf,'*r')
plt.text(layer2_AI, layer2_Perf*0.8, 'L2')

layer3_AI = 55
layer3_Perf = 32610
plt.loglog(layer3_AI, layer3_Perf,'*r')
plt.text(layer3_AI, layer3_Perf*0.8, 'L3')

layer4_AI = 111
layer4_Perf = 51796.3
plt.loglog(layer4_AI, layer4_Perf,'*r')
plt.text(layer4_AI*0.8, layer4_Perf*0.8, 'L4')

layer5_AI = 217
layer5_Perf = 53627.8
plt.loglog(layer5_AI, layer5_Perf,'*r')
plt.text(layer5_AI*0.9, layer5_Perf, 'L5')

layer6_AI = 193
layer6_Perf = 6799.84
plt.loglog(layer6_AI, layer6_Perf,'*r')
plt.text(layer6_AI, layer6_Perf*0.8, 'L6')

layer7_AI = 83
layer7_Perf = 877.18
plt.loglog(layer7_AI, layer7_Perf,'*r')
plt.text(layer7_AI, layer7_Perf*0.8, 'L7')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('outputplots/rooflinetest.png', bbox_inches='tight')

exit()