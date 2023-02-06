from convutils import addlabels

import matplotlib.pyplot as plt
import numpy as np



#median_dace_new = [21.48, 60.88, 31.39, 18.96, 10.19, 3.41, 6.18]
median_dace_direct = [21.48, 60.88, 31.39, 18.96, 10.19, 3.41, 3.71]
median_dace_implicit = [26.95, 44.9, 23.03, 12.33, 8.87, 2.8, 0.85]
median_cudnn = [13.34, 12.83, 5.24, 2.56, 1.33, 0.61, 0.39]
#array_speedup = (np.array(median_dace)/np.array(median_cudnn))
#ratio_speedup = array_speedup.tolist()
layer_names  = ['L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6']
#print(ratio_speedup)
print("INFO: Plotting summary graph")
# set width of bar
barWidth = 0.2
fig = plt.subplots(figsize =(12, 8))
# Set position of bar on X axis
br1 = np.arange(len(median_cudnn))
br2 = [x + barWidth for x in br1]
br3 = [x + 2*barWidth for x in br1]

# Make the plot
plt.bar(br1, median_cudnn, color ='pink', width = barWidth, edgecolor ='grey', label ='cudnn')
plt.bar(br2, median_dace_direct, color ='skyblue', width = barWidth, edgecolor ='grey', label ='dace_direct')
plt.bar(br3, median_dace_implicit, color ='lightgreen', width = barWidth, edgecolor ='grey', label ='dace__implicit')
addlabels(br1, median_cudnn)
addlabels(br2, median_dace_direct)
addlabels(br3, median_dace_implicit)
# Adding Xticks
plt.xlabel('Variation across different layers', fontweight ='bold', fontsize = 15)
plt.ylabel('Median runtime in ms', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(median_dace_direct))], layer_names)
plt.legend()
plt.xticks(rotation=45, ha='right')
outdir = '/users/abaranwa/dacelocal/samples/simple/outputplots'
plt.savefig(f'{outdir}/median_runtime', bbox_inches='tight')



outdepth = 1
outheight = 1
outwidth = 1
batchsize = 1
outchannels = 1
inchannels = 1
kdim = 3

def direct_conv3d(input, kernel, output):
    DHW = outdepth*outheight*outwidth
    HW = outheight*outwidth
    for n in range(0, batchsize):
        for dhw in range(0, DHW):        
            for oc in range(0, outchannels):
                d, dhw_residual = divmod(dhw, HW)
                h, w = divmod(dhw_residual, outheight)
                for ic in range(0, inchannels):
                    for kd in range(0, kdim):
                        for kh in range(0, kdim):
                            for kw in range(0, kdim):
                                    output[n, d, h, w, oc] = output[n, d, h, w, oc] + input[ n, d+kd, h+kh, w+kw, ic]*kernel[oc, kd, kh, kw, ic]