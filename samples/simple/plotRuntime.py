import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def addlabels(x,y):
    for i in range(len(x)):
        y[i] =round(y[i],2)
        plt.text(i,y[i],y[i])

outdir = 'outputplots'

'''
layer_name,dace_median,cudnn_median
in_16X32X34X34X34_k_3X3X3_och_64,28.05764865875244,4.81276798248291
in_16X64X18X18X18_k_3X3X3_och_128,12.46281623840332,2.3683199882507324
in_16X128X10X10X10_k_3X3X3_och_256,6.205536127090454,1.2108799815177917
in_16X256X6X6X6_k_3X3X3_och_256,2.394287943840027,0.5612640082836151
in_16X256X4X4X4_k_3X3X3_och_256,0.7526400089263916,0.36294400691986084

layer_name,dace_median,cudnn_median
in_16X32X34X34X34_k_3X3X3_och_64,24.66044807434082,4.814144134521484
in_16X64X18X18X18_k_3X3X3_och_128,11.852767944335938,2.366128087043762
in_16X128X10X10X10_k_3X3X3_och_256,6.297840118408203,1.2071359753608704
in_16X256X6X6X6_k_3X3X3_och_256,2.3207520246505737,0.5592159926891327
in_16X256X4X4X4_k_3X3X3_och_256,1.598207950592041,0.3630400002002716

layer_name,dace_median,cudnn_median
in_16X32X34X34X34_k_3X3X3_och_64,24.733983993530273,4.812367916107178
in_16X64X18X18X18_k_3X3X3_och_128,11.840783596038818,2.3660800457000732
in_16X128X10X10X10_k_3X3X3_och_256,6.449120044708252,1.2071679830551147
in_16X256X6X6X6_k_3X3X3_och_256,2.375183939933777,0.5593440234661102
in_16X256X4X4X4_k_3X3X3_och_256,1.603279948234558,0.3620480000972748

layer_name,dace_median,cudnn_median
in_16X32X34X34X34_k_3X3X3_och_64,28.01974391937256,4.813888072967529
in_16X64X18X18X18_k_3X3X3_och_128,12.447167873382568,2.360592007637024
in_16X128X10X10X10_k_3X3X3_och_256,6.199504137039185,1.2111679911613464
in_16X256X6X6X6_k_3X3X3_och_256,2.39027202129364,0.5612480044364929
in_16X256X4X4X4_k_3X3X3_och_256,0.7563839852809906,0.36425599455833435

'''

median_dace = [ 30.17, 99.75, 55.53, 44.72, 40.24, 25.9]
median_ref = [ 12.06 , 11.98, 4.77, 2.39, 1.17, 0.54]

layer_names = [ 'L0', 'L1', 'L2', 'L3', 'L4', 'L5']

if len(median_dace) != 0 and len(median_ref) !=0:
    font_size = 16
    plt.rc('font', size=font_size)
    plt.rc('axes', titlesize=font_size)     # fontsize of the axes title
    plt.rc('axes', labelsize=font_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=font_size)    # legend fontsize
    plt.rc('figure', titlesize=font_size)
    
    print("INFO: Plotting summary graph")
    # set width of bar
    barWidth = 0.2
    fig = plt.subplots(figsize =(8, 6))
    # Set position of bar on X axis
    br1 = np.arange(len(median_ref))
    br2 = [x + barWidth for x in br1]
    
    # Make the plot
    plt.bar(br1, median_ref, color ='pink', width = barWidth, edgecolor ='grey', label ='cudnn')
    plt.bar(br2, median_dace, color ='skyblue', width = barWidth, edgecolor ='grey', label ='dace')
    addlabels(br1, median_ref)
    addlabels(br2, median_dace)
    # Adding Xticks
    plt.xlabel('Variation across different layers', fontweight ='bold', fontsize = 15)
    plt.ylabel('Median runtime in ms', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(median_dace))], layer_names)
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.savefig(f'{outdir}/tileIC', bbox_inches='tight')