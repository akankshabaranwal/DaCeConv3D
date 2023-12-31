import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def addlabels(x,y):
    for i in range(len(x)):
        y[i] =round(y[i],2)
        plt.text(i,y[i],y[i])

outdir = 'outputplots'


median_dace = [ 10.62, 27.9, 13.03, 6.72, 2.37, 1.47, 1.36]
median_ref = [22.24, 7.74, 3.31, 1.71, 1.15, 0.78, 0.76]

layer_names = [ 'L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6']

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
    plt.bar(br1, median_ref, color ='pink', width = barWidth, edgecolor ='grey', label ='miopen')
    plt.bar(br2, median_dace, color ='skyblue', width = barWidth, edgecolor ='grey', label ='dace')
    addlabels(br1, median_ref)
    addlabels(br2, median_dace)
    # Adding Xticks
    plt.xlabel('Variation across different layers', fontweight ='bold', fontsize = 15)
    plt.ylabel('Median runtime in ms', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(median_dace))], layer_names)
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.savefig(f'{outdir}/cosmoflowv1vsmiopen', bbox_inches='tight')