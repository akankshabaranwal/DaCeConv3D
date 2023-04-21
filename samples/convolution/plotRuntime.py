import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def addlabels1(x,y):
    for i in range(len(x)):
        if(y[i]!=0):
            y[i] =round(y[i],2)
            plt.text(i,y[i],y[i], horizontalalignment='right', verticalalignment='top')

def addlabels(x,y):
    for i in range(len(x)):
        if(y[i]!=0):
            y[i] =round(y[i],2)
            plt.text(i,y[i],y[i], horizontalalignment='left', verticalalignment='bottom')

outdir = 'outputplots'

median_new = [ 22.18, 72.22, 36.36, 20.81, 10.92, 3.59,0]
median_prev = [  20.87, 80.02, 41.59, 22.53, 11.7, 4.3,0]

layer_names = [ 'L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6']

if len(median_prev) != 0 and len(median_new) !=0:
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
    br1 = np.arange(len(median_prev))
    br2 = [x + barWidth for x in br1]
    
    # Make the plot
    plt.bar(br1, median_prev, color ='tomato', width = barWidth, edgecolor ='grey', label ='previous best')
    plt.bar(br2, median_new, color ='olive', width = barWidth, edgecolor ='grey', label ='split K')
    
    #plt.bar(br1, median_prev, color ='skyblue', width = barWidth, edgecolor ='grey', label ='dace')
    #plt.bar(br2, median_new, color ='pink', width = barWidth, edgecolor ='grey', label ='cudnn')
    
    #addlabels1(br1, median_prev)
    addlabels(br2, median_new)
    # Adding Xticks
    plt.xlabel('Variation across different layers', fontweight ='bold', fontsize = font_size)
    plt.ylabel('Median runtime in ms', fontweight ='bold', fontsize = font_size)
    plt.xticks([r + barWidth for r in range(len(median_new))], layer_names)
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.savefig(f'{outdir}/tmp', bbox_inches='tight')
