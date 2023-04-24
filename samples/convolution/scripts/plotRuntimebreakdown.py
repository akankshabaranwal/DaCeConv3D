import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load dataset
d = {'variant': ['cosmoflowv1 batchsize 1', 'cosmoflowv1 batchsize 8', 'cosmoflowv2 batchsize 1', 'cosmoflowv2 batchsize 8'], 'convolution layers': [6.56, 18.96, 7.09, 61.81], 'other layers': [0.78, 5.73, 1.66, 12.63]}
df = pd.DataFrame(data=d)

# view dataset
print(df)
  
# plot a Stacked Bar Chart using matplotlib
df.plot(
  x = 'variant', 
  kind = 'barh', 
  stacked = True, 
  title = 'Runtime distribution for cosmoflow (ms)',
  ylabel = 'Runtime in ms',
  xlabel = 'Variant of network model',
  mark_right = True)
  
df_total = df["other layers"] + df["convolution layers"]
df_rel = df[df.columns[1:]].div(df_total, 0)*100

for n in df_rel:
    for i, (cs, ab, pc) in enumerate(zip(df.iloc[:, 1:].cumsum(1)[n], 
                                         df[n], df_rel[n])):
        print(cs, ab, pc)
        # plt.text(cs - ab / 2, i, str(ab) + 'ms', 
        #          va = 'center', ha = 'center')
        
plt.savefig('outputplots/runtimebreakdown.png', bbox_inches='tight')
