import matplotlib.pyplot as plt
import numpy as np

# https://crd.lbl.gov/assets/Uploads/cug19-roofline-final.pdf
# Set style
plt.style.use('seaborn-whitegrid')
plt.xlabel("FLOPS/byte")
plt.ylabel("GFLOPs/s")

# Peak values picked from here: https://crd.lbl.gov/assets/Uploads/cug19-roofline-final.pdf 
plt.title("Roofline Log Log Plot for cosmoflow layer 1")
plt.suptitle("Device: Tesla V100")
x = np.linspace(0.01, 100000000, 10)

# Max performance metrics
fp32Max = 14800
L2roofline = 4198.4 # GB/s
DRAMroofline = 900 # GB/s

# Plotting the maximum floating point performance
plt.axhline(y=fp32Max, color='m', linestyle='--', label='FP32 Maximum 14.8 TFLOPS')

# L2 roofline
y = L2roofline*x
plt.loglog(x, y, '-.g', label='L2 4198.4 GB/s')
plt.ylim(100,1e4*0.3)

# DRAM roofline
y = DRAMroofline*x
plt.loglog(x,y,'-.b',label='DRAM 900 GB/s')
plt.ylim(100,1e5*0.3)


dace_AI = 7.74E+02
dace_Perf = 1.72E+03
plt.loglog(dace_AI, dace_Perf,'.r')
plt.text(dace_AI, dace_Perf*0.8, 'dace_dram')

dace_AI = 1.49E+02
dace_Perf = 1.72E+03
plt.loglog(dace_AI, dace_Perf,'.r')
plt.text(dace_AI*0.5, dace_Perf*1.2, 'dace_l2')


tf_AI = 1.90E+03
tf_Perf = 6.63E+03
plt.loglog(tf_AI, tf_Perf,'.g')
plt.text(tf_AI, tf_Perf*0.8, 'tf_dram')

tf_AI = 4.66E+02
tf_Perf = 6.63E+03
plt.loglog(tf_AI, tf_Perf,'.g')
plt.text(tf_AI, tf_Perf*1.2, 'tf_l2')


plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('outputplots/rooflinetest.png', bbox_inches='tight')

exit()

# Kernelwise measurements from nvprof

RNN_persist_fp_AI = 3.012
RNN_persist_fp_Perf = 339.1266
plt.loglog(RNN_persist_fp_AI,RNN_persist_fp_Perf,'*r',
           label='RNN_persist_fp (Time Util: 11.3%)')

volta_fp16_s884gemm_fp16_256x128_ldg8_f2f_tn_AI = 0.24
volta_fp16_s884gemm_fp16_256x128_ldg8_f2f_tn_Perf = 44.65
plt.loglog(volta_fp16_s884gemm_fp16_256x128_ldg8_f2f_tn_AI,volta_fp16_s884gemm_fp16_256x128_ldg8_f2f_tn_Perf,'*g',
           label='volta_fp16_s884gemm_fp16_256x128_ldg8_f2f_tn (Time Util: 5.31%)')

persistRNN_init_AI = 0.001757674
persistRNN_init_Perf = 0.821578253
plt.loglog(persistRNN_init_AI,persistRNN_init_Perf,'*m',
           label='persistRNN_init (Time Util: 1.65%)')

volta_s884gemm_fp16_64x64_ldg8_nn_AI = 0.372251949
volta_s884gemm_fp16_64x64_ldg8_nn_Perf = 0.036596389
plt.loglog(volta_s884gemm_fp16_64x64_ldg8_nn_AI,volta_s884gemm_fp16_64x64_ldg8_nn_Perf,'*b',
           label='volta_s884gemm_fp16_64x64_ldg8_nn (Time Util: 28.04%)')

LSTM_elementWise_fp_AI = 1.702619287
LSTM_elementWise_fp_Perf = 90.91165477
plt.loglog(LSTM_elementWise_fp_AI,LSTM_elementWise_fp_Perf,'*c',
           label='LSTM_elementWise_fp (Time Util: 5.12%)')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('outputplots/rooflinetest.png', bbox_inches='tight')