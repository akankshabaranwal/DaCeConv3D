#from soaptest import soap_analysis
import dace 
import numpy as np
from dace import dtypes
import torch
from dace.transformation.estimator.soap.io_analysis import perform_soap_analysis
from dace.transformation.estimator.soap.utils import get_lead_term
import sympy

d_M = dace.symbol('d_M')
d_N = dace.symbol('d_N')
d_K = dace.symbol('d_K')

dtype = dace.float32
np_dtype = np.float32

M = 64
N = 32
K = 64

dace.Config.set('compiler', 'default_data_types', value='C')

@dace.program(device=dtypes.DeviceType.GPU, auto_optimize=True)
def dace_matmul( A: dtype[d_M, d_K] @dace.StorageType.GPU_Global ,
                B: dtype[d_K, d_N] @dace.StorageType.GPU_Global,
                C: dtype[d_M, d_N] @dace.StorageType.GPU_Global):
    for i, j, k in dace.map[0:d_M, 0:d_N, 0:d_K]:
        C[i,j] = C[i,j] + A[i,k]*B[k,j]
    
d_A = torch.rand(M, K).cuda()
d_B = torch.rand(K, N).cuda()
d_C = torch.zeros(M, N).cuda()

sdfg_matmul: dace.SDFG = dace_matmul.to_sdfg(d_A, d_B, d_C)
sdfg_matmul.apply_gpu_transformations()
sdfg_matmul(A=d_A, B=d_B, C=d_C, d_M = M, d_N = N, d_K = K)
result = perform_soap_analysis(sdfg_matmul, generate_schedule=False, solver_timeout=60)
print("Result printing starts")
print(result)
print("Result printing ends")