# Optimizing GPU Convnets 

This folder contains all scripts and files used for the thesis "Optimizing GPU Convnets".


## Automated verification and benchmarking
Main script for verification and runtime profiling on NVIDIA GPU is `profileNVConv3D.py` and for AMD GPU is `profileAMDConv3D.py`

### Sample commands
1. To verify layer 2: 
`python profileNVConv3D.py --currlayer 2 --lastlayer 3 --verify`

2. To profile layer 0 to 2 for csv cosmoflow: 
 `python profileNVConv3D.py --currlayer 2 --lastlayer 3 --compareprof --paramscsv cosmoflow`

3. To enable plots violin and comparative runtime: 
`python profileNVConv3D.py --currlayer 2 --lastlayer 3 --compareprof --enableplots`

4. To load from a precompiled SDFG folder: 
`python profileNVConv3D.py --currlayer 2 --lastlayer 3 --compareprof --loadprecompiled`

5. To select different DaCe implementation: 
`python profileNVConv3D.py --implementation directConvNCDHWtileIC --currlayer 2 --lastlayer 3 --verify`

## Available DaCe implementations
In the folder *nv_impl*. The implementations included in the report are:
1. Implicit GEMM algorithm with auto-optimize: `implicitGemmdace`
2. Implicit GEMM algorithm with tiling only: `implicitGemmTiledonlydace`
3. Implicit GEMM algorithm with buffering + tiling: `implicitGemmWarpTileddace`
4. Implicit GEMM algorithm with NCDHW layout: `implicitGemmNCDHWdace`
5. Implicit GEMM algorithm with splitK: `implicitGemmsplitKdace`
6. Direct algorithm with auto-optimize: `directConvNCDHWdace`
7. Direct algorithm with merged DHW: `directConvNCDHWnobuffer`
8. Direct algorithm with merged DHW + buffers: `directConvNCDHWmergeddace`
9. Direct algorithm with inchannel reordered: `directConvNCDHWtileIC`

## Available datasets
Available datasets are in the folder convparams. Only CosmoFlow is runnable.

## Helper code
1. `libcudnn.py` has the Python bindings for cuDNN for the required 3D convolution functions.
2. `libmiopen.py` has the Python bindings for MIOpen for the required 3D convolution functions.
3. `libhip.py` has the Python bindings for HIP for the required 3D convolution functions.
4. `cudnn_conv.cu`, `cudnn_nhwc.cu` has example on using 3D convolution in cuDNN
5. `miopen_conv.cpp` has example on using 3D convolution in MIOpen
6. `examplecudnn.py` is an example on using the libcudnn Python bindings.
7. `examplemiopen.py` is an example on using the libmiopen Python bindings.
8. `convutils.py` has the supporting utilities needed in profileConv3D.py

## Scripts
Scripts for the plots. 
1. `cosmoflowend2end.py` : To run the end to end CosmoFlow network.
2. `plotRoofline.py`: To plot rooflines.
3. `plotRuntimebreakdown.py`: To plot the breakdown of different layers.
4. `stresstest.py`: Code used for stress test.

## DaCe config files
1. For AMD: amd.dace.conf
2. For CUDA: cuda.dace.conf