#!/bin/sh
for i in {0..0}; do 
	echo $i;
	nsys nvprof --profile-from-start off -f -o tf$i python profileConv3D.py --cl 0 --nvtimefuncstf
	nsys stats tf$i.qdrep -o tf$i --format csv
	python getsummary.py -type gpukernsum -prefix tf$i
	nsys nvprof --profile-from-start off -f -o dace$i python profileConv3D.py --cl 0 --nvtimefuncsdace
	nsys stats dace$i.qdrep -o dace$i --format csv
	python getsummary.py -type gpukernsum -prefix dace$i
done
