#!/bin/sh
for i in {0..6}; do 
	echo $i;
	nvprof --metrics flop_count_sp --csv --log-file tfflop$i.csv python profileConv3D.py --proftf --currlayer $i
	nvprof --metrics l2_read_transactions --csv --log-file tfl2read$i.csv python profileConv3D.py --proftf --currlayer $i
	nvprof --metrics l2_write_transactions --csv --log-file tfl2write$i.csv python profileConv3D.py --proftf --currlayer $i
	nvprof --metrics dram_read_transactions --csv --log-file tfdramread$i.csv python profileConv3D.py --proftf --currlayer $i
	nvprof --metrics dram_write_transactions --csv --log-file tfdramwrite$i.csv python profileConv3D.py --proftf --currlayer $i
	nvprof --print-gpu-summary --csv --log-file tftime$i.csv python profileConv3D.py --proftf --currlayer $i
	nvprof --metrics flop_count_sp --csv --log-file daceflop$i.csv python profileConv3D.py --profoptimdace --currlayer $i
	nvprof --metrics l2_read_transactions --csv --log-file dacel2read$i.csv python profileConv3D.py --profoptimdace --currlayer $i
	nvprof --metrics l2_write_transactions --csv --log-file dacel2write$i.csv python profileConv3D.py --profoptimdace --currlayer $i
	nvprof --metrics dram_read_transactions --csv --log-file dacedramread$i.csv python profileConv3D.py --profoptimdace --currlayer $i
	nvprof --metrics dram_write_transactions --csv --log-file dacedramwrite$i.csv python profileConv3D.py --profoptimdace --currlayer $i
	nvprof --print-gpu-summary --csv --log-file dacetime$i.csv python profileConv3D.py --profoptimdace --currlayer $i
done