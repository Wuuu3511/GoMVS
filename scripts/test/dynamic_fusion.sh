#!/usr/bin/env bash
TESTPATH="datasets/tanks/tankandtemples/advanced" # path to dataset
TESTLIST="lists/tnt/adv.txt"					  # path to data_list
OUTDIR="outputs/tnt_testing" 					# path to save the results
if [ ! -d $OUTDIR ]; then
	mkdir -p $OUTDIR
fi


python dynamic_fusion.py \
--testpath=$OUTDIR \
--tntpath=$TESTPATH \
--testlist=$TESTLIST \
--outdir=$OUTDIR \
--photo_threshold=0.18 \
--thres_view=5 \
--test_dataset=tnt