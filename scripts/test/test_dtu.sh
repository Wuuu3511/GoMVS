#!/usr/bin/env bash
TESTPATH="datasets/dtu/test/dtu" 						# path to dataset dtu_test
TESTLIST="lists/dtu/test.txt"							# path to data_list
NORMALPATH="highresdtutest/"							# path to normal map
CKPT_FILE="outputs/dtu_training/model_000012.ckpt"	    # path to checkpoint file
FUSIBLE_PATH="/public/home/jiangwu/fusibile/build/fusibile"	  # path to fusible of gipuma
OUTDIR="./outputs/dtu_test" 						  # path to output
if [ ! -d $OUTDIR ]; then
	mkdir -p $OUTDIR
fi


CUDA_VISIBLE_DEVICES=0 python test.py \
--dataset=general_eval \
--batch_size=1 \
--testpath=$TESTPATH  \
--testlist=$TESTLIST \
--normalpath=$NORMALPATH \
--loadckpt=$CKPT_FILE \
--outdir=$OUTDIR \
--numdepth=192 \
--ndepths="48,32,8" \
--depth_inter_r="4.0,1.0,0.5" \
--interval_scale=1.06 \
--filter_method="o3d" \
--fusibile_exe_path=$FUSIBLE_PATH

