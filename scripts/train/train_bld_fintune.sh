#!/usr/bin/env bash
MVS_TRAINING="datasets/blendermvs/"  # path to BlendedMVS dataset
CKPT="outputs/dtu_training/model_000012.ckpt" # path to checkpoint
NORMAL_PATH="highresblend/"
LOG_DIR="outputs/bld_finetune"
if [ ! -d $LOG_DIR ]; then
	mkdir -p $LOG_DIR
fi


NGPUS=4
BATCH_SIZE=1
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nproc_per_node=$NGPUS finetune.py \
--logdir=$LOG_DIR \
--dataset=bld_train \
--trainpath=$MVS_TRAINING \
--normalpath=$NORMAL_PATH \
--ndepths="48,32,8"  \
--depth_inter_r="4,1,0.25" \
--dlossw="1.0,1.0,1.0" \
--loadckpt=$CKPT \
--eval_freq=1 \
--wd=0.0001 \
--nviews=9 \
--batch_size=$BATCH_SIZE \
--lr=0.0002 \
--lrepochs="6,10,14:2" \
--epochs=16 \
--mode="train" \
--trainlist=lists/bld/training_list.txt \
--testlist=lists/bld/validation_list.txt \
--numdepth=192 ${@:3} | tee -a $LOG_DIR/log.txt > trainbld.log 2>&1 &
