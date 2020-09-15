#!/usr/bin/env bash
echo "Test SPCL project"
time=$(date +%F)
CUDA_VISIBLE_DEVICES=0,1,2,3 python projects/SPCL/train_net.py --config-file projects/SPCL/configs/Market1501/usl_R50.yml OUTPUT_DIR logs/test/$time/dev-wuyiming/spcl/market1501 &
sleep 10s
CUDA_VISIBLE_DEVICES=4,5,6,7 python projects/SPCL/train_net.py --config-file projects/SPCL/configs/DukeMTMC/usl_R50.yml OUTPUT_DIR logs/test/$time/dev-wuyiming/spcl/dukemtmc &
sleep 10s
CUDA_VISIBLE_DEVICES=4,5,6,7 python projects/SPCL/train_net.py --config-file projects/SPCL/configs/MSMT17/usl_R50.yml OUTPUT_DIR logs/test/$time/dev-wuyiming/spcl/msmt17 &