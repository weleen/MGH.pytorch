#!/usr/bin/env bash
echo "Test SPCL project"
time=$(date +%F)
python projects/SPCL/train_net.py --config-file projects/SPCL/configs/Market1501/usl_R50.yml --num-gpus 4 OUTPUT_DIR logs/test/$time/dev/spcl/market1501 &
sleep 10s
python projects/SPCL/train_net.py --config-file projects/SPCL/configs/DukeMTMC/usl_R50.yml --num-gpus 4 OUTPUT_DIR logs/test/$time/dev/spcl/dukemtmc &
sleep 10s
python projects/SPCL/train_net.py --config-file projects/SPCL/configs/MSMT17/usl_R50.yml --num-gpus 4 OUTPUT_DIR logs/test/$time/dev/spcl/msmt17 &