#!/usr/bin/env bash
echo "Test SPCL project"
python projects/SPCL/train_net.py --config-file projects/SPCL/configs/Market1501/usl_R50.yml --num-gpus 1 OUTPUT_DIR logs/test/spcl/market1501
python projects/SPCL/train_net.py --config-file projects/SPCL/configs/DukeMTMC/usl_R50.yml --num-gpus 1 OUTPUT_DIR logs/test/spcl/dukemtmc
python projects/SPCL/train_net.py --config-file projects/SPCL/configs/MSMT17/usl_R50.yml --num-gpus 1 OUTPUT_DIR logs/test/spcl/msmt17