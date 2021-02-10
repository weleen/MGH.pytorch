#!/bin/bash
echo "Run ICE"
python projects/SMT/train_net.py --num-gpus 4 --config-file projects/SMT/configs/Market1501/BoT_R50_ice.yml OUTPUT_DIR logs/SMT/BoT_R50_ice/market1501
python projects/SMT/train_net.py --num-gpus 4 --config-file projects/SMT/configs/DukeMTMC/BoT_R50_ice.yml OUTPUT_DIR logs/SMT/BoT_R50_ice/dukemtmc
python projects/SMT/train_net.py --num-gpus 4 --config-file projects/SMT/configs/MSMT17/BoT_R50_ice.yml OUTPUT_DIR logs/SMT/BoT_R50_ice/msmt17 TEST.DO_VAL True