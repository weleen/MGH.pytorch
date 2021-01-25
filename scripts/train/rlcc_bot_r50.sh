#!/usr/bin/env bash
echo "Train RLCC with DataParallel"
python projects/RLCC/train_net.py --config-file projects/RLCC/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/RLCC/BoT_R50/market1501
python projects/RLCC/train_net.py --config-file projects/RLCC/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/RLCC/BoT_R50/dukemtmc
python projects/RLCC/train_net.py --config-file projects/RLCC/configs/MSMT17/BoT_R50.yml OUTPUT_DIR logs/RLCC/BoT_R50/msmt17 TEST.DO_VAL True
