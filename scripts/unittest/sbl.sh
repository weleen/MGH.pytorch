#!/usr/bin/env bash
echo "Test SBL project"
time=$(date +%F)
python tools/train_net.py --config-file configs/SBL/Market1501/AGW_R50.yml OUTPUT_DIR logs/test/$time/dev/sbl/market1501
sleep 10s
python tools/train_net.py --config-file configs/SBL/DukeMTMC/AGW_R50.yml OUTPUT_DIR logs/test/$time/dev/sbl/dukemtmc
sleep 10s
python tools/train_net.py --config-file configs/SBL/MSMT17/AGW_R50.yml OUTPUT_DIR logs/test/$time/dev/sbl/msmt17