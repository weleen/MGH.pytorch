#!/bin/env bash
echo "Run supervised learning BoT_R50 with only cross entropy loss"
time=$(date +%F)
python tools/train_net.py --config-file configs/Market1501/bagtricks_R50.yml OUTPUT_DIR logs/test/BoT_R50/market1501 MODEL.LOSSES.NAME "(\"CrossEntropyLoss\",)"
python tools/train_net.py --config-file configs/DukeMTMC/bagtricks_R50.yml OUTPUT_DIR logs/test/BoT_R50/dukemtmc MODEL.LOSSES.NAME "(\"CrossEntropyLoss\",)"
python tools/train_net.py --config-file configs/MSMT17/bagtricks_R50.yml OUTPUT_DIR logs/test/BoT_R50/msmt17 TEST.DO_VAL True MODEL.LOSSES.NAME "(\"CrossEntropyLoss\",)"