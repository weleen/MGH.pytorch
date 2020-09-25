#!/usr/bin/env bash
echo "Train model with gem pooling"
time=$(date +%F)
python tools/train_net.py --config-file configs/Market1501/AGW_R50.yml --num-gpus 4 MODEL.BACKBONE.NORM "syncBN" MODEL.HEADS.NORM "syncBN" MODEL.BACKBONE.WITH_NL False DATASETS.TESTS "('Market1501', 'DukeMTMC', 'MSMT17')" OUTPUT_DIR logs/UDAPretrain/market1501
sleep 10s
python tools/train_net.py --config-file configs/DukeMTMC/AGW_R50.yml --num-gpus 4 MODEL.BACKBONE.NORM "syncBN" MODEL.HEADS.NORM "syncBN" MODEL.BACKBONE.WITH_NL False DATASETS.TESTS "('Market1501', 'DukeMTMC', 'MSMT17')" OUTPUT_DIR logs/UDAPretrain/dukemtmc
sleep 10s
python tools/train_net.py --config-file configs/MSMT17/AGW_R50.yml --num-gpus 4 MODEL.BACKBONE.NORM "syncBN" MODEL.HEADS.NORM "syncBN" MODEL.BACKBONE.WITH_NL False DATASETS.TESTS "('Market1501', 'DukeMTMC', 'MSMT17')" OUTPUT_DIR logs/UDAPretrain/msmt17
