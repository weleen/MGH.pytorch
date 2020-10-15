###
 # @Author: wuyiming
 # @Date: 2020-10-11 13:45:17
 # @LastEditTime: 2020-10-13 10:06:01
 # @LastEditors: Please set LastEditors
 # @Description: script for training BoT_R50 models
 # @FilePath: /git/fast-reid/scripts/unittest/BoT_R50_4gpus.sh
### 
#!/bin/env bash
echo "Test supervised learning BoT_R50"
time=$(date +%F)
python tools/train_net.py --config-file configs/Market1501/bagtricks_R50.yml --num-gpus 4 MODEL.BACKBONE.NORM "syncBN" MODEL.HEADS.NORM "syncBN" OUTPUT_DIR logs/test/BoT_R50/market1501-4gpus &
sleep 10s
python tools/train_net.py --config-file configs/DukeMTMC/bagtricks_R50.yml --num-gpus 4 MODEL.BACKBONE.NORM "syncBN" MODEL.HEADS.NORM "syncBN"OUTPUT_DIR logs/test/BoT_R50/dukemtmc-4gpus &
sleep 10s
python tools/train_net.py --config-file configs/MSMT17/bagtricks_R50.yml --num-gpus 4 MODEL.BACKBONE.NORM "syncBN" MODEL.HEADS.NORM "syncBN"OUTPUT_DIR logs/test/BoT_R50/msmt17-4gpus TEST.DO_VAL True &
