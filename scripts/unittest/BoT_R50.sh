###
 # @Author: wuyiming
 # @Date: 2020-10-11 13:45:17
 # @LastEditTime: 2020-10-12 19:53:12
 # @LastEditors: Please set LastEditors
 # @Description: script for training BoT_R50 models
 # @FilePath: /git/fast-reid/scripts/unittest/BoT_R50.sh
### 
#!/bin/env bash
echo "Test supervised learning BoT_R50"
time=$(date +%F)
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/Market1501/bagtricks_R50.yml OUTPUT_DIR logs/test/BoT_R50/market1501 &
sleep 10s
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/DukeMTMC/bagtricks_R50.yml OUTPUT_DIR logs/test/BoT_R50/dukemtmc &
sleep 10s
CUDA_VISIBLE_DEVICES=2 python tools/train_net.py --config-file configs/MSMT17/bagtricks_R50.yml OUTPUT_DIR logs/test/BoT_R50/msmt17 TEST.DO_VAL True &