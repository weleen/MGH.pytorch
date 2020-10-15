###
 # @Author: your name
 # @Date: 2020-10-14 17:13:25
 # @LastEditTime: 2020-10-14 17:14:07
 # @LastEditors: Please set LastEditors
 # @Description: script for training BoT with gem pooling, HARD_MINING is set to False
 # @FilePath: /fast-reid/scripts/unittest/BoT_R50_gempooling.sh
### 
#!/bin/env bash
echo "Test supervised learning BoT_R50 with gem pooling"
time=$(date +%F)
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/Market1501/bagtricks_R50.yml OUTPUT_DIR logs/test/BoT_R50_gem/market1501 TEST.DO_VAL True MODEL.HEADS.POOL_LAYER "gempoolP" MODEL.LOSSES.TRI.HARD_MINING False &
sleep 10s
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/DukeMTMC/bagtricks_R50.yml OUTPUT_DIR logs/test/BoT_R50_gem/dukemtmc TEST.DO_VAL True MODEL.HEADS.POOL_LAYER "gempoolP" MODEL.LOSSES.TRI.HARD_MINING False &
sleep 10s
CUDA_VISIBLE_DEVICES=2 python tools/train_net.py --config-file configs/MSMT17/bagtricks_R50.yml OUTPUT_DIR logs/test/BoT_R50_gem/msmt17 TEST.DO_VAL True MODEL.HEADS.POOL_LAYER "gempoolP" MODEL.LOSSES.TRI.HARD_MINING False &