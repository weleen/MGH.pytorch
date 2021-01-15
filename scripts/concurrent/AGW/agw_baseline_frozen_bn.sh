###
 # @Author: your name
 # @Date: 2020-11-21 22:42:41
 # @LastEditTime: 2020-11-25 21:21:20
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /fast-reid/scripts/concurrent/agw_baseline_frozen_bn.sh
### 
#/bin/env bash
time=$(date +%F)
python projects/ConcurrentReID/train_net.py --config-file projects/ConcurrentReID/configs/Market1501/AGW_R50.yml --num-gpus 4 MODEL.BACKBONE.NORM "FrozenBN" MODEL.HEADS.NORM "FrozenBN" OUTPUT_DIR logs/ConcurrentReID/market1501/agw_baseline_fronzen_bn SOLVER.IMS_PER_BATCH 64 DATALOADER.NUM_INSTANCE 4 CONCURRENT.ENABLED False

python projects/ConcurrentReID/train_net.py --config-file projects/ConcurrentReID/configs/DukeMTMC/AGW_R50.yml --num-gpus 4 MODEL.BACKBONE.NORM "FrozenBN" MODEL.HEADS.NORM "FrozenBN" OUTPUT_DIR logs/ConcurrentReID/dukemtmc/agw_baseline_fronzen_bn SOLVER.IMS_PER_BATCH 64 DATALOADER.NUM_INSTANCE 4 CONCURRENT.ENABLED False

python projects/ConcurrentReID/train_net.py --config-file projects/ConcurrentReID/configs/MSMT17/AGW_R50.yml --num-gpus 4 MODEL.BACKBONE.NORM "FrozenBN" MODEL.HEADS.NORM "FrozenBN" OUTPUT_DIR logs/ConcurrentReID/msmt17/agw_baseline_fronzen_bn SOLVER.IMS_PER_BATCH 64 DATALOADER.NUM_INSTANCE 4 TEST.DO_VAL True CONCURRENT.ENABLED False