###
 # @Author: your name
 # @Date: 2020-11-21 22:42:41
 # @LastEditTime: 2020-12-01 11:22:33
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /fast-reid/scripts/concurrent/agw_baseline_frozen_bn_bs256_ni16.sh
### 
#/bin/env bash
time=$(date +%F)
python projects/ConcurrentReID/train_net.py --config-file projects/ConcurrentReID/configs/Market1501/AGW_R50.yml --num-gpus 2 MODEL.BACKBONE.NORM "FrozenBN" MODEL.HEADS.NORM "FrozenBN" OUTPUT_DIR logs/ConcurrentReID/market1501/agw_baseline_fronzen_bn_bs256_ni16 SOLVER.IMS_PER_BATCH 256 DATALOADER.NUM_INSTANCE 16 SOLVER.STEPS [160,360] SOLVER.MAX_EPOCH 480 CONCURRENT.ENABLED False

python projects/ConcurrentReID/train_net.py --config-file projects/ConcurrentReID/configs/DukeMTMC/AGW_R50.yml --num-gpus 2 MODEL.BACKBONE.NORM "FrozenBN" MODEL.HEADS.NORM "FrozenBN" OUTPUT_DIR logs/ConcurrentReID/dukemtmc/agw_baseline_fronzen_bn_bs256_ni16 SOLVER.IMS_PER_BATCH 256 DATALOADER.NUM_INSTANCE 16 SOLVER.STEPS [160,360] SOLVER.MAX_EPOCH 480 CONCURRENT.ENABLED False

python projects/ConcurrentReID/train_net.py --config-file projects/ConcurrentReID/configs/MSMT17/AGW_R50.yml --num-gpus 4 MODEL.BACKBONE.NORM "FrozenBN" MODEL.HEADS.NORM "FrozenBN" OUTPUT_DIR logs/ConcurrentReID/msmt17/agw_baseline_fronzen_bn_bs256_ni16 SOLVER.IMS_PER_BATCH 256 DATALOADER.NUM_INSTANCE 16 SOLVER.STEPS [160,360] SOLVER.MAX_EPOCH 480 TEST.DO_VAL True CONCURRENT.ENABLED False
