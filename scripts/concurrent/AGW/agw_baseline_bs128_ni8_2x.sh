###
 # @Author: your name
 # @Date: 2020-11-21 22:42:41
 # @LastEditTime: 2020-12-02 14:32:40
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /fast-reid/scripts/concurrent/agw_baseline_bs128_ni8.sh
### 
#/bin/env bash
time=$(date +%F)
python projects/ConcurrentReID/train_net.py --config-file projects/ConcurrentReID/configs/Market1501/AGW_R50.yml --num-gpus 4 MODEL.BACKBONE.NORM "syncBN" MODEL.HEADS.NORM "syncBN" OUTPUT_DIR logs/ConcurrentReID/market1501/agw_baseline_bs128_ni8_2x SOLVER.IMS_PER_BATCH 128 DATALOADER.NUM_INSTANCE 8 SOLVER.STEPS [80,180] SOLVER.MAX_EPOCH 240 CONCURRENT.ENABLED False

python projects/ConcurrentReID/train_net.py --config-file projects/ConcurrentReID/configs/DukeMTMC/AGW_R50.yml --num-gpus 4 MODEL.BACKBONE.NORM "syncBN" MODEL.HEADS.NORM "syncBN" OUTPUT_DIR logs/ConcurrentReID/dukemtmc/agw_baseline_bs128_ni8_2x SOLVER.IMS_PER_BATCH 128 DATALOADER.NUM_INSTANCE 8 SOLVER.STEPS [80,180] SOLVER.MAX_EPOCH 240 CONCURRENT.ENABLED False

python projects/ConcurrentReID/train_net.py --config-file projects/ConcurrentReID/configs/MSMT17/AGW_R50.yml --num-gpus 4 MODEL.BACKBONE.NORM "syncBN" MODEL.HEADS.NORM "syncBN" OUTPUT_DIR logs/ConcurrentReID/msmt17/agw_baseline_bs128_ni8_2x SOLVER.IMS_PER_BATCH 128 DATALOADER.NUM_INSTANCE 8 SOLVER.STEPS [80,180] SOLVER.MAX_EPOCH 240 TEST.DO_VAL True CONCURRENT.ENABLED False
