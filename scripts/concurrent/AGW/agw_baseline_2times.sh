###
 # @Author: your name
 # @Date: 2020-11-21 22:42:41
 # @LastEditTime: 2020-11-28 15:54:02
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /fast-reid/scripts/concurrent/agw_baseline_2times.sh
### 
#/bin/env bash
time=$(date +%F)
python projects/ConcurrentReID/train_net.py --config-file projects/ConcurrentReID/configs/Market1501/AGW_R50.yml --num-gpus 4 MODEL.BACKBONE.NORM "syncBN" MODEL.HEADS.NORM "syncBN" OUTPUT_DIR logs/ConcurrentReID/market1501/agw_baseline_2times INPUT.MUTUAL.ENABLED True INPUT.MUTUAL.TIMES 2 CONCURRENT.ENABLED False CONCURRENT.BLOCK_SIZE "(1,2)"

python projects/ConcurrentReID/train_net.py --config-file projects/ConcurrentReID/configs/DukeMTMC/AGW_R50.yml --num-gpus 4 MODEL.BACKBONE.NORM "syncBN" MODEL.HEADS.NORM "syncBN" OUTPUT_DIR logs/ConcurrentReID/dukemtmc/agw_baseline_2times INPUT.MUTUAL.ENABLED True INPUT.MUTUAL.TIMES 2 CONCURRENT.ENABLED False CONCURRENT.BLOCK_SIZE "(1,2)"

python projects/ConcurrentReID/train_net.py --config-file projects/ConcurrentReID/configs/MSMT17/AGW_R50.yml --num-gpus 4 MODEL.BACKBONE.NORM "syncBN" MODEL.HEADS.NORM "syncBN" OUTPUT_DIR logs/ConcurrentReID/msmt17/agw_baseline_2times INPUT.MUTUAL.ENABLED True INPUT.MUTUAL.TIMES 2 TEST.DO_VAL True CONCURRENT.ENABLED False CONCURRENT.BLOCK_SIZE "(1,2)"