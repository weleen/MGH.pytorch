###
 # @Author: your name
 # @Date: 2020-11-21 00:53:42
 # @LastEditTime: 2020-11-25 21:35:05
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /fast-reid/scripts/concurrent/agw_concurrent_6time.sh
### 
#/bin/env bash
time=$(date +%F)
python projects/ConcurrentReID/train_net.py --config-file projects/ConcurrentReID/configs/Market1501/AGW_R50.yml --num-gpus 4 MODEL.BACKBONE.NORM "syncBN" MODEL.HEADS.NORM "syncBN" OUTPUT_DIR logs/ConcurrentReID/market1501/agw_concurrent_6times INPUT.MUTUAL.ENABLED True INPUT.MUTUAL.TIMES 6 CONCURRENT.ENABLED True CONCURRENT.BLOCK_SIZE "(2,3)"

python projects/ConcurrentReID/train_net.py --config-file projects/ConcurrentReID/configs/DukeMTMC/AGW_R50.yml --num-gpus 4 MODEL.BACKBONE.NORM "syncBN" MODEL.HEADS.NORM "syncBN" OUTPUT_DIR logs/ConcurrentReID/dukemtmc/agw_concurrent_6times INPUT.MUTUAL.ENABLED True INPUT.MUTUAL.TIMES 6 CONCURRENT.ENABLED True CONCURRENT.BLOCK_SIZE "(2,3)"

python projects/ConcurrentReID/train_net.py --config-file projects/ConcurrentReID/configs/MSMT17/AGW_R50.yml --num-gpus 4 MODEL.BACKBONE.NORM "syncBN" MODEL.HEADS.NORM "syncBN" OUTPUT_DIR logs/ConcurrentReID/msmt17/agw_concurrent_6times INPUT.MUTUAL.ENABLED True INPUT.MUTUAL.TIMES 6 TEST.DO_VAL True CONCURRENT.ENABLED True CONCURRENT.BLOCK_SIZE "(2,3)"