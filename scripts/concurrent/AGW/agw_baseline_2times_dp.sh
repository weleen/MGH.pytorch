###
 # @Author: your name
 # @Date: 2020-11-21 22:42:41
 # @LastEditTime: 2020-11-27 20:05:32
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /fast-reid/scripts/concurrent/agw_baseline_2times_dp.sh
### 
#/bin/env bash
time=$(date +%F)
python projects/ConcurrentReID/train_net.py --config-file projects/ConcurrentReID/configs/Market1501/AGW_R50.yml --num-gpus 1 OUTPUT_DIR logs/ConcurrentReID/market1501/agw_baseline_2times_dp INPUT.MUTUAL.ENABLED True INPUT.MUTUAL.TIMES 2 CONCURRENT.ENABLED False CONCURRENT.BLOCK_SIZE "(1,2)"

python projects/ConcurrentReID/train_net.py --config-file projects/ConcurrentReID/configs/DukeMTMC/AGW_R50.yml --num-gpus 1 OUTPUT_DIR logs/ConcurrentReID/dukemtmc/agw_baseline_2times_dp INPUT.MUTUAL.ENABLED True INPUT.MUTUAL.TIMES 2 CONCURRENT.ENABLED False CONCURRENT.BLOCK_SIZE "(1,2)"

python projects/ConcurrentReID/train_net.py --config-file projects/ConcurrentReID/configs/MSMT17/AGW_R50.yml --num-gpus 1 OUTPUT_DIR logs/ConcurrentReID/msmt17/agw_baseline_2times_dp INPUT.MUTUAL.ENABLED True INPUT.MUTUAL.TIMES 2 TEST.DO_VAL True CONCURRENT.ENABLED False CONCURRENT.BLOCK_SIZE "(1,2)"