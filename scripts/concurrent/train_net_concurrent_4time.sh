###
 # @Author: your name
 # @Date: 2020-11-21 00:29:14
 # @LastEditTime: 2020-11-21 08:17:31
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /fast-reid/scripts/concurrent/train_net_concurrent_4time.sh
### 
#/bin/env bash
echo "Test supervised learning AGW_R50"
time=$(date +%F)
python projects/ConcurrentReID/train_net.py --config-file projects/ConcurrentReID/configs/Market1501/AGW_R50.yml --num-gpus 1 OUTPUT_DIR logs/ConcurrentReID/market1501/concurrent_mutual4times INPUT.REA.ENABLED True INPUT.MUTUAL.ENABLED True INPUT.MUTUAL.TIMES 4 CONCURRENT.ENABLED True CONCURRENT.BLOCK_SIZE "(2,2)"
python projects/ConcurrentReID/train_net.py --config-file projects/ConcurrentReID/configs/DukeMTMC/AGW_R50.yml --num-gpus 1 OUTPUT_DIR logs/ConcurrentReID/dukemtmc/concurrent_mutual4times INPUT.REA.ENABLED True INPUT.MUTUAL.ENABLED True INPUT.MUTUAL.TIMES 4 CONCURRENT.ENABLED True CONCURRENT.BLOCK_SIZE "(2,2)"
python projects/ConcurrentReID/train_net.py --config-file projects/ConcurrentReID/configs/MSMT17/AGW_R50.yml --num-gpus 1 OUTPUT_DIR logs/ConcurrentReID/msmt17/concurrent_mutual4times INPUT.REA.ENABLED True INPUT.MUTUAL.ENABLED True TEST.DO_VAL True INPUT.MUTUAL.TIMES 4 CONCURRENT.ENABLED True CONCURRENT.BLOCK_SIZE "(2,2)"