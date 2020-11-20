###
 # @Author: your name
 # @Date: 2020-11-20 09:46:37
 # @LastEditTime: 2020-11-20 09:53:58
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /fast-reid/scripts/concurrent/train_net.sh
### 
#/bin/env bash
echo "Test supervised learning AGW_R50"
time=$(date +%F)
python projects/ConcurrentReID/train_net.py --config-file projects/ConcurrentReID/configs/Market1501/AGW_R50.yml --num-gpus 4 MODEL.BACKBONE.NORM "syncBN" MODEL.HEADS.NORM "syncBN" OUTPUT_DIR logs/ConcurrentReID/market1501/baseline_mutual4times INPUT.REA.ENABLED True INPUT.MUTUAL.ENABLED True TEST.DO_VAL True INPUT.MUTUAL.TIMES 4
python projects/ConcurrentReID/train_net.py --config-file projects/ConcurrentReID/configs/DukeMTMC/AGW_R50.yml --num-gpus 4 MODEL.BACKBONE.NORM "syncBN" MODEL.HEADS.NORM "syncBN" OUTPUT_DIR logs/ConcurrentReID/dukemtmc/baseline_mutual4times INPUT.REA.ENABLED True INPUT.MUTUAL.ENABLED True TEST.DO_VAL True INPUT.MUTUAL.TIMES 4
python projects/ConcurrentReID/train_net.py --config-file projects/ConcurrentReID/configs/MSMT17/AGW_R50.yml --num-gpus 4 MODEL.BACKBONE.NORM "syncBN" MODEL.HEADS.NORM "syncBN" OUTPUT_DIR logs/ConcurrentReID/msmt17/baseline_mutual4times INPUT.REA.ENABLED True INPUT.MUTUAL.ENABLED True TEST.DO_VAL True INPUT.MUTUAL.TIMES 4