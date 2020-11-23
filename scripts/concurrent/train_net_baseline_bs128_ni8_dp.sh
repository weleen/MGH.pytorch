###
 # @Author: your name
 # @Date: 2020-11-21 22:42:41
 # @LastEditTime: 2020-11-22 00:27:11
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /fast-reid/scripts/concurrent/train_net_baseline_bs128_ni8.sh
### 
#/bin/env bash
echo "Test supervised learning AGW_R50"
time=$(date +%F)
python projects/ConcurrentReID/train_net.py --config-file projects/ConcurrentReID/configs/Market1501/AGW_R50.yml --num-gpus 1 OUTPUT_DIR logs/ConcurrentReID/market1501/baseline_bs128_ni8_dp CONCURRENT.ENABLED False SOLVER.IMS_PER_BATCH 128 DATALOADER.NUM_INSTANCE 8 SOLVER.STEPS [20,45] SOLVER.MAX_ITER 60
python projects/ConcurrentReID/train_net.py --config-file projects/ConcurrentReID/configs/DukeMTMC/AGW_R50.yml --num-gpus 1 OUTPUT_DIR logs/ConcurrentReID/dukemtmc/baseline_bs128_ni8_dp CONCURRENT.ENABLED False SOLVER.IMS_PER_BATCH 128 DATALOADER.NUM_INSTANCE 8 SOLVER.STEPS [20,45] SOLVER.MAX_ITER 60
python projects/ConcurrentReID/train_net.py --config-file projects/ConcurrentReID/configs/MSMT17/AGW_R50.yml --num-gpus 1 logs/ConcurrentReID/msmt17/baseline_bs128_ni8_dp CONCURRENT.ENABLED False SOLVER.IMS_PER_BATCH 128 DATALOADER.NUM_INSTANCE 8 SOLVER.STEPS [20,45] SOLVER.MAX_ITER 60