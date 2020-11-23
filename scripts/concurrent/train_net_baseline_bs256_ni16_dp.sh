###
 # @Author: your name
 # @Date: 2020-11-21 22:42:41
 # @LastEditTime: 2020-11-22 00:30:02
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /fast-reid/scripts/concurrent/train_net_baseline_bs128_ni8.sh
### 
#/bin/env bash
echo "Test supervised learning AGW_R50"
time=$(date +%F)
python projects/ConcurrentReID/train_net.py --config-file projects/ConcurrentReID/configs/Market1501/AGW_R50.yml --num-gpus 1 OUTPUT_DIR logs/ConcurrentReID/market1501/baseline_bs256_ni16_dp CONCURRENT.ENABLED False SOLVER.IMS_PER_BATCH 256 DATALOADER.NUM_INSTANCE 16 SOLVER.STEPS [10,23] SOLVER.MAX_ITER 30
python projects/ConcurrentReID/train_net.py --config-file projects/ConcurrentReID/configs/DukeMTMC/AGW_R50.yml --num-gpus 1 OUTPUT_DIR logs/ConcurrentReID/dukemtmc/baseline_bs256_ni16_dp CONCURRENT.ENABLED False SOLVER.IMS_PER_BATCH 256 DATALOADER.NUM_INSTANCE 16 SOLVER.STEPS [10,23] SOLVER.MAX_ITER 30
python projects/ConcurrentReID/train_net.py --config-file projects/ConcurrentReID/configs/MSMT17/AGW_R50.yml --num-gpus 1 OUTPUT_DIR logs/ConcurrentReID/msmt17/baseline_bs256_ni16_dp CONCURRENT.ENABLED False SOLVER.IMS_PER_BATCH 256 DATALOADER.NUM_INSTANCE 16 SOLVER.STEPS [10,23] SOLVER.MAX_ITER 30