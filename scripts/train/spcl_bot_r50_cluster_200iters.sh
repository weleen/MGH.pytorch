###
 # @Author: WuYiming
 # @Date: 2020-10-26 22:43:13
 # @LastEditTime: 2020-11-06 13:03:12
 # @LastEditors: Please set LastEditors
 # @Description: script for training spcl bot_r50 with rectifying labels.
 # @FilePath: /fast-reid/scripts/train/spcl_bot_r50_weighted_rectify.sh
### 
#!/usr/bin/env bash
echo "weighted contrastive SpCL project with DataParallel"
time=$(date +%F)
python projects/ActiveReID/train_net.py --config-file projects/ActiveReID/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/SpCL_cluster200iters/BoT_R50/market1501 ACTIVE.RECTIFY False PSEUDO.MEMORY.WEIGHTED False PSEUDO.CLUSTER_EPOCH 200
sleep 10s
python projects/ActiveReID/train_net.py --config-file projects/ActiveReID/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/SpCL_cluster200iters/BoT_R50/dukemtmc ACTIVE.RECTIFY False PSEUDO.MEMORY.WEIGHTED False PSEUDO.CLUSTER_EPOCH 200
sleep 10s
python projects/ActiveReID/train_net.py --config-file projects/ActiveReID/configs/MSMT17/BoT_R50.yml OUTPUT_DIR logs/SpCL_cluster200iters/BoT_R50/msmt17 ACTIVE.RECTIFY False PSEUDO.MEMORY.WEIGHTED False PSEUDO.CLUSTER_EPOCH 2