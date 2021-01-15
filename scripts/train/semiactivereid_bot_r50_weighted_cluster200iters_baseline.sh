###
 # @Author: WuYiming
 # @Date: 2020-10-26 22:43:13
 # @LastEditTime: 2020-11-10 14:01:11
 # @LastEditors: Please set LastEditors
 # @Description: script for training spcl bot_r50 with rectifying labels.
 # @FilePath: /fast-reid/scripts/train/semiactivereid_bot_r50_weighted_cluster200iters_baseline.sh
### 
#!/usr/bin/env bash
echo "weighted contrastive SpCL project with DataParallel"
time=$(date +%F)
python projects/Semi-ActiveReID/train_net.py --config-file projects/Semi-ActiveReID/configs/Market1501/BoT_R50.yml \
                                             OUTPUT_DIR logs/SpCL_wcl_cluster200iters/BoT_R50/market1501 \
                                             PSEUDO.MEMORY.WEIGHTED True \
                                             PSEUDO.CLUSTER_EPOCH 1 \
                                             ACTIVE.RECTIFY False \
                                             ACTIVE.BUILD_DATALOADER False
sleep 10s
python projects/Semi-ActiveReID/train_net.py --config-file projects/Semi-ActiveReID/configs/DukeMTMC/BoT_R50.yml \
                                             OUTPUT_DIR logs/SpCL_wcl_cluster200iters/BoT_R50/dukemtmc \
                                             PSEUDO.MEMORY.WEIGHTED True \
                                             PSEUDO.CLUSTER_EPOCH 1 \
                                             ACTIVE.RECTIFY False \
                                             ACTIVE.BUILD_DATALOADER False
sleep 10s
python projects/Semi-ActiveReID/train_net.py --config-file projects/Semi-ActiveReID/configs/MSMT17/BoT_R50.yml \
                                             OUTPUT_DIR logs/SpCL_wcl_cluster200iters/BoT_R50/msmt17 \
                                             PSEUDO.MEMORY.WEIGHTED True \
                                             PSEUDO.CLUSTER_EPOCH 1 \
                                             ACTIVE.RECTIFY False \
                                             ACTIVE.BUILD_DATALOADER False \
                                             TEST.DO_VAL True