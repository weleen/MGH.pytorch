###
 # @Author: WuYiming
 # @Date: 2020-10-26 22:43:13
 # @LastEditTime: 2020-11-06 16:44:53
 # @LastEditors: Please set LastEditors
 # @Description: script for training spcl bot_r50 with rectifying labels.
 # @FilePath: /fast-reid/scripts/train/spcl_bot_r50_weighted_rectify.sh
### 
#!/usr/bin/env bash
echo "Semi-ActiveReID project with DataParallel"
time=$(date +%F)
python projects/Semi-ActiveReID/train_net.py --config-file projects/Semi-ActiveReID/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/SemiActive_SpCL_Contrast_Rectify/BoT_R50/market1501 PSEUDO.MEMORY.WEIGHTED False ACTIVE.RECTIFY True ACTIVE.BUILD_DATALOADER False ACTIVE.EDGE_PROP True ACTIVE.NODE_PROP False ACTIVE.START_ITER 20
# sleep 10s
# python projects/Semi-ActiveReID/train_net.py --config-file projects/Semi-ActiveReID/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/SemiActive_SpCL_Contrast_Rectify/BoT_R50/dukemtmc PSEUDO.MEMORY.WEIGHTED False ACTIVE.RECTIFY True ACTIVE.BUILD_DATALOADER False ACTIVE.EDGE_PROP True ACTIVE.NODE_PROP False ACTIVE.START_ITER 20
# sleep 10s
# python projects/Semi-ActiveReID/train_net.py --config-file projects/Semi-ActiveReID/configs/MSMT17/BoT_R50.yml OUTPUT_DIR logs/SemiActive_SpCL_Contrast_Rectify/BoT_R50/msmt17 PSEUDO.MEMORY.WEIGHTED False TEST.DO_VAL True ACTIVE.RECTIFY True ACTIVE.BUILD_DATALOADER False ACTIVE.EDGE_PROP True ACTIVE.NODE_PROP False