###
 # @Author: WuYiming
 # @Date: 2020-10-26 22:43:13
 # @LastEditTime: 2020-11-16 15:59:41
 # @LastEditors: Please set LastEditors
 # @Description: script for training spcl bot_r50 
 # @FilePath: /fast-reid/scripts/train/spcl_bot_r50_contrast_tripletloss.sh
### 
#!/usr/bin/env bash
echo "SpCL with triplet loss by DataParallel"
time=$(date +%F)
python projects/SpCL_new/train_net.py --config-file projects/SpCL_new/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/SpCL_BoT_R50_contrast_tripletloss/market1501 MODEL.LOSSES.NAME \(\"ContrastiveLoss\",\"TripletLoss\"\) 
# PSEUDO.MEMORY.MOMENTUM 0.
# sleep 10s
# python projects/SpCL_new/train_net.py --config-file projects/SpCL_new/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/SpCL_BoT_R50_contrast_tripletloss/dukemtmc MODEL.LOSSES.NAME \(\"ContrastiveLoss\",\"TripletLoss\"\)
# sleep 10s
# python projects/SpCL_new/train_net.py --config-file projects/SpCL_new/configs/MSMT17/BoT_R50.yml OUTPUT_DIR logs/SpCL_BoT_R50_contrast_tripletloss/msmt17 MODEL.LOSSES.NAME \(\"ContrastiveLoss\",\"TripletLoss\"\)