###
 # @Author: WuYiming
 # @Date: 2020-10-26 22:43:13
 # @LastEditTime: 2020-10-27 19:18:43
 # @LastEditors: Please set LastEditors
 # @Description: script for training spcl bot_r50 with rectifying labels.
 # @FilePath: /fast-reid/scripts/train/spcl_bot_r50_weighted_rectify.sh
### 
#!/usr/bin/env bash
echo "Masked weighted contrastive ActiveReID project with DataParallel"
time=$(date +%F)
python projects/ActiveReID/train_net.py --config-file projects/ActiveReID/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/Active_SpCL_MaskedWeightedContrast_Rectify/BoT_R50/market1501 PSEUDO.MEMORY.WEIGHTED True
sleep 10s
python projects/ActiveReID/train_net.py --config-file projects/ActiveReID/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/Active_SpCL_MaskedWeightedContrast_Rectify/BoT_R50/dukemtmc PSEUDO.MEMORY.WEIGHTED True
sleep 10s
python projects/ActiveReID/train_net.py --config-file projects/ActiveReID/configs/MSMT17/BoT_R50.yml OUTPUT_DIR logs/Active_SpCL_MaskedWeightedContrast_Rectify/BoT_R50/msmt17 PSEUDO.MEMORY.WEIGHTED True