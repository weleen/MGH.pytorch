###
 # @Author: WuYiming
 # @Date: 2020-09-30 10:35:58
 # @LastEditTime: 2020-10-27 10:59:20
 # @LastEditors: Please set LastEditors
 # @Description: script for training SpCL
 # @FilePath: /git/fast-reid/scripts/unittest/spcl.sh
### 
#!/usr/bin/env bash
echo "Test masked weighted contrastive SpCL project with DataParallel"
time=$(date +%F)
python projects/Semi-ActiveReID/train_net.py --config-file projects/Semi-ActiveReID/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/SpCL_wcl/BoT_R50/market1501 PSEUDO.MEMORY.WEIGHTED True ACTIVE.RECTIFY False ACTIVE.BUILD_DATALOADER False
#sleep 10s
#python projects/Semi-ActiveReID/train_net.py --config-file projects/Semi-ActiveReID/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/SpCL_wcl/BoT_R50/dukemtmc PSEUDO.MEMORY.WEIGHTED True ACTIVE.RECTIFY False ACTIVE.BUILD_DATALOADER False
#sleep 10s
#python projects/Semi-ActiveReID/train_net.py --config-file projects/Semi-ActiveReID/configs/MSMT17/BoT_R50.yml OUTPUT_DIR logs/SpCL_wcl/BoT_R50/msmt17 PSEUDO.MEMORY.WEIGHTED True ACTIVE.RECTIFY False ACTIVE.BUILD_DATALOADER False TEST.DO_VAL True 
