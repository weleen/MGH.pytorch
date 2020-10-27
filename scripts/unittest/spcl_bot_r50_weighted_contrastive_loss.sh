###
 # @Author: WuYiming
 # @Date: 2020-09-30 10:35:58
 # @LastEditTime: 2020-10-26 15:35:39
 # @LastEditors: Please set LastEditors
 # @Description: script for training SpCL
 # @FilePath: /git/fast-reid/scripts/unittest/spcl.sh
### 
#!/usr/bin/env bash
echo "Test SpCL project with DataParallel"
time=$(date +%F)
python projects/ActiveReID/train_net.py --config-file projects/ActiveReID/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/test/$time/WeightedContrast/BoT_R50/market1501 ACTIVE.RECTIFY False PSEUDO.MEMORY.WEIGHTED True
sleep 10s
python projects/ActiveReID/train_net.py --config-file projects/ActiveReID/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/test/$time/WeightedContrast/BoT_R50/dukemtmc ACTIVE.RECTIFY False PSEUDO.MEMORY.WEIGHTED True
sleep 10s
python projects/ActiveReID/train_net.py --config-file projects/ActiveReID/configs/MSMT17/BoT_R50.yml OUTPUT_DIR logs/test/$time/WeightedContrast/BoT_R50/msmt17 ACTIVE.RECTIFY False PSEUDO.MEMORY.WEIGHTED True