###
 # @Author: WuYiming
 # @Date: 2020-09-30 10:35:58
 # @LastEditTime: 2020-10-14 16:46:54
 # @LastEditors: Please set LastEditors
 # @Description: script for training SpCL
 # @FilePath: /git/fast-reid/scripts/unittest/spcl.sh
### 
#!/usr/bin/env bash
echo "Test SpCL project with DataParallel"
time=$(date +%F)
python projects/SpCL_new/train_net.py --config-file projects/SpCL_new/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/test/$time/SpCL_new/BoT_R50/market1501
python projects/SpCL_new/train_net.py --config-file projects/SpCL_new/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/test/$time/SpCL_new/BoT_R50/dukemtmc
python projects/SpCL_new/train_net.py --config-file projects/SpCL_new/configs/MSMT17/BoT_R50.yml OUTPUT_DIR logs/test/$time/SpCL_new/BoT_R50/msmt17
