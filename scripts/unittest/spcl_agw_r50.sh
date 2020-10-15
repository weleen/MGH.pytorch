###
 # @Author: WuYiming
 # @Date: 2020-09-30 10:35:58
 # @LastEditTime: 2020-10-14 16:46:54
 # @LastEditors: Please set LastEditors
 # @Description: script for training SpCL
 # @FilePath: /git/fast-reid/scripts/unittest/spcl.sh
### 
#!/usr/bin/env bash
echo "Test SpCL project"
time=$(date +%F)
python projects/SpCL_new/train_net.py --config-file configs/SpCL/Market1501/AGW_R50.yml --num-gpus 4 OUTPUT_DIR logs/test/$time/SpCL/market1501
sleep 10s
python projects/SpCL_new/train_net.py --config-file configs/SpCL/DukeMTMC/AGW_R50.yml --num-gpus 4 OUTPUT_DIR logs/test/$time/SpCL/dukemtmc
sleep 10s
python projects/SpCL_new/train_net.py --config-file configs/SpCL/MSMT17/AGW_R50.yml  --num-gpus 4 OUTPUT_DIR logs/test/$time/SpCL/msmt17