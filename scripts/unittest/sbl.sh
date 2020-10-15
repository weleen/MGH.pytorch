###
 # @Author: WuYiming
 # @Date: 2020-09-30 10:35:58
 # @LastEditTime: 2020-10-14 14:45:47
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /git/fast-reid/scripts/unittest/sbl.sh
### 
#!/usr/bin/env bash
echo "Test SBL project"
time=$(date +%F)
python tools/train_net.py --config-file configs/SBL/Market1501/BoT_R50.yml --num-gpus 4 OUTPUT_DIR logs/test/$time/SBL/market1501
sleep 10s
python tools/train_net.py --config-file configs/SBL/DukeMTMC/BoT_R50.yml --num-gpus 4 OUTPUT_DIR logs/test/$time/SBL/dukemtmc
sleep 10s
python tools/train_net.py --config-file configs/SBL/MSMT17/BoT_R50.yml --num-gpus 4 OUTPUT_DIR logs/test/$time/SBL/msmt17 TEST.DO_VAL True