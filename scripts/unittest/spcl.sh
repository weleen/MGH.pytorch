###
 # @Author: your name
 # @Date: 2020-09-30 10:35:58
 # @LastEditTime: 2020-10-12 16:20:48
 # @LastEditors: your name
 # @Description: In User Settings Edit
 # @FilePath: /git/fast-reid/scripts/unittest/spcl.sh
### 
#!/usr/bin/env bash
echo "Test SPCL project"
time=$(date +%F)
python projects/SPCL/train_net.py --config-file projects/SPCL/configs/Market1501/usl_R50.yml OUTPUT_DIR logs/test/$time/dev/spcl/market1501
sleep 10s
python projects/SPCL/train_net.py --config-file projects/SPCL/configs/DukeMTMC/usl_R50.yml OUTPUT_DIR logs/test/$time/dev/spcl/dukemtmc
sleep 10s
python projects/SPCL/train_net.py --config-file projects/SPCL/configs/MSMT17/usl_R50.yml OUTPUT_DIR logs/test/$time/dev/spcl/msmt17