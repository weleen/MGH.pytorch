###
 # @Author: WuYiming
 # @Date: 2020-09-30 10:35:58
 # @LastEditTime: 2020-10-25 12:57:38
 # @LastEditors: Please set LastEditors
 # @Description: script for training SpCL
 # @FilePath: /git/fast-reid/scripts/unittest/spcl.sh
### 
#!/usr/bin/env bash
echo "Test SpCL project with DistributedDataParallel and fp16"
time=$(date +%F)
python projects/SpCL_new/train_net.py --config-file projects/SpCL_new/configs/Market1501/BoT_R50.yml --num-gpus 4 OUTPUT_DIR logs/test/$time/SpCL_new_BoT_R50_4gpus_fp16/market1501 SOLVER.FP16_ENABLED True
sleep 10s
python projects/SpCL_new/train_net.py --config-file projects/SpCL_new/configs/DukeMTMC/BoT_R50.yml --num-gpus 4 OUTPUT_DIR logs/test/$time/SpCL_new_BoT_R50_4gpus_fp16/dukemtmc SOLVER.FP16_ENABLED True 
sleep 10s
python projects/SpCL_new/train_net.py --config-file projects/SpCL_new/configs/MSMT17/BoT_R50.yml --num-gpus 4 OUTPUT_DIR logs/test/$time/SpCL_new_BoT_R50_4gpus_fp16/msmt17 SOLVER.FP16_ENABLED True TEST.DO_VAL True
