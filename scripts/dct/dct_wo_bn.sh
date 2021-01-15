###
 # @Author: your name
 # @Date: 2020-12-08 16:37:12
 # @LastEditTime: 2020-12-08 16:38:15
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /fast-reid/scripts/dct/dct_wo_bn.sh
### 
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/Market1501/DCT_R50.yml MODEL.HEADS.WITH_BNNECK False OUTPUT_DIR logs/DCT_R50/market1501_without_bn
sleep 10s
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/DukeMTMC/DCT_R50.yml MODEL.HEADS.WITH_BNNECK False OUTPUT_DIR logs/DCT_R50/dukemtmc_without_bn
sleep 10s
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/MSMT17/DCT_R50.yml TEST.DO_VAL True MODEL.HEADS.WITH_BNNECK False OUTPUT_DIR logs/DCT_R50/msmt17_without_bn
