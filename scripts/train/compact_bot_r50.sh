###
 # @Author: wuyiming
 # @Date: 2020-09-27 16:05:25
 # @LastEditTime: 2020-10-05 00:51:32
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /fast-reid/scripts/train/train_compact.sh
### 
#/bin/env bash
echo "Test supervised learning AGW_R50"
echo "Market1501"
python projects/CompactReID/train_net.py --config-file projects/CompactReID/configs/Market1501/AGW_R50.yml \
                                         --num-gpus 4 \
                                         OUTPUT_DIR logs/CompactReID/market1501/AGW_R50/combination

python projects/CompactReID/train_net.py --config-file projects/CompactReID/configs/Market1501/AGW_R50.yml \
                                         --num-gpus 4 \
                                         OUTPUT_DIR logs/CompactReID/market1501/AGW_R50/pretext \
                                         INPUT.REA.ENABLED False \
                                         MODEL.LOSSES.CE.SCALE 0. \
                                         MODEL.LOSSES.TRI.SCALE 0. \
                                         TEST.EVAL_PERIOD 30

python projects/CompactReID/train_net.py --config-file projects/CompactReID/configs/Market1501/AGW_R50.yml \
                                         --num-gpus 4 \
                                         MODEL.OPEN_LAYERS "['heads', ]" \
                                         MODEL.WEIGHTS logs/CompactReID/market1501/AGW_R50/pretext/model_final.pth \
                                         OUTPUT_DIR logs/CompactReID/market1501/AGW_R50/finetune \
                                         MODEL.LOSSES.NAME "('CrossEntropyLoss', 'TripletLoss')" \
                                         SOLVER.FREEZE_ITERS 60 \
                                         SOLVER.MAX_EPOCH 60 \
                                         SOLVER.STEPS "[30, 50]" \
                                         COMPACT.LOSS_SCALE 0.

echo "DukeMTMC"
python projects/CompactReID/train_net.py --config-file projects/CompactReID/configs/DukeMTMC/AGW_R50.yml \
                                         --num-gpus 4 \
                                         OUTPUT_DIR logs/CompactReID/dukemtmc/AGW_R50/combination

python projects/CompactReID/train_net.py --config-file projects/CompactReID/configs/DukeMTMC/AGW_R50.yml \
                                         --num-gpus 4 \
                                         OUTPUT_DIR logs/CompactReID/dukemtmc/AGW_R50/pretext \
                                         INPUT.REA.ENABLED False \
                                         MODEL.LOSSES.CE.SCALE 0. \
                                         MODEL.LOSSES.TRI.SCALE 0. \
                                         TEST.EVAL_PERIOD 30

python projects/CompactReID/train_net.py --config-file projects/CompactReID/configs/DukeMTMC/AGW_R50.yml \
                                         --num-gpus 4 \
                                         MODEL.OPEN_LAYERS "['heads', ]" \
                                         MODEL.WEIGHTS logs/CompactReID/dukemtmc/AGW_R50/pretext/model_final.pth \
                                         OUTPUT_DIR logs/CompactReID/dukemtmc/AGW_R50/finetune \
                                         MODEL.LOSSES.NAME "('CrossEntropyLoss', 'TripletLoss')" \
                                         SOLVER.FREEZE_ITERS 60 \
                                         SOLVER.MAX_EPOCH 60 \
                                         SOLVER.STEPS "[30, 50]" \
                                         COMPACT.LOSS_SCALE 0.