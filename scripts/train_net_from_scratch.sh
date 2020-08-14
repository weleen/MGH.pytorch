#!/usr/bin/env bash
# enviornment for socket
# please use ifconfig to check local network name
python tools/train_net.py --config-file configs/VERIWild/AGW_R50.yml \
                          --num-gpus 4 \
                          MODEL.HEADS.CLS_LAYER "arcface" \
                          MODEL.HEADS.NECK_FEAT "after" \
                          MODEL.LOSSES.NAME "('CrossEntropyLoss',)" \
                          MODEL.HEADS.MARGIN 0.35 \
                          MODEL.HEADS.SCALE 64 \
                          DATALOADER.PK_SAMPLER False \
                          SOLVER.WARMUP_ITERS 0 \
                          SOLVER.OPT "SGD" \
                          SOLVER.BASE_LR 0.0003 \
                          SOLVER.IMS_PER_BATCH 128 \
                          SOLVER.MAX_ITER 42 \
                          SOLVER.STEPS [20,40] \
                          SOLVER.CHECKPOINT_PERIOD 10 \
                          TEST.EVAL_PERIOD 10 \
                          TEST.IMS_PER_BATCH 256 \
                          OUTPUT_DIR logs/image/veriwild/agw_R50_twostage/1stage

python tools/train_net.py --config-file configs/VERIWild/AGW_R50.yml \
                          --num-gpus 4 \
                          MODEL.HEADS.CLS_LAYER "arcface" \
                          MODEL.HEADS.NECK_FEAT "after" \
                          MODEL.LOSSES.NAME "('CrossEntropyLoss',)" \
                          MODEL.HEADS.MARGIN 0.35 \
                          MODEL.HEADS.SCALE 64 \
                          MODEL.WEIGHTS "logs/image/veriwild/agw_R50_twostage/1stage/model_final.pth" \
                          DATALOADER.NUM_INSTANCE 8 \
                          SOLVER.WARMUP_ITERS 0 \
                          SOLVER.OPT "SGD" \
                          SOLVER.BASE_LR 0.000003 \
                          SOLVER.IMS_PER_BATCH 128 \
                          SOLVER.MAX_ITER 4 \
                          SOLVER.CHECKPOINT_PERIOD 2 \
                          TEST.EVAL_PERIOD 2 \
                          TEST.IMS_PER_BATCH 256 \
                          OUTPUT_DIR logs/image/veriwild/agw_R50_twostage/2stage
#CONFIG=$1
#GPU=$2
#PY_ARGS=${@：3}
#python --config-file configs/MSMT17/AGW_R50.yml \
#        MODEL.HEADS.CLS_LAYER "arcface" \
#        SOLVER.WARMUP_ITERS 0 \
#        MODEL.LOSSES.NAME "('CrossEntropyLoss',)" \
#        MODEL.HEADS.MARGIN 0.35 \
#        MODEL.HEADS.SCALE 64 \
#        DATALOADER.PK_SAMPLER False \
#        OUTPUT_DIR logs/image/msmt17/agw_R50_2stage/1stages
#
#python --config-file configs/MSMT17/AGW_R50.yml \
#        MODEL.BACKBONE.PRETRAIN_PATH "logs/image/msmt17/agw_R50_2stage/1stage/model_final.pth" \
#        MODEL.HEADS.CLS_LAYER "arcface" \
#        SOLVER.WARMUP_ITERS 0 \
#        SOLVER.BASE_LR
#        MODEL.HEADS.MARGIN 0.35 \
#        MODEL.HEADS.SCALE 64 \
#        DATALOADER.PK_SAMPLER False \
#        OUTPUT_DIR logs/image/msmt17/agw_R50_2stage/2stage

#python ./tools/train_net.py --config-file $CONFIG \
#                            --num-gpus $GPU \
#                            $(PY_ARGS)