#!/bin/bash
echo "Run CAP"
CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501 &
CUDA_VISIBLE_DEVICES=1 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc &
CUDA_VISIBLE_DEVICES=2 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/MSMT17/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/msmt17 TEST.DO_VAL True &

CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml PSEUDO.NAME "hypergraph" OUTPUT_DIR logs/CAP/BoT_R50_hg/market1501 &
CUDA_VISIBLE_DEVICES=1 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml PSEUDO.NAME "hypergraph" OUTPUT_DIR logs/CAP/BoT_R50_hg/dukemtmc &
CUDA_VISIBLE_DEVICES=2 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/MSMT17/BoT_R50.yml PSEUDO.NAME "hypergraph" OUTPUT_DIR logs/CAP/BoT_R50_hg/msmt17 TEST.DO_VAL True &

CUDA_VISIBLE_DEVICES=1 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50_smt/market1501 MODEL.LOSSES.NAME "(
\"CrossEntropyLoss\", \"HardViewContrastiveLoss\", \"CameraAwareLoss\")" MODEL.MEAN_NET True MODEL.LOSSES.CE.START_EPOCH 5 MODEL.LOSSES.CE.SCALE 0.5 MODEL.LOSSES.VCL.START_EPOCH 5 MODEL.LOSSES.VCL.SCALE 0.5 &
CUDA_VISIBLE_DEVICES=2 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50_smt/dukemtmc MODEL.LOSSES.NAME "(
\"CrossEntropyLoss\", \"HardViewContrastiveLoss\", \"CameraAwareLoss\")" MODEL.MEAN_NET True MODEL.LOSSES.CE.START_EPOCH 5 MODEL.LOSSES.CE.SCALE 0.5 MODEL.LOSSES.VCL.START_EPOCH 5 MODEL.LOSSES.VCL.SCALE 0.5 &
CUDA_VISIBLE_DEVICES=3 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/MSMT17/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50_smt/msmt17 TEST.DO_VAL True MODEL.LOSSES.NAME "(
\"CrossEntropyLoss\", \"HardViewContrastiveLoss\", \"CameraAwareLoss\")" MODEL.MEAN_NET True MODEL.LOSSES.CE.START_EPOCH 5 MODEL.LOSSES.CE.SCALE 0.5 MODEL.LOSSES.VCL.START_EPOCH 5 MODEL.LOSSES.VCL.SCALE 0.5 &