#!/bin/bash
echo "Run CAP"
# baseline
CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501 &
CUDA_VISIBLE_DEVICES=1 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc &
CUDA_VISIBLE_DEVICES=2 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/MSMT17/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/msmt17 TEST.DO_VAL True &
# bs64
python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50_bs64_4gpus/market1501 SOLVER.IMS_PER_BATCH 64 &
python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50_bs64_4gpus/dukemtmc SOLVER.IMS_PER_BATCH 64 &
python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/MSMT17/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50_bs64_4gpus/msmt17 TEST.DO_VAL True SOLVER.IMS_PER_BATCH 64 &
# distillation
CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50_w_vcl/market1501 MODEL.LOSSES.NAME "(\"HardViewContrastiveLoss\", \"CameraAwareLoss\")" MODEL.MEAN_NET True MODEL.LOSSES.VCL.START_EPOCH 5 MODEL.LOSSES.VCL.SCALE 0.5 CAP.LOSS_WEIGHT 0. CAP.INTERCAM_EPOCH 50 &
CUDA_VISIBLE_DEVICES=1 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50_w_vcl/dukemtmc MODEL.LOSSES.NAME "(\"HardViewContrastiveLoss\", \"CameraAwareLoss\")" MODEL.MEAN_NET True MODEL.LOSSES.VCL.START_EPOCH 5 MODEL.LOSSES.VCL.SCALE 0.5 CAP.LOSS_WEIGHT 0. CAP.INTERCAM_EPOCH 50 &
CUDA_VISIBLE_DEVICES=2 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/MSMT17/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50_w_vcl/msmt17 TEST.DO_VAL True MODEL.LOSSES.NAME "(\"HardViewContrastiveLoss\", \"CameraAwareLoss\")" MODEL.MEAN_NET True MODEL.LOSSES.VCL.START_EPOCH 5 MODEL.LOSSES.VCL.SCALE 0.5 CAP.LOSS_WEIGHT 0. CAP.INTERCAM_EPOCH 50
# hypergraph
CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml PSEUDO.NAME "hypergraph" OUTPUT_DIR logs/CAP/BoT_R50_hg/market1501 &
CUDA_VISIBLE_DEVICES=1 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml PSEUDO.NAME "hypergraph" OUTPUT_DIR logs/CAP/BoT_R50_hg/dukemtmc &
CUDA_VISIBLE_DEVICES=2 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/MSMT17/BoT_R50.yml PSEUDO.NAME "hypergraph" OUTPUT_DIR logs/CAP/BoT_R50_hg/msmt17 TEST.DO_VAL True &

CUDA_VISIBLE_DEVICES=1 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50_smt/market1501 MODEL.LOSSES.NAME "(
\"CrossEntropyLoss\", \"HardViewContrastiveLoss\", \"CameraAwareLoss\")" MODEL.MEAN_NET True MODEL.LOSSES.CE.START_EPOCH 5 MODEL.LOSSES.CE.SCALE 0.5 MODEL.LOSSES.VCL.START_EPOCH 5 MODEL.LOSSES.VCL.SCALE 0.5 &
CUDA_VISIBLE_DEVICES=2 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50_smt/dukemtmc MODEL.LOSSES.NAME "(
\"CrossEntropyLoss\", \"HardViewContrastiveLoss\", \"CameraAwareLoss\")" MODEL.MEAN_NET True MODEL.LOSSES.CE.START_EPOCH 5 MODEL.LOSSES.CE.SCALE 0.5 MODEL.LOSSES.VCL.START_EPOCH 5 MODEL.LOSSES.VCL.SCALE 0.5 &
CUDA_VISIBLE_DEVICES=3 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/MSMT17/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50_smt/msmt17 TEST.DO_VAL True MODEL.LOSSES.NAME "(
\"CrossEntropyLoss\", \"HardViewContrastiveLoss\", \"CameraAwareLoss\")" MODEL.MEAN_NET True MODEL.LOSSES.CE.START_EPOCH 5 MODEL.LOSSES.CE.SCALE 0.5 MODEL.LOSSES.VCL.START_EPOCH 5 MODEL.LOSSES.VCL.SCALE 0.5 &
# 2-21
# tianjian
# market
CUDA_VISIBLE_DEVICES=0 python projects/CAP_tianjian/train_net.py --num-gpus 1 --config-file projects/CAP_tianjian/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/CAP_tianjian/BoT_R50_single_gpu_2-21/market1501 &
CUDA_VISIBLE_DEVICES=1 python projects/CAP_tianjian/train_net.py --num-gpus 1 --config-file projects/CAP_tianjian/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/CAP_tianjian/BoT_R50_single_gpu_2-21/market1501_rho PSEUDO.DBSCAN.BASE 'rho' PSEUDO.DBSCAN.RHO 2.2e-3 &
CUDA_VISIBLE_DEVICES=2 python projects/CAP_tianjian/train_net.py --num-gpus 1 --config-file projects/CAP_tianjian/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/CAP_tianjian/BoT_R50_single_gpu_2-21/market1501_steps50 SOLVER.STEPS [50,] &
CUDA_VISIBLE_DEVICES=3 python projects/CAP_tianjian/train_net.py --num-gpus 1 --config-file projects/CAP_tianjian/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/CAP_tianjian/BoT_R50_single_gpu_2-21/market1501_proxysampler5 PSEUDO.START_EPOCH 5 &
# dukemtmc
CUDA_VISIBLE_DEVICES=0 python projects/CAP_tianjian/train_net.py --num-gpus 1 --config-file projects/CAP_tianjian/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/CAP_tianjian/BoT_R50_single_gpu_2-21/dukemtmc &
CUDA_VISIBLE_DEVICES=1 python projects/CAP_tianjian/train_net.py --num-gpus 1 --config-file projects/CAP_tianjian/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/CAP_tianjian/BoT_R50_single_gpu_2-21/dukemtmc_rho PSEUDO.DBSCAN.BASE 'rho' PSEUDO.DBSCAN.RHO 2.2e-3 &
CUDA_VISIBLE_DEVICES=2 python projects/CAP_tianjian/train_net.py --num-gpus 1 --config-file projects/CAP_tianjian/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/CAP_tianjian/BoT_R50_single_gpu_2-21/dukemtmc_steps50 SOLVER.STEPS [50,] &
CUDA_VISIBLE_DEVICES=3 python projects/CAP_tianjian/train_net.py --num-gpus 1 --config-file projects/CAP_tianjian/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/CAP_tianjian/BoT_R50_single_gpu_2-21/dukemtmc_proxysampler5 PSEUDO.START_EPOCH 5 &
# msmt
CUDA_VISIBLE_DEVICES=3 python projects/CAP_tianjian/train_net.py --num-gpus 1 --config-file projects/CAP_tianjian/configs/MSMT17/BoT_R50.yml OUTPUT_DIR logs/CAP/CAP_tianjian/BoT_R50_single_gpu_2-21/msmt17 TEST.DO_VAL True &
CUDA_VISIBLE_DEVICES=3 python projects/CAP_tianjian/train_net.py --num-gpus 1 --config-file projects/CAP_tianjian/configs/MSMT17/BoT_R50.yml OUTPUT_DIR logs/CAP/CAP_tianjian/BoT_R50_single_gpu_rho_2-21/msmt17 TEST.DO_VAL True PSEUDO.DBSCAN.BASE 'rho' PSEUDO.DBSCAN.RHO 2.2e-3 &
# syncbn
python projects/CAP_tianjian/train_net.py --num-gpus 4 --config-file projects/CAP_tianjian/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/CAP_tianjian/BoT_R50_4_gpu_2-21/market1501 MODEL.BACKBONE.NORM 'syncBN' MODEL.HEADS.NORM 'syncBN' &
python projects/CAP_tianjian/train_net.py --num-gpus 4 --config-file projects/CAP_tianjian/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/CAP_tianjian/BoT_R50_4_gpu_2-21/dukemtmc MODEL.BACKBONE.NORM 'syncBN' MODEL.HEADS.NORM 'syncBN' &
python projects/CAP_tianjian/train_net.py --num-gpus 4 --config-file projects/CAP_tianjian/configs/MSMT17/BoT_R50.yml OUTPUT_DIR logs/CAP/CAP_tianjian/BoT_R50_4_gpu_2-21/msmt17 TEST.DO_VAL True MODEL.BACKBONE.NORM 'syncBN' MODEL.HEADS.NORM 'syncBN' &
# 2-22
# with spatial temporal information
CUDA_VISIBLE_DEVICES=2 python projects/CAP_tianjian/train_net.py --num-gpus 1 --config-file projects/CAP_tianjian/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/CAP_tianjian/BoT_R50_single_gpu_2-22/market1501_dbscan_st PSEUDO.NAME 'dbscan_st'

# hypergraph
CUDA_VISIBLE_DEVICES=3 python projects/CAP_tianjian/train_net.py --num-gpus 1 --config-file projects/CAP_tianjian/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/CAP_tianjian/BoT_R50_single_gpu_2-22/market1501_hg PSEUDO.NAME 'hypergraph'

# upperbound
CUDA_VISIBLE_DEVICES=3 python projects/CAP_tianjian/train_net.py --num-gpus 1 --config-file projects/CAP_tianjian/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/CAP_tianjian/BoT_R50_single_gpu_2-22/market1501_upperbound PSEUDO.TRUE_LABEL True

# 2-23
# cap_tianjian
CUDA_VISIBLE_DEVICES=0 python projects/CAP_tianjian/train_net.py --num-gpus 1 --config-file projects/CAP_tianjian/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/CAP_tianjian/BoT_R50_single_gpu_2-23/market1501_warmup10_steps30 SOLVER.STEPS [20,] &
CUDA_VISIBLE_DEVICES=1 python projects/CAP_tianjian/train_net.py --num-gpus 1 --config-file projects/CAP_tianjian/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/CAP_tianjian/BoT_R50_single_gpu_2-23/market1501_warmup10_steps20_40 &
CUDA_VISIBLE_DEVICES=2 python projects/CAP_tianjian/train_net.py --num-gpus 1 --config-file projects/CAP_tianjian/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/CAP_tianjian/BoT_R50_single_gpu_2-23/market1501_rho_warmup10_steps20_40 PSEUDO.DBSCAN.BASE 'rho' PSEUDO.DBSCAN.RHO 2.2e-3 &
CUDA_VISIBLE_DEVICES=4 python projects/CAP_tianjian/train_net.py --num-gpus 1 --config-file projects/CAP_tianjian/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/CAP_tianjian/BoT_R50_single_gpu_2-23/dukemtmc_warmup10_steps30 SOLVER.STEPS [20,]  &
CUDA_VISIBLE_DEVICES=5 python projects/CAP_tianjian/train_net.py --num-gpus 1 --config-file projects/CAP_tianjian/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/CAP_tianjian/BoT_R50_single_gpu_2-23/dukemtmc_warmup10_steps20_40 &
CUDA_VISIBLE_DEVICES=6 python projects/CAP_tianjian/train_net.py --num-gpus 1 --config-file projects/CAP_tianjian/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/CAP_tianjian/BoT_R50_single_gpu_2-23/dukemtmc_rho_warmup10_steps20_40 PSEUDO.DBSCAN.BASE 'rho' PSEUDO.DBSCAN.RHO 2.2e-3 &

# 2-24
CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_warmup0_steps20-40_dbscan_cam PSEUDO.NAME 'dbscan_cam'
CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_warmup0_steps20-40
CUDA_VISIBLE_DEVICES=1 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_warmup0_steps20-40_camera_metric PSEUDO.CAMERA_CLUSTER_METRIC True &
CUDA_VISIBLE_DEVICES=2 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_warmup0_steps20-40_dbscan_cam_camera_metric PSEUDO.CAMERA_CLUSTER_METRIC True PSEUDO.NAME 'dbscan_cam'
CUDA_VISIBLE_DEVICES=3 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_rho1.6e-3_warmup0_steps20-40
CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_warmup0_steps20-40 &
CUDA_VISIBLE_DEVICES=1 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_warmup0_steps20-40
CUDA_VISIBLE_DEVICES=1 python projects/CAP_tianjian/train_net.py --num-gpus 1 --config-file projects/CAP_tianjian/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP_tianjian/BoT_R50/dukemtmc_warmup0_steps20_40 SOLVER.WARMUP_EPOCHS 0 SOLVER.STEPS [20,40] &
CUDA_VISIBLE_DEVICES=2 python projects/CAP_tianjian/train_net.py --num-gpus 1 --config-file projects/CAP_tianjian/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP_tianjian/BoT_R50/dukemtmc_rho_warmup0_steps20_40 SOLVER.WARMUP_EPOCHS 0 SOLVER.STEPS [20,40] PSEUDO.DBSCAN.BASE 'rho' PSEUDO.DBSCAN.RHO 2.2e-3 &
CUDA_VISIBLE_DEVICES=2 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_rho1.6e-3_warmup0_steps20-40_dbscan_st PSEUDO.NAME 'dbscan_st' &
CUDA_VISIBLE_DEVICES=3 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_rho2.2e-3_warmup0_steps20-40_dbscan_st PSEUDO.NAME 'dbscan_st' PSEUDO.DBSCAN.BASE 'rho' PSEUDO.DBSCAN.RHO 2.2e-3 &
CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_warmup0_steps20-40_dbscan_cam_single_gpu PSEUDO.NAME 'dbscan_cam'

# 2-25
CUDA_VISIBLE_DEVICES=1 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_rho2.2e-3_warmup0_steps20-40 PSEUDO.DBSCAN.BASE 'rho' PSEUDO.DBSCAN.RHO 2.2e-3