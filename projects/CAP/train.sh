CUDA_VISIBLE_DEVICES=3 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501 CAP.INSTANCE_LOSS = False

CUDA_VISIBLE_DEVICES=3 python projects/CAP/train_net.py --eval-only --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/debug MODEL.WEIGHTS /home/wuyiming/disk50/project/reid/logs/CAP/BoT_R50/market1501/model_final.pth CAP.ST_TEST True

CUDA_VISIBLE_DEVICES=3 python projects/CAP/train_net.py --eval-only --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/debug MODEL.WEIGHTS /home/wuyiming/disk50/project/reid/logs/CAP/BoT_R50/dukemtmc/model_final.pth CAP.ST_TEST True

CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --eval-only --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/debug MODEL.WEIGHTS /home/wuyiming/disk50/project/reid/logs/CAP/BoT_R50/market1501/model_0019999.pth CAP.ST_TEST True

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
# ce + intra-camera + inter-camera
CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_ce+cameraloss MODEL.LOSSES.NAME "(\"CrossEntropyLoss\", \"CameraAwareLoss\")" &
CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_warmup5_steps20_40_with_classifier SOLVER.WARMUP_EPOCHS 5 SOLVER.STEPS [15,35]
CUDA_VISIBLE_DEVICES=1 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_warmup0_steps20-40_weight PSEUDO.MEMORY.WEIGHTED True CAP.WEIGHTED_INTRA True CAP.WEIGHTED_INTER True &
CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_warmup0_steps20-40_weight PSEUDO.MEMORY.WEIGHTED True CAP.WEIGHTED_INTRA True CAP.WEIGHTED_INTER True 
CUDA_VISIBLE_DEVICES=2 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_rho1.6e-3_warmup0_steps20-40_dbscan_st PSEUDO.NAME 'dbscan_st' PSEUDO.DBSCAN.BASE 'rho' PSEUDO.DBSCAN.RHO 1.6e-3
CUDA_VISIBLE_DEVICES=1 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_warmup0_steps20-40_weight_intra PSEUDO.MEMORY.WEIGHTED True CAP.WEIGHTED_INTRA True &
CUDA_VISIBLE_DEVICES=3 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_warmup0_steps20-40_weight_inter PSEUDO.MEMORY.WEIGHTED True CAP.WEIGHTED_INTER True &

# 2-26
# gem pooling
CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/AGW_R50.yml OUTPUT_DIR logs/CAP/AGW_R50/market1501 &
CUDA_VISIBLE_DEVICES=2 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/AGW_R50.yml OUTPUT_DIR logs/CAP/AGW_R50/dukemtmc &
CUDA_VISIBLE_DEVICES=1 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_warmup0_steps20-40_weight_intra PSEUDO.MEMORY.WEIGHTED True CAP.WEIGHTED_INTRA True
CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/AGW_R50.yml OUTPUT_DIR logs/CAP/AGW_R50/market1501_k1=20 PSEUDO.DBSCAN.K1 20 &
CUDA_VISIBLE_DEVICES=3 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/AGW_R50.yml OUTPUT_DIR logs/CAP/AGW_R50/dukemtmc_k1=20 PSEUDO.DBSCAN.K1 20 &
CUDA_VISIBLE_DEVICES=1 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_k1=20 PSEUDO.DBSCAN.K1 20 &
CUDA_VISIBLE_DEVICES=2 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_k1=20 PSEUDO.DBSCAN.K1 20 &

# 2-27
# no hard
CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_k1_20_no_hard CAP.ENABLE_HARD_NEG False &
CUDA_VISIBLE_DEVICES=1 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_k1_20_weighted_intra_no_hard PSEUDO.MEMORY.WEIGHTED True CAP.WEIGHTED_INTRA True CAP.ENABLE_HARD_NEG False &
CUDA_VISIBLE_DEVICES=2 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_k1_20_weighted_inter_no_hard PSEUDO.MEMORY.WEIGHTED True CAP.WEIGHTED_INTER True CAP.ENABLE_HARD_NEG False &
CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_k1_20_weighted_no_hard PSEUDO.MEMORY.WEIGHTED True CAP.WEIGHTED_INTRA True CAP.WEIGHTED_INTER True CAP.ENABLE_HARD_NEG False &
CUDA_VISIBLE_DEVICES=1 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_k1_20_weighted_no_hard_rho1.6e-3 PSEUDO.MEMORY.WEIGHTED True CAP.WEIGHTED_INTRA True CAP.WEIGHTED_INTER True CAP.ENABLE_HARD_NEG False PSEUDO.DBSCAN.BASE 'rho' PSEUDO.DBSCAN.RHO 1.6e-3 &
CUDA_VISIBLE_DEVICES=2 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_k1_20_weighted_no_hard_rho1.6e-3_dbscan_st PSEUDO.MEMORY.WEIGHTED True CAP.WEIGHTED_INTRA True CAP.WEIGHTED_INTER True CAP.ENABLE_HARD_NEG False PSEUDO.NAME 'dbscan_st' PSEUDO.DBSCAN.BASE 'rho' PSEUDO.DBSCAN.RHO 1.6e-3 &

CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_k1_20_no_hard CAP.ENABLE_HARD_NEG False &
CUDA_VISIBLE_DEVICES=1 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_k1_20_weighted_intra_no_hard PSEUDO.MEMORY.WEIGHTED True CAP.WEIGHTED_INTRA True CAP.ENABLE_HARD_NEG False &
CUDA_VISIBLE_DEVICES=2 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_k1_20_weighted_inter_no_hard PSEUDO.MEMORY.WEIGHTED True CAP.WEIGHTED_INTER True CAP.ENABLE_HARD_NEG False &
CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_k1_20_weighted_no_hard PSEUDO.MEMORY.WEIGHTED True CAP.WEIGHTED_INTRA True CAP.WEIGHTED_INTER True CAP.ENABLE_HARD_NEG False &
CUDA_VISIBLE_DEVICES=1 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_k1_20_weighted_no_hard_rho1.6e-3 PSEUDO.MEMORY.WEIGHTED True CAP.WEIGHTED_INTRA True CAP.WEIGHTED_INTER True CAP.ENABLE_HARD_NEG False PSEUDO.DBSCAN.BASE 'rho' PSEUDO.DBSCAN.RHO 1.6e-3 &
CUDA_VISIBLE_DEVICES=2 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_k1_20_weighted_no_hard_rho1.6e-3_dbscan_st PSEUDO.MEMORY.WEIGHTED True CAP.WEIGHTED_INTRA True CAP.WEIGHTED_INTER True CAP.ENABLE_HARD_NEG False PSEUDO.NAME 'dbscan_st' PSEUDO.DBSCAN.BASE 'rho' PSEUDO.DBSCAN.RHO 1.6e-3 &

# hard
CUDA_VISIBLE_DEVICES=3 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_k1_20_hard &
CUDA_VISIBLE_DEVICES=3 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_k1_20_hard &

# 3-1
CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_k1_30_with_classifier PSEUDO.DBSCAN.K1 30 PSEUDO.WITH_CLASSIFIER True &
CUDA_VISIBLE_DEVICES=1 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_k1_30_with_classifier_weight_intra PSEUDO.MEMORY.WEIGHTED True CAP.WEIGHTED_INTRA True PSEUDO.DBSCAN.K1 30 PSEUDO.WITH_CLASSIFIER True &
CUDA_VISIBLE_DEVICES=2 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_k1_20_with_classifier_weight_intra PSEUDO.MEMORY.WEIGHTED True CAP.WEIGHTED_INTRA True PSEUDO.DBSCAN.K1 20 PSEUDO.WITH_CLASSIFIER True &
CUDA_VISIBLE_DEVICES=3 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_k1_20_with_classifier_rho1.6e-3_weight_intra PSEUDO.MEMORY.WEIGHTED True CAP.WEIGHTED_INTRA True PSEUDO.DBSCAN.K1 20 PSEUDO.WITH_CLASSIFIER True  PSEUDO.DBSCAN.BASE 'rho' PSEUDO.DBSCAN.RHO 1.6e-3 &
CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_k1_30_with_classifier_with_outliers PSEUDO.DBSCAN.K1 30 PSEUDO.WITH_CLASSIFIER True PSEUDO.USE_OUTLIERS True &
CUDA_VISIBLE_DEVICES=1 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_k1_30_with_classifier_with_outliers PSEUDO.DBSCAN.K1 30 PSEUDO.WITH_CLASSIFIER True PSEUDO.USE_OUTLIERS True &
CUDA_VISIBLE_DEVICES=2 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_k1_20_with_classifier_with_outliers PSEUDO.WITH_CLASSIFIER True PSEUDO.USE_OUTLIERS True &
CUDA_VISIBLE_DEVICES=3 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_k1_20_with_classifier_with_outliers PSEUDO.WITH_CLASSIFIER True PSEUDO.USE_OUTLIERS True &
CUDA_VISIBLE_DEVICES=4 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_k1_20_with_classifier_with_outliers_weighted_intra PSEUDO.MEMORY.WEIGHTED True CAP.WEIGHTED_INTRA True PSEUDO.WITH_CLASSIFIER True PSEUDO.USE_OUTLIERS True &
CUDA_VISIBLE_DEVICES=5 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_k1_20_with_classifier_with_outliers_weighted_intra PSEUDO.MEMORY.WEIGHTED True CAP.WEIGHTED_INTRA True PSEUDO.WITH_CLASSIFIER True PSEUDO.USE_OUTLIERS True &
CUDA_VISIBLE_DEVICES=6 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_k1_30_with_classifier_with_outliers_weighted_intra PSEUDO.MEMORY.WEIGHTED True CAP.WEIGHTED_INTRA True PSEUDO.DBSCAN.K1 30 PSEUDO.WITH_CLASSIFIER True PSEUDO.USE_OUTLIERS True &
CUDA_VISIBLE_DEVICES=7 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_k1_30_with_classifier_with_outliers_weighted_intra PSEUDO.MEMORY.WEIGHTED True CAP.WEIGHTED_INTRA True PSEUDO.DBSCAN.K1 30 PSEUDO.WITH_CLASSIFIER True PSEUDO.USE_OUTLIERS True &

# 3-2
CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_k1_30_with_classifier_with_outliers_ProxyBalancedSampler PSEUDO.DBSCAN.K1 30 PSEUDO.WITH_CLASSIFIER True PSEUDO.USE_OUTLIERS True &
CUDA_VISIBLE_DEVICES=1 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_k1_30_with_classifier_with_outliers_ProxyBalancedSampler PSEUDO.DBSCAN.K1 30 PSEUDO.WITH_CLASSIFIER True PSEUDO.USE_OUTLIERS True &
CUDA_VISIBLE_DEVICES=2 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_k1_20_with_classifier_with_outliers_ProxyBalancedSampler PSEUDO.WITH_CLASSIFIER True PSEUDO.USE_OUTLIERS True &
CUDA_VISIBLE_DEVICES=3 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_k1_20_with_classifier_with_outliers_ProxyBalancedSampler PSEUDO.WITH_CLASSIFIER True PSEUDO.USE_OUTLIERS True &
CUDA_VISIBLE_DEVICES=4 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_k1_20_with_classifier_with_outliers_weighted_intra_ProxyBalancedSampler PSEUDO.MEMORY.WEIGHTED True CAP.WEIGHTED_INTRA True PSEUDO.WITH_CLASSIFIER True PSEUDO.USE_OUTLIERS True &
CUDA_VISIBLE_DEVICES=5 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_k1_20_with_classifier_with_outliers_weighted_intra_ProxyBalancedSampler PSEUDO.MEMORY.WEIGHTED True CAP.WEIGHTED_INTRA True PSEUDO.WITH_CLASSIFIER True PSEUDO.USE_OUTLIERS True &
CUDA_VISIBLE_DEVICES=6 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_k1_30_with_classifier_with_outliers_weighted_intra_ProxyBalancedSampler PSEUDO.MEMORY.WEIGHTED True CAP.WEIGHTED_INTRA True PSEUDO.DBSCAN.K1 30 PSEUDO.WITH_CLASSIFIER True PSEUDO.USE_OUTLIERS True &
CUDA_VISIBLE_DEVICES=7 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_k1_30_with_classifier_with_outliers_weighted_intra_ProxyBalancedSampler PSEUDO.MEMORY.WEIGHTED True CAP.WEIGHTED_INTRA True PSEUDO.DBSCAN.K1 30 PSEUDO.WITH_CLASSIFIER True PSEUDO.USE_OUTLIERS True &
CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_k1=20 PSEUDO.DBSCAN.K1 20 &
CUDA_VISIBLE_DEVICES=1 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_k1_20_without_classifier PSEUDO.DBSCAN.K1 20 PSEUDO.WITH_CLASSIFIER False &
CUDA_VISIBLE_DEVICES=1 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/debug/cap_market1501 PSEUDO.DBSCAN.K1 30 PSEUDO.WITH_CLASSIFIER True &

# 3-3
CUDA_VISIBLE_DEVICES=1 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/debug/cap_market1501_k1_20_ PSEUDO.DBSCAN.K1 20 PSEUDO.WITH_CLASSIFIER True &

# 3-4
## market
CUDA_VISIBLE_DEVICES=2 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_base_identity CAP.LOSS_IDENTITY.START_EPOCH 0 CAP.LOSS_IDENTITY.SCALE 1.0 CAP.LOSS_CAMERA.START_EPOCH 100 CAP.LOSS_INSTANCE.START_EPOCH 100 &
CUDA_VISIBLE_DEVICES=3 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_base_camera CAP.LOSS_IDENTITY.START_EPOCH 100 CAP.LOSS_INSTANCE.START_EPOCH 100 &
CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_base_identity_camera CAP.LOSS_INSTANCE.START_EPOCH 100 &
CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_base_identity_camera_instance &
CUDA_VISIBLE_DEVICES=1 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_hg PSEUDO.NAME "dbscan_hg_lp" CAP.LOSS_INSTANCE.START_EPOCH 100 &
CUDA_VISIBLE_DEVICES=2 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_rho1.6e-3_hg PSEUDO.NAME "dbscan_hg_lp" PSEUDO.DBSCAN.BASE 'rho' PSEUDO.DBSCAN.RHO 1.6e-3 CAP.LOSS_INSTANCE.START_EPOCH 100 &
CUDA_VISIBLE_DEVICES=3 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_hg_weighted_intra PSEUDO.NAME "dbscan_hg_lp" CAP.LOSS_CAMERA.WEIGHTED True CAP.LOSS_INSTANCE.START_EPOCH 100 &
CUDA_VISIBLE_DEVICES=4 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_rho1.6e-3_hg_weighted_intra PSEUDO.NAME "dbscan_hg_lp" CAP.LOSS_CAMERA.WEIGHTED True PSEUDO.DBSCAN.BASE 'rho' PSEUDO.DBSCAN.RHO 1.6e-3 CAP.LOSS_INSTANCE.START_EPOCH 100 &
CUDA_VISIBLE_DEVICES=5 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_hg_weighted_intra_listwise_loss PSEUDO.NAME "dbscan_hg_lp" CAP.LOSS_CAMERA.WEIGHTED True &
CUDA_VISIBLE_DEVICES=6 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_weighted_intra_listwise_loss CAP.LOSS_CAMERA.WEIGHTED True &
CUDA_VISIBLE_DEVICES=6 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_weighted_intra_listwise_loss_temp_0.1 CAP.LOSS_CAMERA.WEIGHTED True CAP.LOSS_INSTANCE.TEMP 0.1 &
CUDA_VISIBLE_DEVICES=7 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_rho1.6e-3_hg_weighted_intra_listwise_loss PSEUDO.NAME "dbscan_hg_lp" CAP.LOSS_CAMERA.WEIGHTED True PSEUDO.DBSCAN.BASE 'rho' PSEUDO.DBSCAN.RHO 1.6e-3 &

CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_base_1.0identity_camera_0.1instance CAP.LOSS_IDENTITY.SCALE 1.0 &
CUDA_VISIBLE_DEVICES=1 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_base_0.5identity_camera CAP.LOSS_IDENTITY.SCALE 0.5 CAP.LOSS_INSTANCE.START_EPOCH 100 &

CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_base_1.0identity0_camera_0.1instance CAP.LOSS_IDENTITY.START_EPOCH 0 CAP.LOSS_IDENTITY.SCALE 1.0 &
CUDA_VISIBLE_DEVICES=1 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_base_1.0identity0_camera_1.0instance CAP.LOSS_IDENTITY.START_EPOCH 0 CAP.LOSS_IDENTITY.SCALE 1.0 CAP.LOSS_INSTANCE.SCALE 1.0 &
CUDA_VISIBLE_DEVICES=2 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_hg_st PSEUDO.NAME "dbscan_hg_lp" CAP.LOSS_INSTANCE.START_EPOCH 100 PSEUDO.HG.ST True &
CUDA_VISIBLE_DEVICES=3 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_hg_st_weighted_camera PSEUDO.NAME "dbscan_hg_lp" CAP.LOSS_CAMERA.WEIGHTED True CAP.LOSS_INSTANCE.START_EPOCH 100 PSEUDO.HG.ST True &

## dukemtmc
CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_base_identity_camera_instance &

# 3-5
CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_rho1.6e-3_hg_st PSEUDO.NAME "dbscan_hg_lp" PSEUDO.DBSCAN.BASE 'rho' PSEUDO.DBSCAN.RHO 1.6e-3 CAP.LOSS_INSTANCE.START_EPOCH 100 PSEUDO.HG.ST True &
CUDA_VISIBLE_DEVICES=1 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_rho1.6e-3_hg_st_weighted_camera PSEUDO.NAME "dbscan_hg_lp" CAP.LOSS_CAMERA.WEIGHTED True PSEUDO.DBSCAN.BASE 'rho' PSEUDO.DBSCAN.RHO 1.6e-3 CAP.LOSS_INSTANCE.START_EPOCH 100 PSEUDO.HG.ST True &
CUDA_VISIBLE_DEVICES=2 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_rho1.6e-3_hg_st_weighted_camera_instance PSEUDO.NAME "dbscan_hg_lp" CAP.LOSS_CAMERA.WEIGHTED True PSEUDO.DBSCAN.BASE 'rho' PSEUDO.DBSCAN.RHO 1.6e-3 PSEUDO.HG.ST True &
CUDA_VISIBLE_DEVICES=3 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_hg_st_weighted_camera_instance PSEUDO.NAME "dbscan_hg_lp" CAP.LOSS_CAMERA.WEIGHTED True PSEUDO.HG.ST True &

# market
# ablation study
CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_base_1.0identity5_camera CAP.LOSS_IDENTITY.START_EPOCH 5 CAP.LOSS_IDENTITY.SCALE 1.0 CAP.LOSS_INSTANCE.START_EPOCH 100 &
CUDA_VISIBLE_DEVICES=1 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_base_1.0identity0_camera_1.0instance5 CAP.LOSS_IDENTITY.START_EPOCH 0 CAP.LOSS_IDENTITY.SCALE 1.0 CAP.LOSS_INSTANCE.START_EPOCH 5 CAP.LOSS_INSTANCE.SCALE 1.0 &
CUDA_VISIBLE_DEVICES=2 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501_base_0.5identity5_camera_1.0instance5 CAP.LOSS_IDENTITY.START_EPOCH 5 CAP.LOSS_IDENTITY.SCALE 0.5 CAP.LOSS_INSTANCE.START_EPOCH 5 CAP.LOSS_INSTANCE.SCALE 1.0 &


# next
## dukemtmc
CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_base_identity_camera_instance &
CUDA_VISIBLE_DEVICES=1 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_hg PSEUDO.NAME "dbscan_hg_lp" CAP.LOSS_INSTANCE.START_EPOCH 100 &
CUDA_VISIBLE_DEVICES=2 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_rho1.6e-3_hg PSEUDO.NAME "dbscan_hg_lp" PSEUDO.DBSCAN.BASE 'rho' PSEUDO.DBSCAN.RHO 1.6e-3 CAP.LOSS_INSTANCE.START_EPOCH 100 &
CUDA_VISIBLE_DEVICES=3 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_hg_weighted_intra PSEUDO.NAME "dbscan_hg_lp" CAP.LOSS_CAMERA.WEIGHTED True CAP.LOSS_INSTANCE.START_EPOCH 100 &
CUDA_VISIBLE_DEVICES=4 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_rho1.6e-3_hg_weighted_intra PSEUDO.NAME "dbscan_hg_lp" CAP.LOSS_CAMERA.WEIGHTED True PSEUDO.DBSCAN.BASE 'rho' PSEUDO.DBSCAN.RHO 1.6e-3 CAP.LOSS_INSTANCE.START_EPOCH 100 &
CUDA_VISIBLE_DEVICES=5 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_hg_weighted_intra_listwise_loss PSEUDO.NAME "dbscan_hg_lp" CAP.LOSS_CAMERA.WEIGHTED True &
CUDA_VISIBLE_DEVICES=6 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_weighted_intra_listwise_loss CAP.LOSS_CAMERA.WEIGHTED True &
CUDA_VISIBLE_DEVICES=6 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_weighted_intra_listwise_loss_temp_0.1 CAP.LOSS_CAMERA.WEIGHTED True CAP.LOSS_INSTANCE.TEMP 0.1 &
CUDA_VISIBLE_DEVICES=7 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/dukemtmc_rho1.6e-3_hg_weighted_intra_listwise_loss PSEUDO.NAME "dbscan_hg_lp" CAP.LOSS_CAMERA.WEIGHTED True PSEUDO.DBSCAN.BASE 'rho' PSEUDO.DBSCAN.RHO 1.6e-3 &