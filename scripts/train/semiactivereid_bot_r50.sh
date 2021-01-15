###
 # @Author: WuYiming
 # @Date: 2020-10-26 22:43:13
 # @LastEditTime: 2020-11-18 09:48:35
 # @LastEditors: Please set LastEditors
 # @Description: script for training spcl bot_r50 with rectifying labels.
 # @FilePath: /fast-reid/scripts/train/spcl_bot_r50_weighted_rectify.sh
### 
#!/usr/bin/env bash
echo "Semi-ActiveReID project with DataParallel"
time=$(date +%F)
python projects/Semi-ActiveReID/train_net.py --config-file projects/Semi-ActiveReID/configs/Market1501/BoT_R50.yml \
                                             OUTPUT_DIR logs/SemiActive_SpCL_Rectify/BoT_R50/market1501/start20_end160_sample10_m0.05_queryrandom_rectifytrue_activefalse_edgetrue_nodefalse_k-1 \
                                             PSEUDO.MEMORY.WEIGHTED False \
                                             ACTIVE.START_ITER 20 \
                                             ACTIVE.END_ITER 160 \
                                             ACTIVE.SAMPLE_ITER 10 \
                                             ACTIVE.SAMPLE_M 0.05 \
                                             ACTIVE.SAMPLER.QUERY_FUNC 'random' \
                                             ACTIVE.RECTIFY True \
                                             ACTIVE.BUILD_DATALOADER False \
                                             ACTIVE.EDGE_PROP True \
                                             ACTIVE.NODE_PROP False \
                                             ACTIVE.NODE_PROP_K -1

python projects/Semi-ActiveReID/train_net.py --config-file projects/Semi-ActiveReID/configs/Market1501/BoT_R50.yml \
                                             OUTPUT_DIR logs/SemiActive_SpCL_Rectify/BoT_R50/market1501/start120_end140_sample10_m0.2_queryentropy_rectifytrue_activefalse_edgetrue_nodefalse_k-1 \
                                             PSEUDO.MEMORY.WEIGHTED False \
                                             ACTIVE.START_ITER 120 \
                                             ACTIVE.END_ITER 140 \
                                             ACTIVE.SAMPLE_ITER 10 \
                                             ACTIVE.SAMPLE_M 0.2 \
                                             ACTIVE.SAMPLER.QUERY_FUNC 'entropy' \
                                             ACTIVE.RECTIFY True \
                                             ACTIVE.BUILD_DATALOADER False \
                                             ACTIVE.EDGE_PROP True \
                                             ACTIVE.NODE_PROP False \
                                             ACTIVE.NODE_PROP_K -1

python projects/Semi-ActiveReID/train_net.py --config-file projects/Semi-ActiveReID/configs/Market1501/BoT_R50.yml \
                                             OUTPUT_DIR logs/SemiActive_SpCL_Rectify/BoT_R50/market1501/start20_end160_sample10_m0.05_queryconfidence_rectifytrue_activefalse_edgetrue_nodefalse_k-1 \
                                             PSEUDO.MEMORY.WEIGHTED False \
                                             ACTIVE.START_ITER 20 \
                                             ACTIVE.END_ITER 160 \
                                             ACTIVE.SAMPLE_ITER 10 \
                                             ACTIVE.SAMPLE_M 0.05 \
                                             ACTIVE.SAMPLER.QUERY_FUNC 'confidence' \
                                             ACTIVE.RECTIFY True \
                                             ACTIVE.BUILD_DATALOADER False \
                                             ACTIVE.EDGE_PROP True \
                                             ACTIVE.NODE_PROP False \
                                             ACTIVE.NODE_PROP_K -1

python projects/Semi-ActiveReID/train_net.py --config-file projects/Semi-ActiveReID/configs/MSMT17/BoT_R50.yml \
                                             OUTPUT_DIR logs/SemiActive_SpCL_Rectify/BoT_R50/msmt17/start0_end140_sample10_m0.2_queryentropy_rectifytrue_activefalse_edgetrue_nodefalse_k-1 \
                                             PSEUDO.MEMORY.WEIGHTED False \
                                             ACTIVE.START_ITER 0 \
                                             ACTIVE.END_ITER 140 \
                                             ACTIVE.SAMPLE_ITER 10 \
                                             ACTIVE.SAMPLE_M 0.2 \
                                             ACTIVE.SAMPLER.QUERY_FUNC 'entropy' \
                                             ACTIVE.RECTIFY True \
                                             ACTIVE.BUILD_DATALOADER False \
                                             ACTIVE.EDGE_PROP True \
                                             ACTIVE.NODE_PROP False \
                                             ACTIVE.NODE_PROP_K -1

python projects/Semi-ActiveReID/train_net.py --config-file projects/Semi-ActiveReID/configs/Market1501/BoT_R50.yml \
                                             OUTPUT_DIR logs/SemiActive_SpCL_Rectify/BoT_R50/market1501/start0_end160_sample10_m0.3_queryrandom_rectifytrue_activefalse_edgetrue_nodefalse_k-1 \
                                             PSEUDO.MEMORY.WEIGHTED False \
                                             PSEUDO.CLUSTER_EPOCH 2 \
                                             ACTIVE.START_ITER 0 \
                                             ACTIVE.END_ITER 160 \
                                             ACTIVE.SAMPLE_ITER 10 \
                                             ACTIVE.SAMPLE_M 0.3 \
                                             ACTIVE.SAMPLER.QUERY_FUNC 'random' \
                                             ACTIVE.RECTIFY True \
                                             ACTIVE.BUILD_DATALOADER False \
                                             ACTIVE.EDGE_PROP True \
                                             ACTIVE.NODE_PROP False \
                                             ACTIVE.NODE_PROP_K -1

python projects/Semi-ActiveReID/train_net.py --config-file projects/Semi-ActiveReID/configs/Market1501/BoT_R50.yml \
                                             OUTPUT_DIR logs/SemiActive_SpCL_Rectify/BoT_R50/market1501/baseline \
                                             PSEUDO.MEMORY.WEIGHTED False \
                                             PSEUDO.CLUSTER_EPOCH 2 \
                                             ACTIVE.START_ITER 0 \
                                             ACTIVE.END_ITER 160 \
                                             ACTIVE.SAMPLE_ITER 10 \
                                             ACTIVE.SAMPLE_M 0.3 \
                                             ACTIVE.SAMPLER.QUERY_FUNC 'random' \
                                             ACTIVE.RECTIFY False \
                                             ACTIVE.BUILD_DATALOADER False \
                                             ACTIVE.EDGE_PROP False \
                                             ACTIVE.EDGE_PROP_MAX_EPOCH 50 \
                                             ACTIVE.NODE_PROP False \
                                             ACTIVE.NODE_PROP_K -1

python projects/Semi-ActiveReID/train_net.py --config-file projects/Semi-ActiveReID/configs/Market1501/BoT_R50.yml \
                                             OUTPUT_DIR logs/SemiActive_SpCL_Rectify/BoT_R50/market1501/start0_end160_sample10_m0.3_queryrandom_rectifytrue_activefalse_edgetrue_edgemaxiter10_nodefalse_k-1_twolosses_momentum0.2 \
                                             MODEL.LOSSES.NAME \(\"ContrastiveLoss\",\"TripletLoss\"\) \
                                             PSEUDO.MEMORY.WEIGHTED False \
                                             PSEUDO.CLUSTER_EPOCH 2 \
                                             PSEUDO.MEMORY.MOMENTUM 0.2 \
                                             ACTIVE.START_ITER 0 \
                                             ACTIVE.END_ITER 160 \
                                             ACTIVE.SAMPLE_ITER 10 \
                                             ACTIVE.SAMPLE_M 0.2 \
                                             ACTIVE.SAMPLER.QUERY_FUNC 'random' \
                                             ACTIVE.RECTIFY True \
                                             ACTIVE.BUILD_DATALOADER False \
                                             ACTIVE.EDGE_PROP True \
                                             ACTIVE.EDGE_PROP_MAX_EPOCH 10 \
                                             ACTIVE.NODE_PROP False \
                                             ACTIVE.NODE_PROP_K -1

python projects/Semi-ActiveReID/train_net.py --config-file projects/Semi-ActiveReID/configs/Market1501/BoT_R50.yml \
                                             OUTPUT_DIR logs/SemiActive_SpCL_Rectify/BoT_R50/market1501/start0_end160_sample10_m0.3_queryentropy_rectifytrue_activefalse_edgetrue_edgemaxiter10_nodefalse_k-1_twolosses_momentum0.2 \
                                             MODEL.LOSSES.NAME \(\"ContrastiveLoss\",\"TripletLoss\"\) \
                                             PSEUDO.MEMORY.WEIGHTED False \
                                             PSEUDO.CLUSTER_EPOCH 2 \
                                             PSEUDO.MEMORY.MOMENTUM 0.2 \
                                             ACTIVE.START_ITER 0 \
                                             ACTIVE.END_ITER 160 \
                                             ACTIVE.SAMPLE_ITER 10 \
                                             ACTIVE.SAMPLE_M 0.2 \
                                             ACTIVE.SAMPLER.QUERY_FUNC 'entropy' \
                                             ACTIVE.RECTIFY True \
                                             ACTIVE.BUILD_DATALOADER False \
                                             ACTIVE.EDGE_PROP True \
                                             ACTIVE.EDGE_PROP_MAX_EPOCH 10 \
                                             ACTIVE.NODE_PROP False \
                                             ACTIVE.NODE_PROP_K -1
##

python projects/Semi-ActiveReID/train_net.py --config-file projects/Semi-ActiveReID/configs/Market1501/BoT_R50.yml \
                                             OUTPUT_DIR logs/SemiActive_SpCL_Rectify/BoT_R50/market1501/start0_end160_sample10_m0.05_queryrandom_rectifytrue_activefalse_edgetrue_edgemaxiter10_nodefalse_k-1_twolosses_momentum0.2 \
                                             MODEL.LOSSES.NAME \(\"ContrastiveLoss\",\"TripletLoss\"\) \
                                             PSEUDO.MEMORY.WEIGHTED False \
                                             PSEUDO.CLUSTER_EPOCH 2 \
                                             PSEUDO.MEMORY.MOMENTUM 0.2 \
                                             ACTIVE.START_ITER 0 \
                                             ACTIVE.END_ITER 160 \
                                             ACTIVE.SAMPLE_ITER 10 \
                                             ACTIVE.SAMPLE_M 0.05 \
                                             ACTIVE.SAMPLER.QUERY_FUNC 'random' \
                                             ACTIVE.RECTIFY True \
                                             ACTIVE.BUILD_DATALOADER False \
                                             ACTIVE.EDGE_PROP True \
                                             ACTIVE.EDGE_PROP_MAX_EPOCH 10 \
                                             ACTIVE.NODE_PROP False \
                                             ACTIVE.NODE_PROP_K -1

python projects/Semi-ActiveReID/train_net.py --config-file projects/Semi-ActiveReID/configs/Market1501/BoT_R50.yml \
                                             OUTPUT_DIR logs/SemiActive_SpCL_Rectify/BoT_R50/market1501/start0_end160_sample10_m0.1_queryrandom_rectifytrue_activefalse_edgetrue_edgemaxiter10_nodefalse_k-1_twolosses_momentum0.2 \
                                             MODEL.LOSSES.NAME \(\"ContrastiveLoss\",\"TripletLoss\"\) \
                                             PSEUDO.MEMORY.WEIGHTED False \
                                             PSEUDO.CLUSTER_EPOCH 2 \
                                             PSEUDO.MEMORY.MOMENTUM 0.2 \
                                             ACTIVE.START_ITER 0 \
                                             ACTIVE.END_ITER 160 \
                                             ACTIVE.SAMPLE_ITER 10 \
                                             ACTIVE.SAMPLE_M 0.1 \
                                             ACTIVE.SAMPLER.QUERY_FUNC 'random' \
                                             ACTIVE.RECTIFY True \
                                             ACTIVE.BUILD_DATALOADER False \
                                             ACTIVE.EDGE_PROP True \
                                             ACTIVE.EDGE_PROP_MAX_EPOCH 10 \
                                             ACTIVE.NODE_PROP False \
                                             ACTIVE.NODE_PROP_K -1

python projects/Semi-ActiveReID/train_net.py --config-file projects/Semi-ActiveReID/configs/Market1501/BoT_R50.yml \
                                             OUTPUT_DIR logs/SemiActive_SpCL_Rectify/BoT_R50/market1501/start0_end160_sample10_m0.2_queryrandom_rectifytrue_activefalse_edgetrue_edgemaxiter10_nodefalse_k-1_twolosses_momentum0.2 \
                                             MODEL.LOSSES.NAME \(\"ContrastiveLoss\",\"TripletLoss\"\) \
                                             PSEUDO.MEMORY.WEIGHTED False \
                                             PSEUDO.CLUSTER_EPOCH 2 \
                                             PSEUDO.MEMORY.MOMENTUM 0.2 \
                                             ACTIVE.START_ITER 0 \
                                             ACTIVE.END_ITER 160 \
                                             ACTIVE.SAMPLE_ITER 10 \
                                             ACTIVE.SAMPLE_M 0.2 \
                                             ACTIVE.SAMPLER.QUERY_FUNC 'random' \
                                             ACTIVE.RECTIFY True \
                                             ACTIVE.BUILD_DATALOADER False \
                                             ACTIVE.EDGE_PROP True \
                                             ACTIVE.EDGE_PROP_MAX_EPOCH 10 \
                                             ACTIVE.NODE_PROP False \
                                             ACTIVE.NODE_PROP_K -1
# sleep 10s
# python projects/Semi-ActiveReID/train_net.py --config-file projects/Semi-ActiveReID/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/SemiActive_SpCL_Contrast_Rectify/BoT_R50/dukemtmc PSEUDO.MEMORY.WEIGHTED False ACTIVE.RECTIFY True ACTIVE.BUILD_DATALOADER False ACTIVE.EDGE_PROP True ACTIVE.NODE_PROP False ACTIVE.START_ITER 20
# sleep 10s
# python projects/Semi-ActiveReID/train_net.py --config-file projects/Semi-ActiveReID/configs/MSMT17/BoT_R50.yml OUTPUT_DIR logs/SemiActive_SpCL_Contrast_Rectify/BoT_R50/msmt17 PSEUDO.MEMORY.WEIGHTED False TEST.DO_VAL True ACTIVE.RECTIFY True ACTIVE.BUILD_DATALOADER False ACTIVE.EDGE_PROP True ACTIVE.NODE_PROP False