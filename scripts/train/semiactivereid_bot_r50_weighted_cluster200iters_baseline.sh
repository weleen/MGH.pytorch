#!/usr/bin/env bash
echo "weighted contrastive SpCL project with DataParallel"
python projects/Semi-ActiveReID/train_net.py --config-file projects/Semi-ActiveReID/configs/Market1501/BoT_R50.yml \
                                             OUTPUT_DIR logs/SpCL_wcl_cluster200iters/BoT_R50/market1501 \
                                             PSEUDO.MEMORY.WEIGHTED True \
                                             PSEUDO.CLUSTER_EPOCH 1 \
                                             ACTIVE.RECTIFY False \
                                             ACTIVE.BUILD_DATALOADER False
python projects/Semi-ActiveReID/train_net.py --config-file projects/Semi-ActiveReID/configs/DukeMTMC/BoT_R50.yml \
                                             OUTPUT_DIR logs/SpCL_wcl_cluster200iters/BoT_R50/dukemtmc \
                                             PSEUDO.MEMORY.WEIGHTED True \
                                             PSEUDO.CLUSTER_EPOCH 1 \
                                             ACTIVE.RECTIFY False \
                                             ACTIVE.BUILD_DATALOADER False
python projects/Semi-ActiveReID/train_net.py --config-file projects/Semi-ActiveReID/configs/MSMT17/BoT_R50.yml \
                                             OUTPUT_DIR logs/SpCL_wcl_cluster200iters/BoT_R50/msmt17 \
                                             PSEUDO.MEMORY.WEIGHTED True \
                                             PSEUDO.CLUSTER_EPOCH 1 \
                                             ACTIVE.RECTIFY False \
                                             ACTIVE.BUILD_DATALOADER False \
                                             TEST.DO_VAL True