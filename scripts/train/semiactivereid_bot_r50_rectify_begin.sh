#!/usr/bin/env bash
echo "Semi-ActiveReID project with DataParallel"
# rectify from begining
python projects/Semi-ActiveReID/train_net.py --config-file projects/Semi-ActiveReID/configs/Market1501/BoT_R50.yml ACTIVE.RECTIFY True ACTIVE.BUILD_DATALOADER False ACTIVE.EDGE_PROP True ACTIVE.END_EPOCH 100 OUTPUT_DIR logs/SpCL_SemiActive/BoT_R50_rectify_begin_random/market1501
python projects/Semi-ActiveReID/train_net.py --config-file projects/Semi-ActiveReID/configs/DukeMTMC/BoT_R50.yml ACTIVE.RECTIFY True ACTIVE.BUILD_DATALOADER False ACTIVE.EDGE_PROP True ACTIVE.END_EPOCH 100 OUTPUT_DIR logs/SpCL_SemiActive/BoT_R50_rectify_begin_random/dukemtmc
python projects/Semi-ActiveReID/train_net.py --config-file projects/Semi-ActiveReID/configs/MSMT17/BoT_R50.yml ACTIVE.RECTIFY True ACTIVE.BUILD_DATALOADER False ACTIVE.EDGE_PROP True ACTIVE.END_EPOCH 100 OUTPUT_DIR logs/SpCL_SemiActive/BoT_R50_rectify_begin_random/msmt17