#!/usr/bin/env bash
echo "SpCL project and clustering every epoch with DataParallel"
python projects/SpCL_new/train_net.py --config-file projects/SpCL_new/configs/Market1501/BoT_R50.yml PSEUDO.CLUSTER_EPOCH 1 OUTPUT_DIR logs/SpCL_new/BoT_R50_cluster1epoch/market1501
python projects/SpCL_new/train_net.py --config-file projects/SpCL_new/configs/DukeMTMC/BoT_R50.yml PSEUDO.CLUSTER_EPOCH 1 OUTPUT_DIR logs/SpCL_new/BoT_R50_cluster1epoch/dukemtmc
python projects/SpCL_new/train_net.py --config-file projects/SpCL_new/configs/MSMT17/BoT_R50.yml PSEUDO.CLUSTER_EPOCH 1 OUTPUT_DIR logs/SpCL_new/BoT_R50_cluster1epoch/msmt17