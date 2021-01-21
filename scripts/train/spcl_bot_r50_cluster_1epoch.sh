#!/usr/bin/env bash
echo "SpCL project and clustering every epoch with DataParallel"
time=$(date +%F)
python projects/SpCL_new/train_net.py --config-file projects/SpCL_new/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/SpCL_new/BoT_R50_cluster1epoch/market1501 PSEUDO.CLUSTER_EPOCH 1
python projects/SpCL_new/train_net.py --config-file projects/SpCL_new/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/SpCL_new/BoT_R50_cluster1epoch/dukemtmc PSEUDO.CLUSTER_EPOCH 1
python projects/SpCL_new/train_net.py --config-file projects/SpCL_new/configs/MSMT17/BoT_R50.yml OUTPUT_DIR logs/SpCL_new/BoT_R50_cluster1epoch/msmt17 PSEUDO.CLUSTER_EPOCH 1