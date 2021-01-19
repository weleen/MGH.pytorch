#!/usr/bin/env bash
echo "weighted contrastive SpCL project with DataParallel"
time=$(date +%F)
python projects/SpCL_new/train_net.py --config-file projects/SpCL_new/configs/Market1501/BoT_R50.yml --num-gpus 1 OUTPUT_DIR logs/SpCL_new_no_outliers_single_dbscan/BoT_R50/market1501 PSEUDO.USE_OUTLIERS False PSEUDO.DBSCAN.EPS [0.6] PSEUDO.RESET_OPT True
python projects/SpCL_new/train_net.py --config-file projects/SpCL_new/configs/Market1501/BoT_R50.yml --num-gpus 1 OUTPUT_DIR logs/SpCL_new_use_outliers_single_dbscan/BoT_R50/market1501 PSEUDO.USE_OUTLIERS True PSEUDO.DBSCAN.EPS [0.6] PSEUDO.RESET_OPT True
# sleep 10s
# python projects/SpCL_new/train_net.py --config-file projects/SpCL_new/configs/DukeMTMC/BoT_R50.yml --num-gpus 1 OUTPUT_DIR logs/SpCL_new_no_outliers_single_dbscan/BoT_R50/dukemtmc PSEUDO.USE_OUTLIERS False PSEUDO.DBSCAN.EPS [0.6] PSEUDO.RESET_OPT True
# python projects/SpCL_new/train_net.py --config-file projects/SpCL_new/configs/DukeMTMC/BoT_R50.yml --num-gpus 1 OUTPUT_DIR logs/SpCL_new_use_outliers_single_dbscan/BoT_R50/dukemtmc PSEUDO.USE_OUTLIERS True PSEUDO.DBSCAN.EPS [0.6] PSEUDO.RESET_OPT True
# sleep 10s
# python projects/SpCL_new/train_net.py --config-file projects/SpCL_new/configs/MSMT17/BoT_R50.yml --num-gpus 1 OUTPUT_DIR logs/SpCL_new_no_outliers_single_dbscan/BoT_R50/msmt17 PSEUDO.USE_OUTLIERS False PSEUDO.DBSCAN.EPS [0.6] PSEUDO.RESET_OPT True
# python projects/SpCL_new/train_net.py --config-file projects/SpCL_new/configs/MSMT17/BoT_R50.yml --num-gpus 1 OUTPUT_DIR logs/SpCL_new_use_outliers_single_dbscan/BoT_R50/msmt17 PSEUDO.USE_OUTLIERS True PSEUDO.DBSCAN.EPS [0.6] PSEUDO.RESET_OPT True