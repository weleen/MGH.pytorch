#!/usr/bin/env bash
echo "Test SpCL soft label with DataParallel"
python projects/SpCL_new/train_net.py --config-file projects/SpCL_new/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/SpCL_new/BoT_R50_softlabel/market1501 PSEUDO.MEMORY.SOFT_LABEL_START_EPOCH 0 PSEUDO.MEMORY.SOFT_LABEL True
python projects/SpCL_new/train_net.py --config-file projects/SpCL_new/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/SpCL_new/BoT_R50_softlabel/dukemtmc PSEUDO.MEMORY.SOFT_LABEL_START_EPOCH 0 PSEUDO.MEMORY.SOFT_LABEL True
python projects/SpCL_new/train_net.py --config-file projects/SpCL_new/configs/MSMT17/BoT_R50.yml OUTPUT_DIR logs/SpCL_new/BoT_R50_softlabel/msmt17 PSEUDO.MEMORY.SOFT_LABEL_START_EPOCH 0 PSEUDO.MEMORY.SOFT_LABEL True
