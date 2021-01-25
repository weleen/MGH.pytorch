#!/usr/bin/env bash
echo "Test weighted contrastive SpCL project with DataParallel"
python projects/SpCL_new/train_net.py --config-file projects/SpCL_new/configs/Market1501/BoT_R50.yml PSEUDO.MEMORY.WEIGHTED True OUTPUT_DIR logs/SpCL_new/BoT_R50_weighted/market1501
python projects/SpCL_new/train_net.py --config-file projects/SpCL_new/configs/DukeMTMC/BoT_R50.yml PSEUDO.MEMORY.WEIGHTED True OUTPUT_DIR logs/SpCL_new/BoT_R50_weighted/dukemtmc
python projects/SpCL_new/train_net.py --config-file projects/SpCL_new/configs/MSMT17/BoT_R50.yml PSEUDO.MEMORY.WEIGHTED True OUTPUT_DIR logs/SpCL_new/BoT_R50_weighted/msmt17 TEST.DO_VAL True
