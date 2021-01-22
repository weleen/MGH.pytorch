#/bin/env bash
echo "Train Low Level baseline BoT_R50"
python tools/train_net.py --config-file configs/Market1501/LL_R50.yml OUTPUT_DIR log/LL_R50/no_mask/market1501
python tools/train_net.py --config-file configs/DukeMTMC/LL_R50.yml OUTPUT_DIR log/LL_R50/no_mask/dukemtmc
python tools/train_net.py --config-file configs/MSMT17/LL_R50.yml OUTPUT_DIR log/LL_R50/no_mask/msmt17
