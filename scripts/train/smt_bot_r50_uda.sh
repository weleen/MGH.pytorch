#!/bin/bash
GPUS=${1:-4}
echo "Run SMT UDA"
echo "Duke->Market"
python projects/SMT/train_net.py --num-gpus $GPUS --config-file projects/SMT/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/SMT/BoT_R50/dukemtmc_to_market1501 MODEL.WEIGHTS logs/BoT_R50/dukemtmc/model_final_add_net.pth
echo "Market->Duke"
python projects/SMT/train_net.py --num-gpus $GPUS --config-file projects/SMT/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/SMT/BoT_R50/market1501_to_dukemtmc MODEL.WEIGHTS logs/BoT_R50/market1501/model_final_add_net.pth
echo "Market->MSMT"
python projects/SMT/train_net.py --num-gpus $GPUS --config-file projects/SMT/configs/MSMT17/BoT_R50.yml OUTPUT_DIR logs/SMT/BoT_R50/market1501_to_msmt17 TEST.DO_VAL True MODEL.WEIGHTS logs/BoT_R50/market1501/model_final_add_net.pth
echo "Duke->MSMT"
python projects/SMT/train_net.py --num-gpus $GPUS --config-file projects/SMT/configs/MSMT17/BoT_R50.yml OUTPUT_DIR logs/SMT/BoT_R50/dukemtmc_to_msmt17 TEST.DO_VAL True MODEL.WEIGHTS logs/BoT_R50/dukemtmc/model_final_add_net.pth