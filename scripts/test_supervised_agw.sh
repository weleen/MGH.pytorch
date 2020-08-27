#/bin/env bash
echo "Test supervised learning AGW_R50"
python tools/train_net.py --config-file configs/Market1501/AGW_R50.yml OUTPUT_DIR logs/test/supervised/market1501 TEST.EVAL_PERIOD 1
python tools/train_net.py --config-file configs/DukeMTMC/AGW_R50.yml OUTPUT_DIR logs/test/supervised/dukemtmc  TEST.EVAL_PERIOD 1
python tools/train_net.py --config-file configs/MSMT17/AGW_R50.yml OUTPUT_DIR logs/test/supervised/msmt17  TEST.EVAL_PERIOD 1