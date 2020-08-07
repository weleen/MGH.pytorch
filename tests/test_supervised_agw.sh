#/bin/env bash
python tools/train_net.py --config-file configs/Market1501/AGW_R50.yml --num-gpus 1 OUTPUT_DIR logs/test/market1501
python tools/train_net.py --config-file configs/DukeMTMC/AGW_R50.yml --num-gpus 1 OUTPUT_DIR logs/test/dukemtmc
python tools/train_net.py --config-file configs/MSMT17/AGW_R50.yml --num-gpus 1 OUTPUT_DIR logs/test/msmt17