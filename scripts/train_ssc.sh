#!/usr/bin/env bash
python projects/SSC/train_net.py --config-file projects/SSC/configs/Market1501/usl_R50.yml --num-gpus 4 OUTPUT_DIR logs/SSC/debug/market1501/ SOLVER.IMS_PER_BATCH 128
python projects/SSC/train_net.py --config-file projects/SSC/configs/DukeMTMC/usl_R50.yml --num-gpus 4 OUTPUT_DIR logs/SSC/debug/dukemtmc/ SOLVER.IMS_PER_BATCH 128
python projects/SSC/train_net.py --config-file projects/SSC/configs/MSMT17/usl_R50.yml --num-gpus 4 OUTPUT_DIR logs/SSC/debug/msmt17/ SOLVER.IMS_PER_BATCH 128