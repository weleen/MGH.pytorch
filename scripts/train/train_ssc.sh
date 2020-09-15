#!/usr/bin/env bash
#python projects/SSC/train_net.py --config-file projects/SSC/configs/Market1501/AGW_R50.yml --num-gpus 1 OUTPUT_DIR logs/SSC/SimCLR/market1501/ SOLVER.IMS_PER_BATCH 32 TEST.EVAL_PERIOD 1
#python projects/SSC/train_net.py --config-file projects/SSC/configs/DukeMTMC/AGW_R50.yml --num-gpus 1 OUTPUT_DIR logs/SSC/SimCLR/dukemtmc/ SOLVER.IMS_PER_BATCH 32 TEST.EVAL_PERIOD 1
#python projects/SSC/train_net.py --config-file projects/SSC/configs/MSMT17/AGW_R50.yml --num-gpus 1 OUTPUT_DIR logs/SSC/SimCLR/msmt17/ SOLVER.IMS_PER_BATCH 32 TEST.EVAL_PERIOD 1

python projects/SSC/train_net.py --config-file projects/SSC/configs/Market1501/usl_R50.yml --num-gpus 1 OUTPUT_DIR logs/SSC/AutoNovel/market1501/ SOLVER.IMS_PER_BATCH 32 TEST.EVAL_PERIOD 1
python projects/SSC/train_net.py --config-file projects/SSC/configs/DukeMTMC/usl_R50.yml --num-gpus 1 OUTPUT_DIR logs/SSC/AutoNovel/dukemtmc/ SOLVER.IMS_PER_BATCH 32 TEST.EVAL_PERIOD 1
python projects/SSC/train_net.py --config-file projects/SSC/configs/MSMT17/usl_R50.yml --num-gpus 1 OUTPUT_DIR logs/SSC/AutoNovel/msmt17/ SOLVER.IMS_PER_BATCH 32 TEST.EVAL_PERIOD 1