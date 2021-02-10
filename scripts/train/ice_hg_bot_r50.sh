#!/bin/bash
echo "Run ICE"
python projects/SMT/train_net.py --num-gpus 4 --config-file projects/SMT/configs/Market1501/BoT_R50_ice.yml OUTPUT_DIR logs/SMT/BoT_R50_ice_hg/market1501 PSEUDO.NAME "hypergraph" PSEUDO.NUM_CLUSTER [800,]
python projects/SMT/train_net.py --num-gpus 4 --config-file projects/SMT/configs/DukeMTMC/BoT_R50_ice.yml OUTPUT_DIR logs/SMT/BoT_R50_ice_hg/dukemtmc PSEUDO.NAME "hypergraph" PSEUDO.NUM_CLUSTER [800,]
python projects/SMT/train_net.py --num-gpus 4 --config-file projects/SMT/configs/MSMT17/BoT_R50_ice.yml OUTPUT_DIR logs/SMT/BoT_R50_ice_hg/msmt17 TEST.DO_VAL True PSEUDO.NAME "hypergraph" PSEUDO.NUM_CLUSTER [800,]