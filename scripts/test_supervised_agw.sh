#/bin/env bash
echo "Test supervised learning AGW_R50"
python tools/train_net.py --config-file configs/Market1501/AGW_R50.yml --num-gpus 4 OUTPUT_DIR logs/test/market1501 MODEL.BACKBONE.NORM "syncBN" MODEL.HEADS.NORM "syncBN"
python tools/train_net.py --config-file configs/DukeMTMC/AGW_R50.yml --num-gpus 4 OUTPUT_DIR logs/test/dukemtmc MODEL.BACKBONE.NORM "syncBN" MODEL.HEADS.NORM "syncBN"
python tools/train_net.py --config-file configs/MSMT17/AGW_R50.yml --num-gpus 4 OUTPUT_DIR logs/test/msmt17 MODEL.BACKBONE.NORM "syncBN" MODEL.HEADS.NORM "syncBN"

echo "Test SPCL project"
python projects/SPCL/train_net.py --config-file projects/SPCL/configs/Market1501/usl_R50.yml --num-gpus 1 OUTPUT_DIR logs/test/spcl/market1501
python projects/SPCL/train_net.py --config-file projects/SPCL/configs/DukeMTMC/usl_R50.yml --num-gpus 1 OUTPUT_DIR logs/test/spcl/dukemtmc
python projects/SPCL/train_net.py --config-file projects/SPCL/configs/MSMT17/usl_R50.yml --num-gpus 1 OUTPUT_DIR logs/test/spcl/msmt17