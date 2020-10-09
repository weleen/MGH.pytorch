#/bin/env bash
echo "Test supervised learning AGW_R50"
time=$(date +%F)
python tools/train_net.py --config-file configs/Market1501/AGW_R50.yml --num-gpus 4 MODEL.BACKBONE.NORM "syncBN" MODEL.HEADS.NORM "syncBN" OUTPUT_DIR logs/test/$time/dev/supervised/market1501
sleep 10s
python tools/train_net.py --config-file configs/DukeMTMC/AGW_R50.yml --num-gpus 4 MODEL.BACKBONE.NORM "syncBN" MODEL.HEADS.NORM "syncBN" OUTPUT_DIR logs/test/$time/dev/supervised/dukemtmc
#sleep 10s
#python tools/train_net.py --config-file configs/MSMT17/AGW_R50.yml --num-gpus 4 MODEL.BACKBONE.NORM "syncBN" MODEL.HEADS.NORM "syncBN" OUTPUT_DIR logs/test/$time/dev/supervised/msmt17