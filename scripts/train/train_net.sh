#/bin/env bash
echo "Test supervised learning AGW_R50"
time=$(date +%F)
python tools/train_net.py --config-file configs/Market1501/AGW_R50.yml --num-gpus 4 MODEL.BACKBONE.NORM "syncBN" MODEL.HEADS.NORM "syncBN" INPUT.DO_BLUR True OUTPUT_DIR logs/test/$time/blur-supervised/market1501