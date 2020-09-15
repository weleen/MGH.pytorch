#/bin/env bash
echo "Test supervised learning AGW_R50"
time=$(date +%F)
python tools/train_net.py --config-file configs/Market1501/AGW_R50.yml OUTPUT_DIR --num-gpus 4 MODE.BACKBONE.NORM "syncBN" MODEL.HEADS.NORM "syncBN" logs/test/$time/dev/supervised/market1501 TEST.EVAL_PERIOD 10 &
sleep 10s
python tools/train_net.py --config-file configs/DukeMTMC/AGW_R50.yml OUTPUT_DIR --num-gpus 4 MODE.BACKBONE.NORM "syncBN" MODEL.HEADS.NORM "syncBN" logs/test/$time/dev/supervised/dukemtmc TEST.EVAL_PERIOD 10 &
sleep 10s
python tools/train_net.py --config-file configs/MSMT17/AGW_R50.yml OUTPUT_DIR --num-gpus 4 MODE.BACKBONE.NORM "syncBN" MODEL.HEADS.NORM "syncBN" logs/test/$time/dev/supervised/msmt17 TEST.EVAL_PERIOD 10 &