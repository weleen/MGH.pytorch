CUDA_VISIBLE_DEVICES=3 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501 CAP.INSTANCE_LOSS = False

CUDA_VISIBLE_DEVICES=3 python projects/CAP/train_net.py --eval-only --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/debug MODEL.WEIGHTS /home/wuyiming/disk50/project/reid/logs/CAP/BoT_R50/market1501/model_final.pth CAP.ST_TEST True

CUDA_VISIBLE_DEVICES=3 python projects/CAP/train_net.py --eval-only --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/debug MODEL.WEIGHTS /home/wuyiming/disk50/project/reid/logs/CAP/BoT_R50/dukemtmc/model_final.pth CAP.ST_TEST True
