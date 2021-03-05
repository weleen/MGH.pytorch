CUDA_VISIBLE_DEVICES=3 python projects/CAP/train_net.py --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/CAP/BoT_R50/market1501 CAP.INSTANCE_LOSS = False

CUDA_VISIBLE_DEVICES=3 python projects/CAP/train_net.py --eval-only --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/debug MODEL.WEIGHTS /home/wuyiming/disk50/project/reid/logs/CAP/BoT_R50/market1501/model_final.pth CAP.ST_TEST True

CUDA_VISIBLE_DEVICES=3 python projects/CAP/train_net.py --eval-only --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/debug MODEL.WEIGHTS /home/wuyiming/disk50/project/reid/logs/CAP/BoT_R50/dukemtmc/model_final.pth CAP.ST_TEST True

CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --eval-only --num-gpus 1 --config-file projects/CAP/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/active/dukemtmc/ MODEL.WEIGHTS /home/wuyiming/tianjian/fast-reid-0915/logs/dukemtmc/active_tri_from_cap/model_final.pth CAP.ST_TEST True

CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --eval-only --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/active/market1501/ MODEL.WEIGHTS /home/wuyiming/tianjian/fast-reid-0915/logs/market1501/active_tri_from_cap_0225/model_final.pth CAP.ST_TEST True

CUDA_VISIBLE_DEVICES=0 python projects/CAP/train_net.py --eval-only --num-gpus 1 --config-file projects/CAP/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/debug/ MODEL.WEIGHTS /home/wuyiming/tianjian/fast-reid-0915/logs/market1501/active_tri_from_cap_0225/model_final.pth CAP.ST_TEST True

CUDA_VISIBLE_DEVICES=3 python projects/CAP/train_net.py --eval-only --num-gpus 1 --config-file projects/CAP/configs/MSMT17/BoT_R50.yml OUTPUT_DIR logs/active/msmt17/ MODEL.WEIGHTS /home/wuyiming/tianjian/fast-reid-0915/logs/msmt17/active_tri_from_cap/model_0023999.pth CAP.ST_TEST True

CUDA_VISIBLE_DEVICES=3 python projects/CAP/train_net.py --eval-only --num-gpus 1 --config-file projects/CAP/configs/MSMT17/BoT_R50.yml OUTPUT_DIR logs/st_test/msmt17/ MODEL.WEIGHTS /home/wuyiming/disk50/project/reid/logs/CAP/BoT_R50/msmt17/model_0019999.pth CAP.ST_TEST True