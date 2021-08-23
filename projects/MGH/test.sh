# testing
CUDA_VISIBLE_DEVICES=0 python projects/MGH/train_net.py --eval-only --num-gpus 1 --config-file projects/MGH/configs/Market1501/BoT_R50.yml OUTPUT_DIR logs/MGH/market1501_test MODEL.WEIGHTS models/market/model_market.pth CAP.ST_TEST True
CUDA_VISIBLE_DEVICES=1 python projects/MGH/train_net.py --eval-only --num-gpus 1 --config-file projects/MGH/configs/DukeMTMC/BoT_R50.yml OUTPUT_DIR logs/MGH/dukemtmc_test MODEL.WEIGHTS models/duke/model_duke.pth CAP.ST_TEST True
CUDA_VISIBLE_DEVICES=2 python projects/MGH/train_net.py --eval-only --num-gpus 1 --config-file projects/MGH/configs/MSMT17/BoT_R50.yml OUTPUT_DIR logs/MGH/msmt17_test MODEL.WEIGHTS models/msmt/model_msmt.pth CAP.ST_TEST True
