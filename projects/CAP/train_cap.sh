python projects/CAP/train_net.py --config-file projects/CAP/configs/Market1501/BoT_R50.yml \
OUTPUT_DIR logs/cap \
DATALOADER.SAMPLER_NAME ProxyBalancedSampler \
DATALOADER.NUM_INSTANCE 4 \
SOLVER.IMS_PER_BATCH 32 \
SOLVER.MAX_EPOCH 50 \
SOLVER.STEPS "[10,30]" \
SOLVER.WARMUP_EPOCHS 10 \
PSEUDO.DBSCAN.EPS "[0.5,]"
