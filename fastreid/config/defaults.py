from fvcore.common.config import CfgNode as CN

# NOTE: _EPOCH and _PERIOD means epoch in training, while _ITER and _ITERS means iteration
# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "Baseline"

_C.MODEL.OPEN_LAYERS = ['']

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

_C.MODEL.BACKBONE.NAME = "build_resnet_backbone"
_C.MODEL.BACKBONE.DEPTH = "50x"
_C.MODEL.BACKBONE.LAST_STRIDE = 1
# Backbone feature dimension
_C.MODEL.BACKBONE.FEAT_DIM = 2048
# Normalization method for the convolution layers.
_C.MODEL.BACKBONE.NORM = "BN"
# If use IBN block in backbone
_C.MODEL.BACKBONE.WITH_IBN = False
# If use SE block in backbone
_C.MODEL.BACKBONE.WITH_SE = False
# If use Non-local block in backbone
_C.MODEL.BACKBONE.WITH_NL = False
# If use ImageNet pretrain model
_C.MODEL.BACKBONE.PRETRAIN = True
# Pretrain model path
_C.MODEL.BACKBONE.PRETRAIN_PATH = ''

# ---------------------------------------------------------------------------- #
# REID HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.HEADS = CN()
_C.MODEL.HEADS.NAME = "EmbeddingHead"
# Normalization method for the convolution layers.
_C.MODEL.HEADS.NORM = "BN"
# Number of identity
_C.MODEL.HEADS.NUM_CLASSES = 0
# Embedding dimension in head
_C.MODEL.HEADS.EMBEDDING_DIM = 0
# If use BNneck in embedding
_C.MODEL.HEADS.WITH_BNNECK = True
# Number of cameras
_C.MODEL.HEADS.NUM_CAMERAS = 0
# Triplet feature using feature before(after) bnneck
_C.MODEL.HEADS.NECK_FEAT = "before"  # options: before, after
# Pooling layer type
_C.MODEL.HEADS.POOL_LAYER = "avgpool"

# Classification layer type
_C.MODEL.HEADS.CLS_LAYER = "linear"  # "arcSoftmax" or "circleSoftmax"

# Margin and Scale for margin-based classification layer
_C.MODEL.HEADS.MARGIN = 0.15
_C.MODEL.HEADS.SCALE = 128

# ---------------------------------------------------------------------------- #
# REID LOSSES options
# ---------------------------------------------------------------------------- #
_C.MODEL.LOSSES = CN()
_C.MODEL.LOSSES.NAME = ("CrossEntropyLoss",)

# Cross Entropy Loss options
_C.MODEL.LOSSES.CE = CN()
# if epsilon == 0, it means no label smooth regularization,
# if epsilon == -1, it means adaptive label smooth regularization
_C.MODEL.LOSSES.CE.EPSILON = 0.0
_C.MODEL.LOSSES.CE.ALPHA = 0.2
_C.MODEL.LOSSES.CE.SCALE = 1.0

# Focal Loss options
_C.MODEL.LOSSES.FL = CN()
_C.MODEL.LOSSES.FL.ALPHA = 0.25
_C.MODEL.LOSSES.FL.GAMMA = 2
_C.MODEL.LOSSES.FL.SCALE = 1.0

# Triplet Loss options
_C.MODEL.LOSSES.TRI = CN()
_C.MODEL.LOSSES.TRI.MARGIN = 0.3
_C.MODEL.LOSSES.TRI.NORM_FEAT = False
_C.MODEL.LOSSES.TRI.HARD_MINING = True
_C.MODEL.LOSSES.TRI.SCALE = 1.0

# Circle Loss options
_C.MODEL.LOSSES.CIRCLE = CN()
_C.MODEL.LOSSES.CIRCLE.MARGIN = 0.25
_C.MODEL.LOSSES.CIRCLE.GAMMA = 128
_C.MODEL.LOSSES.CIRCLE.SCALE = 1.0

# Cosface Loss options
_C.MODEL.LOSSES.COSFACE = CN()
_C.MODEL.LOSSES.COSFACE.MARGIN = 0.25
_C.MODEL.LOSSES.COSFACE.GAMMA = 128
_C.MODEL.LOSSES.COSFACE.SCALE = 1.0

# Pairwise Smooth Loss options
_C.MODEL.LOSSES.PS = CN()
_C.MODEL.LOSSES.PS.SCALE = 2.0
_C.MODEL.LOSSES.PS.SIGMA = 1.0

# Path to a checkpoint file to be loaded to the model. You can find available models in the model zoo.
_C.MODEL.WEIGHTS = ""

# Values to be used for image normalization
_C.MODEL.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.MODEL.PIXEL_STD = [0.229, 0.224, 0.225]

# Domain-Specific batch normalization
_C.MODEL.DSBN = True
# Mean Teacher Network
_C.MODEL.MEAN_NET = False
_C.MODEL.MEAN_NET_ALPHA = 0.999

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]

# Random probability for image horizontal flip
_C.INPUT.DO_FLIP = True
_C.INPUT.FLIP_PROB = 0.5

# Value of padding size
_C.INPUT.DO_PAD = True
_C.INPUT.PADDING_MODE = 'constant'
_C.INPUT.PADDING = 10

# Gaussian blur
_C.INPUT.DO_BLUR = False
_C.INPUT.BLUR_PROB = 0.5

# Random color jitter
_C.INPUT.CJ = CN()
_C.INPUT.CJ.ENABLED = False
_C.INPUT.CJ.PROB = 0.5
_C.INPUT.CJ.BRIGHTNESS = 0.15
_C.INPUT.CJ.CONTRAST = 0.15
_C.INPUT.CJ.SATURATION = 0.1
_C.INPUT.CJ.HUE = 0.1

# Auto augmentation
_C.INPUT.DO_AUTOAUG = False
_C.INPUT.AUTOAUG_PROB = 0.0

# Augmix augmentation
_C.INPUT.DO_AUGMIX = False
_C.INPUT.AUGMIX_PROB = 0.0

# Random Erasing
_C.INPUT.REA = CN()
_C.INPUT.REA.ENABLED = False
_C.INPUT.REA.PROB = 0.5
_C.INPUT.REA.VALUE = [0.485, 0.456, 0.406]
# Random Patch
_C.INPUT.RPT = CN()
_C.INPUT.RPT.ENABLED = False
_C.INPUT.RPT.PROB = 0.5
# Mutual tansform
_C.INPUT.MUTUAL = CN()
_C.INPUT.MUTUAL.ENABLED = False
_C.INPUT.MUTUAL.TIMES = 4

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training
_C.DATASETS.NAMES = ("Market1501",)
# List of the dataset names for testing
_C.DATASETS.TESTS = ("Market1501",)
# Combine trainset and testset joint training
_C.DATASETS.COMBINEALL = False

# ----------------------------------------------------------------------------- #
# CUHK03 specific parameters
# ----------------------------------------------------------------------------- #
_C.DATASETS.CUHK03 = CN()
# CUHK03 label or detected
_C.DATASETS.CUHK03.LABELED = False
# new split protocol or not
_C.DATASETS.CUHK03.CLASSIC_SPLIT = False

# ----------------------------------------------------------------------------- #
# Market1501 specific parameters
# ----------------------------------------------------------------------------- #
_C.DATASETS.MARKET1501 = CN()
_C.DATASETS.MARKET1501.ENABLE_500K = False

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Sampler name, support BalancedIdentitySampler, NaiveIdentitySampler, RandomMultipleGallerySampler, TrainingSampler, InferenceSampler
_C.DATALOADER.SAMPLER_NAME = "BalancedIdentitySampler"
# Number of instance for each person
_C.DATALOADER.NUM_INSTANCE = 4
_C.DATALOADER.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Pseudo label
# -----------------------------------------------------------------------------
_C.PSEUDO = CN()
_C.PSEUDO.ENABLED = False
_C.PSEUDO.NAME = 'dbscan'  # 'kmeans', 'cdp'
_C.PSEUDO.UNSUP = (0,)  # unsupervised index for training datasets, support MMT.
_C.PSEUDO.CLUSTER_EPOCH = 2
_C.PSEUDO.USE_OUTLIERS = False  # True for SpCL
_C.PSEUDO.NORM_FEAT = True
_C.PSEUDO.NORM_CENTER = True
_C.PSEUDO.SEARCH_TYPE = 0
_C.PSEUDO.NUM_CLUSTER = [500, ]
_C.PSEUDO.RESET_OPT = False
_C.PSEUDO.WITH_CLASSIFIER = True # False for SpCL

_C.PSEUDO.DBSCAN = CN()
_C.PSEUDO.DBSCAN.BASE = 'eps' # perform clustering based on which metric, 'eps' or 'rho'
_C.PSEUDO.DBSCAN.EPS = [0.6, ]  # [0.58, 0.6, 0.62] for SpCL
_C.PSEUDO.DBSCAN.RHO = 1.6e-3 # set eps as the top_num mean
_C.PSEUDO.DBSCAN.MIN_SAMPLES = 4
_C.PSEUDO.DBSCAN.DIST_METRIC = 'jaccard'
_C.PSEUDO.DBSCAN.K1 = 30
_C.PSEUDO.DBSCAN.K2 = 6

_C.PSEUDO.CDP = CN()
_C.PSEUDO.CDP.K = [25]
_C.PSEUDO.CDP.STRATEGY = 'vote'
_C.PSEUDO.CDP.VOT = CN()
_C.PSEUDO.CDP.VOT.THRESHOLD = [0.66]
_C.PSEUDO.CDP.PROPAGATION = CN()
_C.PSEUDO.CDP.PROPAGATION.MAX_SIZE = 600
_C.PSEUDO.CDP.PROPAGATION.STEP = 0.05
_C.PSEUDO.CDP.PROPAGATION.MAX_ITER = 100

# Memory related options for Self-paced learning
# Temperature for scaling contrastive loss
_C.PSEUDO.MEMORY = CN()
_C.PSEUDO.MEMORY.TEMP = 0.05
_C.PSEUDO.MEMORY.MOMENTUM = 0.2
_C.PSEUDO.MEMORY.WEIGHTED = False
_C.PSEUDO.MEMORY.WEIGHT_MASK_TOPK = -1
_C.PSEUDO.MEMORY.SOFT_LABEL_START_EPOCH = 0
_C.PSEUDO.MEMORY.SOFT_LABEL = False # generate soft pseudo label for training

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

# AUTOMATIC MIXED PRECISION
_C.SOLVER.FP16_ENABLED = False

# Optimizer
_C.SOLVER.OPT = "Adam"

_C.SOLVER.MAX_EPOCH = 120

_C.SOLVER.BASE_LR = 3e-4
_C.SOLVER.BIAS_LR_FACTOR = 1.
_C.SOLVER.HEADS_LR_FACTOR = 1.

_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.NESTEROV = True

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.

# Multi-step learning rate options
_C.SOLVER.SCHED = "MultiStepLR"

_C.SOLVER.DELAY_EPOCHS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = [30, 55]

# Cosine annealing learning rate options
_C.SOLVER.ETA_MIN_LR = 1e-7

# Warmup options
_C.SOLVER.WARMUP_FACTOR = 0.1
_C.SOLVER.WARMUP_EPOCHS = 10
_C.SOLVER.WARMUP_METHOD = "linear"

# Backbone freeze iters
_C.SOLVER.FREEZE_ITERS = 0

# FC freeze iters
_C.SOLVER.FREEZE_FC_ITERS = 0

# SWA options
_C.SOLVER.SWA = CN()
_C.SOLVER.SWA.ENABLED = False
_C.SOLVER.SWA.ITER = 10
_C.SOLVER.SWA.PERIOD = 2
_C.SOLVER.SWA.LR_FACTOR = 10.
_C.SOLVER.SWA.ETA_MIN_LR = 3.5e-6
_C.SOLVER.SWA.LR_SCHED = False

_C.SOLVER.CHECKPOINT_PERIOD = 20

# Logger frequence
_C.SOLVER.LOG_ITERS = 200

# Number of images per batch across all machines.
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

# iters per epoch
_C.SOLVER.ITERS_PER_EPOCH = 0

# ----------------------------------------------------------------------------
# Test
# ----------------------------------------------------------------------------
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()

_C.TEST.EVAL_PERIOD = 20

# Run testing on validation or testing dataset in training stage,
# False means run testing on query and gallery,
# True means run on val split from training dataste.
_C.TEST.DO_VAL = False
# Metric for finding the best model
_C.TEST.METRIC_NAMES = ('Rank-1', 'mAP')

# Number of images per batch in one process.
_C.TEST.IMS_PER_BATCH = 64
_C.TEST.METRIC = "cosine"
_C.TEST.ROC_ENABLED = False
_C.TEST.FLIP_ENABLED = False

# Average query expansion
_C.TEST.AQE = CN()
_C.TEST.AQE.ENABLED = False
_C.TEST.AQE.ALPHA = 3.0
_C.TEST.AQE.QE_TIME = 1
_C.TEST.AQE.QE_K = 5

# Re-rank
_C.TEST.RERANK = CN()
_C.TEST.RERANK.ENABLED = False
_C.TEST.RERANK.K1 = 20 # 26
_C.TEST.RERANK.K2 = 6 # 8
_C.TEST.RERANK.LAMBDA = 0.3 # 0.2
_C.TEST.RERANK.TYPE = 'jaccard'  # 'gnn'

# Precise batchnorm
_C.TEST.PRECISE_BN = CN()
_C.TEST.PRECISE_BN.ENABLED = False
_C.TEST.PRECISE_BN.DATASET = 'Market1501'
_C.TEST.PRECISE_BN.NUM_ITER = 300

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "logs/"

# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
_C.CUDNN_BENCHMARK = False

# deterministic
_C.DETERMINISTIC = True

# Save the project in log path
_C.SAVE_PROJECT = True

# random seed
_C.SEED = 1
