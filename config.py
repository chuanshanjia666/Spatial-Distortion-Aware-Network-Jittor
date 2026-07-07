# BACKEND = "pytorch"
BACKEND = "jittor"
DEVICE = "cuda"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
NUM_WORKERS = 4
TRAIN_DATASETS = ["cepdof[train]","wepdtof[train]"]
# TRAIN_DATASETS = ["habbof[train]"]
VAL_DATASETS   = ["habbof[val]"]
TEST_DATASETS  = ["habbof[test]"]

# ---------------------------------------------------------------------------
# PFDAug — online distortion augmentation (paper Section IV)
# ---------------------------------------------------------------------------
# Applied only to training split; val/test use raw images.
PFDAUG_ENABLED = True
PFDAUG_K = (0.1,0.2,0.3,0.35,0.4,0.45,0.48,0.49,0.5)   # distortion coefficient tuple, randomly sampled per image
PFDAUG_P = 0.2             # probability per sample

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
INPUT_SIZE = 416            # input image size (square: 608×608)

# Gradient accumulation: when enabled, use smaller per-step batches
# and replace BatchNorm with GroupNorm (BN is unstable with small micro-batches).
USE_ACCUMULATION_STEP = True
STEP_BATCH_SIZE = 8 if USE_ACCUMULATION_STEP else BATCH_SIZE
GN_NUM_GROUPS = 32          # GroupNorm groups (auto-clamped to divisor of channels)

# Mixed precision (FP16) — reduces GPU memory, speeds up training
# PyTorch: torch.cuda.amp.GradScaler + autocast (FP32 master weights + FP16 compute)
# Jittor:  jt.flags.auto_mixed_precision_level = 4 (prefer16). Weights → float_auto(),
#          inputs → float_auto(). Reduce ops (sum/mean) internally promote to fp32;
#          explicit fp32 tensors (loss) stay fp32.  ~2× memory savings on both backends.
USE_FP16 = True

LOAD_FROM_PRETRAIN = True

LR = 0.0001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
MAX_ITER = 6000             # fine-tuning iterations on fisheye datasets
WARMUP_ITERS = 20
VALIDATE_INTERVAL = 100        # validate every N iterations
SAVE_INTERVAL = 100            # save checkpoint every N iterations
RESUME = None               # explicit checkpoint path (takes priority over AUTO_RESUME)
AUTO_RESUME = True          # if True and RESUME is None, auto-load latest checkpoint from OUTPUT_DIR
OUTPUT_DIR = "output"       # directory for saving checkpoints

# Cosine annealing LR scheduler
USE_COSINE_SCHEDULER = True  # if True, cosine anneal from LR to MIN_LR over MAX_ITER
MIN_LR = 1e-6                # minimum LR for cosine annealing

# ---------------------------------------------------------------------------
# Model architecture (SDANet)
# ---------------------------------------------------------------------------

# Backbone: DarkNet53
# ---------------------------------------------------------------------------
# Output strides: [8, 16, 32] → feature map scales relative to input
#
# Pretrained backbone weights (COCO-pretrained, backbone only):
#   python pretrain/download_pretrained.py
DARKNET53_PRETRAINED = "pretrain/darknet53_backbone.pth"

# Neck (FPN-like)
# ---------------------------------------------------------------------------
FPN_CHANNELS = [256, 512, 1024]   # channels after backbone at 3 scales
NECK_OUT_CHANNELS = 128           # channels after FPN reduction

# SDABlock
# ---------------------------------------------------------------------------
# Base kernel sizes used to generate the dynamic kernel.
# (1, 3, 5) is the recommended default from the paper.
# Other options: (1, 3), (1, 3, 3, 5), (1, 3, 5, 7)
SDA_BASE_KERNELS = (1, 3, 5)

SDA_FC_RATIO = 128            # FC hidden dim = C / ratio


SSS_GRID = 3                # 3×3 grid (only used if SSS_ENABLED)
SSS_ENABLED = True          # toggle per experiment; paper applies to head

# Head (detection)
# ---------------------------------------------------------------------------
NUM_ANCHORS = 3             # anchors per scale, same as YOLOv3
# Output per anchor: [cx, cy, w, h, θ, obj_conf] — oriented bounding box
BOX_FIELDS = 6              # cx, cy, w, h, θ, confidence

# Anchors (per-scale, auto-computed from k-means on training datasets)
# Default YOLOv3 anchors below; to re-cluster for your data run:
#     python util/anchor_cluster.py
# The training script will automatically load cached anchors if available.
ANCHORS_AUTO_CLUSTER = True  # if True, load from cache; else use ANCHORS below
ANCHORS = [
    [(10, 13), (16, 30), (33, 23)],       # small  scale (stride 8)
    [(30, 61), (62, 45), (59, 119)],      # medium scale (stride 16)
    [(116, 90), (156, 198), (373, 326)],  # large  scale (stride 32)
]

# Strides for each detection scale
STRIDES = [8, 16, 32]

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
BOX_LOSS_WEIGHT = 1.0
CLS_LOSS_WEIGHT = 1.0
OBJ_LOSS_WEIGHT = 1.0
IOU_THRESH = 0.5            # AP@50

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
CONF_THRESH = 0.3
NMS_IOU_THRESH = 0.45
MAX_DETECTIONS = 300

# ---------------------------------------------------------------------------
# Random seed (for reproducibility)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
