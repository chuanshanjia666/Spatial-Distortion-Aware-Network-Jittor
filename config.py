"""
SDANet configuration.

All hyperparameters for the Spatial-Distortion-Aware Network.
"""

# ---------------------------------------------------------------------------
# Backend & device
# ---------------------------------------------------------------------------
# BACKEND = "pytorch"       # "pytorch" | "jittor"
BACKEND = "pytorch"
DEVICE = "cuda" if BACKEND == "pytorch" else "cuda"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
NUM_WORKERS = 4

# Dataset configuration: each entry uses ``preset[split]`` syntax.
#   preset   → one of habbof, cepdof, wepdtof, fisheye8k (see DATASET_PRESETS)
#   split    → "all", "train", "val", or "test" (maps to annotations/<split>.json)
#
# PFDAug is applied to TRAIN_DATASETS only; val/test datasets always use raw images.
#
# Examples:
#   TRAIN_DATASETS = ["habbof[all]"]                          # habbof all→train, aug on
#   VAL_DATASETS   = ["cepdof[all]"]                          # cepdof all→val,   aug off
#   TEST_DATASETS  = ["wepdtof[test]"]                        # wepdtof test→test, aug off
#
#   # Multi-dataset training with different val splits:
#   TRAIN_DATASETS = ["habbof[train]", "cepdof[train]"]
#   VAL_DATASETS   = ["habbof[val]",   "cepdof[val]"]
#   TEST_DATASETS  = ["wepdtof[test]", "fisheye8k[test]"]

TRAIN_DATASETS = ["habbof[train]"]
VAL_DATASETS   = ["habbof[val]"]
TEST_DATASETS  = ["habbof[test]"]                       # empty → skip test evaluation

# ---------------------------------------------------------------------------
# PFDAug — online distortion augmentation (paper Section IV)
# ---------------------------------------------------------------------------
# Applied only to training split; val/test use raw images.
PFDAUG_ENABLED = True
PFDAUG_K = (0.1,0.2,0.3,0.35,0.4,0.45,0.48,0.49,0.5)   # distortion coefficient tuple, randomly sampled per image
PFDAUG_P = 0.5             # probability per sample

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
INPUT_SIZE = 416            # input image size (square: 640×640)

# Gradient accumulation: when enabled, use smaller per-step batches
# and replace BatchNorm with GroupNorm (BN is unstable with small micro-batches).
USE_ACCUMULATION_STEP = False
STEP_BATCH_SIZE = 8 if USE_ACCUMULATION_STEP else BATCH_SIZE
GN_NUM_GROUPS = 32          # GroupNorm groups (auto-clamped to divisor of channels)

LOAD_FROM_PRETRAIN = True

LR = 0.0001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
MAX_ITER = 6000             # fine-tuning iterations on fisheye datasets
WARMUP_ITERS = 20
EPOCHS = 50                 # number of training epochs
RESUME = None               # path to checkpoint for resuming (None = train from scratch)
OUTPUT_DIR = "output"   # directory for saving checkpoints

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

# SDABlock coefficient predict: GAP → FC1 → ReLU → FC2 → softmax
SDA_FC_RATIO = 4            # FC hidden dim = C / ratio

# Spatially Separate Strategy (SSS)
# ---------------------------------------------------------------------------
# Split feature map into grid; each cell uses its own SDABlock.
SSS_GRID = 3                # 3×3 grid (only used if SSS_ENABLED)
SSS_ENABLED = False         # toggle per experiment; paper applies to head

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
