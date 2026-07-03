"""
SDANet configuration.

All hyperparameters for the Spatial-Distortion-Aware Network.
"""

# ---------------------------------------------------------------------------
# Backend & device
# ---------------------------------------------------------------------------
BACKEND = "pytorch"       # "pytorch" | "jittor"
DEVICE = "cuda" if BACKEND == "pytorch" else "cuda"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
NUM_WORKERS = 4
NUM_CLASSES = 1            # person (most fisheye datasets are person-only)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 128
INPUT_SIZE = 640            # input image size (square: 640×640)

# Gradient accumulation: when enabled, use smaller per-step batches
# and replace BatchNorm with GroupNorm (BN is unstable with small micro-batches).
USE_ACCUMULATION_STEP = True
STEP_BATCH_SIZE = 32 if USE_ACCUMULATION_STEP else BATCH_SIZE
GN_NUM_GROUPS = 32          # GroupNorm groups (auto-clamped to divisor of channels)

LOAD_FROM_PRETRAIN = True

LR = 0.0001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
MAX_ITER = 6000             # fine-tuning iterations on fisheye datasets
WARMUP_ITERS = 1000

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

# Anchors (same as YOLOv3, pre-scaled)
# Will be auto-computed from dataset if not specified
ANCHORS = [
    [(10, 13), (16, 30), (33, 23)],     # small  scale (stride 8)
    [(30, 61), (62, 45), (59, 119)],    # medium scale (stride 16)
    [(116, 90), (156, 198), (373, 326)], # large  scale (stride 32)
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
