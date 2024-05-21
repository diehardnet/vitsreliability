MAXIMUM_ERRORS_PER_ITERATION = 512
MAXIMUM_INFOS_PER_ITERATION = 512

# Device capability for pytorch
MINIMUM_DEVICE_CAPABILITY = 5  # Maxwell
MINIMUM_DEVICE_CAPABILITY_TORCH_COMPILE = 7  # Volta

CLASSIFICATION_CRITICAL_TOP_K = 1
# FORCE the gpu to be present
GPU_DEVICE = "cuda:0"
CPU = "cpu"

# Classification CNNs
RESNET50D_IMAGENET_TIMM = "resnet50d"
EFFICIENTNET_B7_TIMM = "tf_efficientnet_b7"
CNN_CONFIGS = [
    RESNET50D_IMAGENET_TIMM,
    EFFICIENTNET_B7_TIMM
]

# Classification ViTs
# Base from the paper
VIT_BASE_PATCH16_224 = "vit_base_patch16_224"
VIT_BASE_PATCH16_384 = "vit_base_patch16_384"
# Same model as before see https://github.com/huggingface/pytorch-image-models/
# blob/51b262e2507c50b277b5f6caa0d6bb7a386cba2e/timm/models/vision_transformer.py#L1864
VIT_BASE_PATCH32_224_SAM = "vit_base_patch32_224.sam"
VIT_BASE_PATCH32_384 = "vit_base_patch32_384"

# Large models
# https://pypi.org/project/timm/0.8.19.dev0/
VIT_LARGE_PATCH14_CLIP_336 = "vit_large_patch14_clip_336.laion2b_ft_in12k_in1k"
VIT_LARGE_PATCH14_CLIP_224 = "vit_large_patch14_clip_224.laion2b_ft_in12k_in1k"
# Huge models
VIT_HUGE_PATCH14_CLIP_336 = "vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k"
VIT_HUGE_PATCH14_CLIP_224 = "vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k"
# Max vit
# https://huggingface.co/timm/maxvit_large_tf_384.in21k_ft_in1k
# https://huggingface.co/timm/maxvit_large_tf_512.in21k_ft_in1k
MAXVIT_LARGE_TF_384 = 'maxvit_large_tf_384.in21k_ft_in1k'
MAXVIT_LARGE_TF_512 = 'maxvit_large_tf_512.in21k_ft_in1k'
# Davit
# https://huggingface.co/timm/davit_small.msft_in1k
# https://huggingface.co/timm/davit_base.msft_in1k
DAVIT_BASE = 'davit_base.msft_in1k'
DAVIT_SMALL = 'davit_small.msft_in1k'
# SwinV2
# https://huggingface.co/timm/swinv2_base_window12to16_192to256.ms_in22k_ft_in1k
# https://huggingface.co/timm/swinv2_base_window12to24_192to384.ms_in22k_ft_in1k
# https://huggingface.co/timm/swinv2_large_window12to16_192to256.ms_in22k_ft_in1k
# https://huggingface.co/timm/swinv2_large_window12to24_192to384.ms_in22k_ft_in1k
SWINV2_BASE_WINDOW12TO16_192to256_22KFT1K = 'swinv2_base_window12to16_192to256.ms_in22k_ft_in1k'
SWINV2_BASE_WINDOW12TO24_192to384_22KFT1K = 'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k'
SWINV2_LARGE_WINDOW12TO16_192to256_22KFT1K = 'swinv2_large_window12to16_192to256.ms_in22k_ft_in1k'
SWINV2_LARGE_WINDOW12TO24_192to384_22KFT1K = 'swinv2_large_window12to24_192to384.ms_in22k_ft_in1k'
# FasterTransformer Swin models
SWIN_BASE_PATCH4_WINDOW12_384 = "swin_base_patch4_window12_384"
SWIN_BASE_PATCH4_WINDOW7_224 = "swin_base_patch4_window7_224"

# EVA
# https://huggingface.co/timm/eva02_large_patch14_448.mim_m38m_ft_in1k
EVA_LARGE_PATCH14_448_MIM = "eva02_large_patch14_448.mim_in22k_ft_in22k_in1k"
EVA_BASE_PATCH14_448_MIM = "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k"
EVA_SMALL_PATCH14_336_MIN = "eva02_small_patch14_336.mim_in22k_ft_in1k"

# Efficient former
# https://huggingface.co/timm/efficientformer_l1.snap_dist_in1k
EFFICIENTFORMER_L1 = "efficientformer_l1.snap_dist_in1k"
EFFICIENTFORMER_L3 = "efficientformer_l3.snap_dist_in1k"
EFFICIENTFORMER_L7 = "efficientformer_l7.snap_dist_in1k"

# DeiT models
# https://huggingface.co/timm/deit_base_patch16_224
# https://huggingface.co/timm/deit_base_patch16_384
DEIT_BASE_PATCH16_224 = "deit_base_patch16_224"
DEIT_BASE_PATCH16_384 = "deit_base_patch16_384"

VIT_CLASSIFICATION_CONFIGS = [
    VIT_BASE_PATCH16_224,
    # VIT_BASE_PATCH32_224_SAM,
    VIT_BASE_PATCH16_384,
    VIT_LARGE_PATCH14_CLIP_336,  # --> Hardening not ready
    VIT_LARGE_PATCH14_CLIP_224,
    VIT_HUGE_PATCH14_CLIP_336,  # --> Hardening not ready
    VIT_HUGE_PATCH14_CLIP_224,
    MAXVIT_LARGE_TF_384,
    MAXVIT_LARGE_TF_512,
    DAVIT_BASE,  # --> Hardening not ready
    DAVIT_SMALL,  # --> Hardening not ready
    SWINV2_LARGE_WINDOW12TO16_192to256_22KFT1K,
    SWINV2_LARGE_WINDOW12TO24_192to384_22KFT1K,
    SWINV2_BASE_WINDOW12TO16_192to256_22KFT1K,
    SWINV2_BASE_WINDOW12TO24_192to384_22KFT1K,
    EVA_LARGE_PATCH14_448_MIM,
    EVA_BASE_PATCH14_448_MIM,
    EVA_SMALL_PATCH14_336_MIN,
    # EFFICIENTFORMER_L1,        --> Hardening not ready
    # EFFICIENTFORMER_L3,        --> Hardening not ready
    # EFFICIENTFORMER_L7         --> Hardening not ready
    SWIN_BASE_PATCH4_WINDOW12_384,
    SWIN_BASE_PATCH4_WINDOW7_224,
    DEIT_BASE_PATCH16_224,
    DEIT_BASE_PATCH16_384,
]

INT8_MODELS = [
    SWIN_BASE_PATCH4_WINDOW7_224,
]

GROUNDING_DINO_SWINT_OGC = "groundingdino_swint_ogc"
GROUNDING_DINO_SWINB_COGCOOR = "groundingdino_swinb_cogcoor"
VITS_MULTIMODAL_CONFIGS = [
    GROUNDING_DINO_SWINT_OGC,
    GROUNDING_DINO_SWINB_COGCOOR
]

ALL_POSSIBLE_MODELS = CNN_CONFIGS + VIT_CLASSIFICATION_CONFIGS + VITS_MULTIMODAL_CONFIGS

# This max size will determine the max number of images in all datasets
DATASET_MAX_SIZE = 50000

IMAGENET = "imagenet"
COCO = "coco"
CIFAR10 = "cifar10"
CIFAR100 = "cifar100"

DATASETS = [IMAGENET, COCO, CIFAR100, CIFAR10]

CIFAR_DATASET_DIR = "/home/carol/cifar"
IMAGENET_DATASET_DIR = "/home/ILSVRC2012"
COCO_DATASET_DIR = "/home/COCO"
COCO_DATASET_VAL = f"{COCO_DATASET_DIR}/val2017"
COCO_DATASET_ANNOTATIONS = f"{COCO_DATASET_DIR}/annotations/instances_val2017.json"

# Checkpoints directory
PTQ4VIT_MODELS_DIR = "/home/PTQ4ViT/models"

# File to save last status of the benchmark when log helper not active
TMP_CRASH_FILE = "/tmp/vitsreliability_crash_file.txt"

# Seed used for sampling
TORCH_SEED = 0

# code types that can be evaluated
GROUNDING_DINO, MAXIMALS, SELECTIVE_ECC, VITS = "grounding_dino", "maximals", "selective_ecc", "vits"
MICROBENCHMARK = "microbenchmark"

ALL_SETUP_TYPES = [
    GROUNDING_DINO, MAXIMALS, SELECTIVE_ECC, VITS, MICROBENCHMARK
]

FP32, FP16, BFLOAT16, INT8 = "fp32", "fp16", "bfloat16", "int8"

ALLOWED_MODEL_PRECISIONS = [
    FP32, FP16, BFLOAT16, INT8
]

# Micro benchmarks setup
ATTENTION, BLOCK, MLP, WINDOW_ATTENTION = 'Attention', 'Block', 'Mlp', 'WindowAttention'
MICROBENCHMARK_MODULES = [ATTENTION, BLOCK, MLP, WINDOW_ATTENTION]

SWIN_BLOCK = 'SwinTransformerBlock'

ATTENTION, BLOCK, MLP, WINDOW_ATTENTION = 'Attention', 'Block', 'Mlp', 'WindowAttention'
MICROBENCHMARK_MODULES = [ATTENTION, BLOCK, MLP, WINDOW_ATTENTION, SWIN_BLOCK]

INT8_CKPT_DIR = "/home/int8_ckpts/"
