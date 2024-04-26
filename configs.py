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

# EVA
# https://huggingface.co/timm/eva02_large_patch14_448.mim_m38m_ft_in1k
EVA_LARGE_PATCH14_448_MIM = "eva02_large_patch14_448.mim_in22k_ft_in22k_in1k"
EVA_BASE_PATCH14_448_MIM = "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k"
EVA_SMALL_PATCH14_448_MIN = "eva02_small_patch14_336.mim_in22k_ft_in1k"

# Efficient former
# https://huggingface.co/timm/efficientformer_l1.snap_dist_in1k
EFFICIENTFORMER_L1 = "efficientformer_l1.snap_dist_in1k"
EFFICIENTFORMER_L3 = "efficientformer_l3.snap_dist_in1k"
EFFICIENTFORMER_L7 = "efficientformer_l7.snap_dist_in1k"

VIT_CLASSIFICATION_CONFIGS = [
    VIT_BASE_PATCH16_224,
    VIT_BASE_PATCH32_224_SAM,
    VIT_BASE_PATCH16_384,
    # VIT_LARGE_PATCH14_CLIP_336, --> Hardening not ready
    VIT_LARGE_PATCH14_CLIP_224,
    # VIT_HUGE_PATCH14_CLIP_336,  --> Hardening not ready
    VIT_HUGE_PATCH14_CLIP_224,
    MAXVIT_LARGE_TF_384,
    MAXVIT_LARGE_TF_512,
    # DAVIT_BASE,                 --> Hardening not ready
    # DAVIT_SMALL,                --> Hardening not ready
    SWINV2_LARGE_WINDOW12TO16_192to256_22KFT1K,
    SWINV2_LARGE_WINDOW12TO24_192to384_22KFT1K,
    SWINV2_BASE_WINDOW12TO16_192to256_22KFT1K,
    SWINV2_BASE_WINDOW12TO24_192to384_22KFT1K,
    EVA_LARGE_PATCH14_448_MIM,
    EVA_BASE_PATCH14_448_MIM,
    EVA_SMALL_PATCH14_448_MIN,
    # EFFICIENTFORMER_L1,        --> Hardening not ready
    # EFFICIENTFORMER_L3,        --> Hardening not ready
    # EFFICIENTFORMER_L7         --> Hardening not ready
]

ALL_POSSIBLE_MODELS = CNN_CONFIGS + VIT_CLASSIFICATION_CONFIGS

# Set the supported goals
CLASSIFICATION = "classify"
SEGMENTATION = "segmentation"
MICROBENCHMARK = "microbenchmark"

DNN_GOAL = {
    # Classification CNNs
    **{k: CLASSIFICATION for k in CNN_CONFIGS},
    # Classification transformer
    **{k: CLASSIFICATION for k in VIT_CLASSIFICATION_CONFIGS},
    # Segmentation nets
}

# Error threshold for the test
DNN_THRESHOLD = {
    CLASSIFICATION: 1.0e-3,
    SEGMENTATION: 1.0e-3,
    MICROBENCHMARK: 1.0e-3,
}

ITERATION_INTERVAL_LOG_HELPER_PRINT = {
    # imagenet not so small
    **{k: 10 for k in CNN_CONFIGS},
    **{k: 10 for k in VIT_CLASSIFICATION_CONFIGS},
    # Segmentation nets, huge
    MICROBENCHMARK: 100
}

# This max size will determine the max number of images in all datasets
DATASET_MAX_SIZE = 50000

IMAGENET = "imagenet"
COCO = "coco"
DATASETS = {
    CLASSIFICATION: IMAGENET,
    SEGMENTATION: COCO
}

CLASSES = {
    IMAGENET: 1000
}

IMAGENET_DATASET_DIR = "/home/ILSVRC2012"
COCO_DATASET_DIR = "/home/COCO"
COCO_DATASET_VAL = f"{COCO_DATASET_DIR}/val2017"
COCO_DATASET_ANNOTATIONS = f"{COCO_DATASET_DIR}/annotations/instances_val2017.json"

# File to save last status of the benchmark when log helper not active
TMP_CRASH_FILE = "/tmp/vitsreliability_crash_file.txt"

# TensorRT file pattern
TENSORRT_FILE_POSFIX = "_tensorrt.ts"

# Seed used for sampling
SAMPLER_SEED = 2147483647

# code types that can be evaluated
GROUDING_DINO, MAXIMALS, SELECTIVE_ECC = range(3)
