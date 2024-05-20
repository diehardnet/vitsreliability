SETUP_PATH = /home/carol/vitsreliability
DATA_DIR = $(SETUP_PATH)/data

# GroundingDINO-B
#MODEL_NAME = groundingdino_swinb_cogcoor
#CFG_PATH = /home/carol/vitsreliability/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py
#CHECKPOINT_PATH = $(DATA_DIR)/weights_grounding_dino/groundingdino_swinb_cogcoor.pth

# GroundingDINO-T
MODEL_NAME = groundingdino_swinb_cogcoor
CFG_PATH = /home/carol/vitsreliability/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
CHECKPOINT_PATH = $(DATA_DIR)/weights_grounding_dino/groundingdino_swint_ogc.pth

#MODEL_NAME = vit_base_patch16_224
#CFG_PATH = /home/carol/vitsreliability/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py
#CHECKPOINT_PATH = $(DATA_DIR)

CHECKPOINTS = $(DATA_DIR)/checkpoints
BATCH_SIZE = 1
TEST_SAMPLES=4
ITERATIONS=1
PRECISION = fp32
FLOAT_THRESHOLD = 1e-3
#SETUP_TYPE = microbenchmark
SETUP_TYPE = grounding_dino
MICRO_TYPE = Attention

ENV_VARS = CUBLAS_WORKSPACE_CONFIG=:4096:8

TARGET = main.py
GOLD_PATH = $(DATA_DIR)/$(MODEL_NAME).pt

ENABLE_MAXIMALS=0

ifeq ($(ENABLE_MAXIMALS), 1)
ADDARGS = --hardenedid
endif

SAVE_LOGITS = 1
ifeq ($(SAVE_LOGITS), 1)
ADDARGS += --savelogits
endif

all: generate test

generate:
	$(ENV_VARS) $(SETUP_PATH)/$(TARGET) --iterations $(ITERATIONS) --precision $(PRECISION) \
                --testsamples $(TEST_SAMPLES)  --generate \
				--goldpath $(GOLD_PATH) \
				--checkpointpath $(CHECKPOINT_PATH) \
				--configpath $(CFG_PATH) --batchsize $(BATCH_SIZE) \
				--setup_type $(SETUP_TYPE) --model $(MODEL_NAME) \
              	$(ADDARGS) --floatthreshold $(FLOAT_THRESHOLD) --loghelperinterval 1 --microop $(MICRO_TYPE)

test:
	$(ENV_VARS) $(SETUP_PATH)/$(TARGET) --iterations $(ITERATIONS) --precision $(PRECISION) \
                --testsamples $(TEST_SAMPLES) \
				--goldpath $(GOLD_PATH) \
				--checkpointpath $(CHECKPOINT_PATH) \
				--configpath $(CFG_PATH) --batchsize $(BATCH_SIZE) \
				--setup_type $(SETUP_TYPE) --model $(MODEL_NAME) \
              	$(ADDARGS) --floatthreshold $(FLOAT_THRESHOLD) --loghelperinterval 1 --microop $(MICRO_TYPE)
