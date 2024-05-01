SETUP_PATH = /home/carol/vitsreliability
DATA_DIR = $(SETUP_PATH)/data
MODEL_NAME = groundingdino_swint_ogc

CHECKPOINTS = $(DATA_DIR)/checkpoints

TARGET = main.py
GOLD_PATH = $(DATA_DIR)/$(MODEL_NAME).pt

CFG_PATH = /home/carol/vitsreliability/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
CHECKPOINT_PATH = $(DATA_DIR)/weights_grouding_dino/groundingdino_swint_ogc.pth

GET_DATASET=0

ifeq ($(GET_DATASET), 1)
ADDARGS= --downloaddataset
endif

all: test generate

BATCH_SIZE = 1
TEST_SAMPLES=8
ITERATIONS=10
PRECISION = fp32

generate:
	$(SETUP_PATH)/$(TARGET) --iterations $(ITERATIONS) \
                  --testsamples $(TEST_SAMPLES) \
              --goldpath $(GOLD_PATH) \
              --checkpointdir $(CHECKPOINTS) \
              --generate $(ADDARGS)

test:
	$(SETUP_PATH)/$(TARGET) --iterations $(ITERATIONS) \
                  --testsamples $(TEST_SAMPLES) \
              --goldpath $(GOLD_PATH) \
              --checkpointdir $(CHECKPOINTS)

generate_dino:
	PYTHONPATH=/home/carol/vitsreliability/GroundingDINO:${PYTHONPATH} \
	$(SETUP_PATH)/$(TARGET) --iterations $(ITERATIONS) --precision $(PRECISION) \
                --testsamples $(TEST_SAMPLES)  --generate \
				--goldpath $(GOLD_PATH) \
				--checkpointpath $(CHECKPOINT_PATH) \
				--configpath $(CFG_PATH) --batchsize $(BATCH_SIZE) \
				--setup_type grounding_dino --model $(MODEL_NAME)

test_dino:
	PYTHONPATH=/home/carol/vitsreliability/GroundingDINO:${PYTHONPATH} \
	$(SETUP_PATH)/$(TARGET) --iterations $(ITERATIONS) --precision $(PRECISION) \
                --testsamples $(TEST_SAMPLES) --hardenedid \
				--goldpath $(GOLD_PATH) \
				--checkpointpath $(CHECKPOINT_PATH) \
				--configpath $(CFG_PATH) --batchsize $(BATCH_SIZE) \
				--setup_type grounding_dino --model $(MODEL_NAME)