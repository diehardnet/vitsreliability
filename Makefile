SETUP_PATH = /home/carol/maximals
DATA_DIR = $(SETUP_PATH)/data
CONFIG_NAME = vit_base_resnet50_384

CHECKPOINTS = $(DATA_DIR)/checkpoints

TARGET = setuppuretorch.py
GOLD_PATH = $(DATA_DIR)/$(CONFIG_NAME).pt

GET_DATASET=0

ifeq ($(GET_DATASET), 1)
ADDARGS= --downloaddataset
endif

all: test generate

TEST_SAMPLES=128
ITERATIONS=10

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