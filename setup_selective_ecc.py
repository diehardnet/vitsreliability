import argparse
import collections
import logging
import os
import random
import re
from typing import Tuple, List, Dict, Union

import timm
import torch
from torchvision import datasets as tv_datasets
from torchvision import transforms as tv_transforms

import configs
import console_logger
import dnn_log_helper
import common
from setup_base import SetupBase


class SetupSelectiveECC(SetupBase):
    def __init__(self, args: argparse.Namespace, output_logger: logging.Logger):
        super().__init__(args=args, output_logger=output_logger)

        self.model = None
        self.golden = list()
        self.input_list = list()
        self.gt_targets = list()
        self.correctness_threshold = 0.6  # Based on the whole dataset accuracy. Used only for golden generate part

    @property
    def num_batches(self):
        return len(self.input_list)

    def __call__(self, batch_id, **kwargs):
        return self.model(self.input_list[batch_id], captions=self.input_captions)

    def load_model(self) -> None:
        pass

    def load_dataset(self) -> None:
        pass

    def check_dnn_accuracy(self) -> None:
        pass

    def compare_inference(self, output: dict, batch_id) -> int:
        pass

    def load_data_at_test(self):
        pass

    def save_setup_data_to_gold_file(self):
        pass

    def clear_gpu_memory_and_reload(self):
        if self.output_logger:
            self.output_logger.info("RELOADING THE MODEL AND THE INPUTS AFTER ERROR")
        del self.input_list
        del self.model
        # Free cuda memory
        torch.cuda.empty_cache()
        self.load_data_at_test()

    @staticmethod
    def copy_to_cpu(dnn_output):
        pass

    def post_inference_process(self, dnn_output_cpu, batch_id) -> int:
        pass
