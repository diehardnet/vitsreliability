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
from setup_base import SetupBaseImageNet


class SetupVits(SetupBaseImageNet):
    def __init__(self, args: argparse.Namespace, output_logger: logging.Logger):
        super().__init__(args=args, output_logger=output_logger)

        self.model = None
        self.golden = list()
        self.input_list = list()
        self.gt_targets = list()
        # self.correctness_threshold = 0.6  # Based on the whole dataset accuracy. Used only for golden generate part

    @property
    def num_batches(self):
        return len(self.input_list)

    # def __call__(self, batch_id, **kwargs):
    #     return self.model(self.input_list[batch_id], captions=self.input_captions)

    def load_model(self) -> None:
        model = timm.create_model(self.model_name, pretrained=True)
        
        if self.precision == configs.FP16:
            model = model.half()
        elif self.precision == configs.INT8:
            raise ValueError("INT8 is not supported for ViTs")

        model.eval()
        model.to(configs.GPU_DEVICE)
        model.zero_grad(set_to_none=True)
        self.model = model
        self.transforms = self.__get_vit_transforms()
        if self.precision == configs.FP16:
            class CustomToFP16:
                def __call__(self, tensor_in):
                    return tensor_in.type(torch.float16)

            self.transforms.transforms.insert(-1, CustomToFP16())

    def __get_vit_config(self) -> Dict:
        return timm.data.resolve_data_config({}, model=self.model)
    
    def __get_vit_transforms(self) -> tv_transforms.Compose:
        return timm.data.transforms_factory.create_transform(**self.__get_vit_config())

    def load_dataset(self) -> None:
        super().load_dataset()
        if self.precision == configs.FP16:
            self.input_list = [input_tensor.half() for input_tensor in self.input_list]
            self.gt_targets = [gt_target.half() for gt_target in self.gt_targets]

    def clear_gpu_memory_and_reload(self):
        if self.output_logger:
            self.output_logger.info("RELOADING THE MODEL AND THE INPUTS AFTER ERROR")
        del self.input_list
        del self.model
        # Free cuda memory
        torch.cuda.empty_cache()
        self.load_data_at_test()

    def check_dnn_accuracy(self) -> None:
        if self.output_logger:
            self.output_logger.debug(f"Checking DNN accuracy")

        if not self.golden or not self.gt_targets:
            raise ValueError("Golden and Ground truth lists should be populated")

        # Default is classification task
        golden_top_k_labels = common.get_top_k_labels_classification(input_tensor=torch.stack(self.golden),
                                                                     top_k=configs.CLASSIFICATION_CRITICAL_TOP_K, dim=2).squeeze()
        gt_top_k_labels = torch.stack(self.gt_targets)

        gt_count = gt_top_k_labels.numel()
        correct = torch.sum(torch.eq(golden_top_k_labels, gt_top_k_labels))

        if self.output_logger:
            correctness = correct / gt_count
            self.output_logger.debug(f"Correct predicted samples:{correct} - ({correctness * 100:.2f}%)")
            if correctness < self.correctness_threshold:
                raise ValueError(f"Low accuracy {correctness * 100.0}%")

    @staticmethod
    def copy_to_cpu(dnn_output):
        return dnn_output.to(configs.CPU)

    def post_inference_process(self, dnn_output_cpu, batch_id) -> int:
        errors = 0
        if self.generate is False:
            errors = self.compare_inference(output=dnn_output_cpu, batch_id=batch_id)
        else:
            self.golden.append(dnn_output_cpu)
        return errors
