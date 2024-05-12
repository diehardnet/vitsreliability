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


class SetupVits(SetupBase):
    def __init__(self, args: argparse.Namespace, output_logger: logging.Logger):
        super().__init__(args=args, output_logger=output_logger)

        self.model = None
        self.golden = list()
        self.input_list = list()
        self.gt_targets = list()
        self.dateset = None
        self.timm = args.timm # load a timm model or not
        # self.correctness_threshold = 0.6  # Based on the whole dataset accuracy. Used only for golden generate part

    @property
    def num_batches(self):
        return len(self.input_list)

    def __call__(self, batch_id, **kwargs):
        return self.model(self.input_list[batch_id], captions=self.input_captions)

    def load_model(self) -> None:
        if self.timm is True:
            model = timm.create_model(self.model_name, pretrained=True)
        else:
            model = torch.load(self.model_checkpoint_path)
        model.eval()
        model.to(configs.GPU_DEVICE)
        self.model = model

    def __get_vit_config(self) -> Dict:
        return timm.data.resolve_data_config({}, model=self.model)
    
    def __get_vit_transforms(self) -> tv_transforms.Compose:
        return timm.data.transforms_factory.create_transform(**self.__get_vit_config())

    def load_dataset(self) -> None:
        transforms = self.__get_vit_transforms()
        dataset = tv_datasets.ImageNet(root=configs.DATA_PATH, split="val", transform=transforms)
        self.dateset = dataset

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        for i, (images, labels) in enumerate(dataloader):
            self.input_list.append(images)
            self.gt_targets.append(labels)
            if i == self.test_sample:
                break

    def check_dnn_accuracy(self) -> None:
        # TODO: Implement this
        pass

    def compare_inference(self, output, batch_id) -> int:
        """ Compare the output with the golden values

        Args:
            output (Tensor): The output of the inference.
            batch_id (int): The ID of the batch.

        Returns:
            int: The result of the comparison. This can be used to determine the reliability of the inference.
        """

        golden = self.golden[batch_id]

        # ensure the output and golden is on the CPU
        golden = golden.to(configs.CPU_DEVICE)
        if output.is_cuda:
            dnn_log_helper.log_and_crash(
                        fatal_string=f"Tensor {output} not on CPU:{output.is_cuda}")

        # Compare the output with the golden
        if common.equal(lhs=output, rhs=golden, threshold=self.float_threshold) is True:
            # exiting now because the output is correct
            return 0

        return common.count_errors(lhs=output, rhs=golden, threshold=self.float_threshold)

    def load_data_at_test(self):
        # This will save time
        if self.generate is False:
            # Save everything in the same list
            [self.golden, self.input_list, self.gt_targets,
             self.model, self.dataset] = torch.load(self.gold_path)
        else:
            # The First step is to load the inputs in the memory
            # Load the model
            self.load_model()
            self.load_dataset()
            self.golden = list()

    def save_setup_data_to_gold_file(self):
        torch.save(obj=[
            self.dataset,
            self.model,
            self.input_list,
            self.gt_targets,
            self.golden
        ], f=os.path.absolute(self.gold_path))

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
        return dnn_output.to(configs.CPU_DEVICE)

    def post_inference_process(self, dnn_output_cpu, batch_id) -> int:
        errors = 0
        if self.generate is False:
            errors = self.compare_inference(output=dnn_output_cpu, batch_id=batch_id)
        else:
            self.golden.append(dnn_output_cpu)
        return errors
