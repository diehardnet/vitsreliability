import argparse
import logging

import torch

import configs
from setup_base import SetupBaseClassification, SetupBaseCIFAR, SetupBaseImageNet
from other_nets.resnet import resnet18


class SetupSelectiveECC(SetupBaseClassification):

    def __init__(self, args: argparse.Namespace, output_logger: logging.Logger):
        super().__init__(args=args, output_logger=output_logger)
        # assign an instance of A or B depending on whether argument use_A is True
        if self.dataset in [configs.CIFAR10, configs.CIFAR100]:
            self.instance = SetupBaseCIFAR(args=args, output_logger=output_logger)
        else:
            self.instance = SetupBaseImageNet(args=args, output_logger=output_logger)

    # called when an attribute is not found:
    def __getattr__(self, name):
        # assume it is implemented by self.instance
        return self.instance.__getattribute__(name)

    def load_model(self) -> None:
        self.model = resnet18(pretrained=False)
        self.model.load_state_dict(torch.load(self.model_checkpoint_path))
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.model = self.model.to(configs.GPU_DEVICE)
