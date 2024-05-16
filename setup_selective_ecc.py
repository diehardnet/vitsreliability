import argparse
import logging

import torch

import configs
from setup_base import SetupBaseCIFAR


from other_nets.resnet import resnet18


class SetupSelectiveECC(SetupBaseCIFAR):

    def __init__(self, args: argparse.Namespace, output_logger: logging.Logger):
        super().__init__(args=args, output_logger=output_logger)

    def load_model(self) -> None:
        self.model = resnet18(pretrained=False)
        self.model.load_state_dict(torch.load(self.model_checkpoint_path))
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.model = self.model.to(configs.GPU_DEVICE)
