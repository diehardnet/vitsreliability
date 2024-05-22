import sys
# sys.path.append('FasterTransformer/examples/pytorch/swin/SwinTransformerQuantization')
# sys.path.append('FasterTransformer/examples/pytorch/vit/ViT-Quantization')
sys.path.extend([
    'FasterTransformer/examples/pytorch/swin/SwinTransformerQuantization',
    # 'FasterTransformer/examples/pytorch/vit/ViT-Quantization',
])
from FasterTransformer.examples.pytorch.swin.SwinTransformerQuantization.models import build_model
from FasterTransformer.examples.pytorch.swin.SwinTransformerQuantization.SwinTransformer.data.build import build_transform
from FasterTransformer.examples.pytorch.swin.SwinTransformerQuantization.SwinTransformer.config import get_config

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
import hardened_identity


class SetupVits(SetupBaseImageNet):
    def __init__(self, args: argparse.Namespace, output_logger: logging.Logger):
        super().__init__(args=args, output_logger=output_logger)

        self.model = None
        self.golden = list()
        self.input_list = list()
        self.gt_targets = list()

        # # for int8 swin models
        # if args.cfg or self.local_rank:
        #     self.args = args
        # self.correctness_threshold = 0.6  # Based on the whole dataset accuracy. Used only for golden generate part

    @property
    def num_batches(self):
        return len(self.input_list)

    # def __call__(self, batch_id, **kwargs):
    #     return self.model(self.input_list[batch_id], captions=self.input_captions)

    def __load_checkpoint(self, model, checkpoint_path: str) -> None:
        if not os.path.exists(checkpoint_path):
                raise ValueError(f"Path {checkpoint_path} does not exist.")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint, strict=False)

    def __parse_options(self):
        """ Taken from FasterTransformer/examples/pytorch/swin/SwinTransformerQuantization/main.py
        """
        parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
        parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
        parser.add_argument(
            "--opts",
            help="Modify config options by adding 'KEY VALUE' pairs. ",
            default=None,
            nargs='+',
        )

        # easy config modification
        parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
        parser.add_argument('--data-path', type=str, help='path to dataset')
        parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
        parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                            help='no: no cache, '
                                'full: cache all data, '
                                'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
        parser.add_argument('--pretrained',
                            help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
        parser.add_argument('--resume', help='resume from checkpoint')
        parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
        parser.add_argument('--use-checkpoint', action='store_true',
                            help="whether to use gradient checkpointing to save memory")
        parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
        parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                            help='mixed precision opt level, if O0, no amp is used (deprecated!)')
        parser.add_argument('--output', default='output', type=str, metavar='PATH',
                            help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
        parser.add_argument('--tag', help='tag of experiment')
        parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
        parser.add_argument('--throughput', action='store_true', help='Test throughput only')
        parser.add_argument("--engine", type=str, help="The directory of swin tensorrt engine.")

        # Calibration
        parser.add_argument('--calib', action='store_true', help='Perform calibration only')
        parser.add_argument('--train', action='store_true', help='Perform training only')
        parser.add_argument('--int8-mode', type=int, help='int8 mode', choices=[1, 2], default=1)
        parser.add_argument('--num-calib-batch', type=int, default=4, help='Number of batches for calibration. 0 will disable calibration.')
        parser.add_argument('--calib-batchsz', type=int, default=8, help='Batch size when doing calibration')
        parser.add_argument('--calib-output-path', type=str, help='Output directory to save calibrated model')
        # distributed training
        parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel', default=0)

        parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs to run QAT fintuning.")
        parser.add_argument("--qat-lr", type=float, default=5e-7, help="learning rate for QAT.")
        parser.add_argument("--distill", action='store_true', help='Using distillation')
        parser.add_argument("--teacher", type=str, help='teacher model path')
        parser.add_argument('--distillation_loss_scale', type=float, default=10000., help="scale applied to distillation component of loss")

        # for acceleration
        parser.add_argument('--fused_window_process', action='store_true', help='Fused window shift & window partition, similar for reversed part.')

        # quant_utils.add_arguments(parser)
        args, unparsed = parser.parse_known_args()
        # args = quant_utils.set_args(args)
        # quant_utils.set_default_quantizers(args)

        config = get_config(args)

        return args, config

    def load_model(self) -> None:
        if self.precision is not configs.INT8:
            model = timm.create_model(self.model_name, pretrained=True)
            self.transforms = self.__get_vit_transforms(model)
            if self.hardened_model:
                hardened_identity.replace_identity(model, profile_or_inference="inference", model_name=self.model_name)
        
        if self.precision == configs.FP16:
            model = model.half()

        elif self.precision == configs.INT8:
            if self.model_name not in configs.INT8_MODELS:
                raise ValueError(f"Model {self.model_name} is not supported with PTQ4ViT. Supported models are: {configs.INT8_MODELS}.")
            
            args, config = self.__parse_options()
            model = build_model(config)

            cp_name = f"{self.model_name}_calib.pth"
            path = os.path.join(configs.INT8_CKPT_DIR, cp_name)
            self.__load_checkpoint(model, path)
            self.transforms = build_transform(False, config)

        elif self.precision == configs.BFLOAT16:
            raise ValueError("BFLOAT16 is not supported for ViTs")

        model.eval()
        model.to(configs.GPU_DEVICE)
        model.zero_grad(set_to_none=True)
        self.model = model
        if self.precision == configs.FP16:
            class CustomToFP16:
                def __call__(self, tensor_in):
                    return tensor_in.type(torch.float16)

            self.transforms.transforms.insert(-1, CustomToFP16())

    def __get_vit_config(self, model) -> Dict:
        return timm.data.resolve_data_config({}, model=model)
    
    def __get_vit_transforms(self, model) -> tv_transforms.Compose:
        return timm.data.transforms_factory.create_transform(**self.__get_vit_config(model))

    def load_dataset(self) -> None:
        # if self.precision is not configs.INT8:
        super().load_dataset()
        if self.precision == configs.FP16:
            self.input_list = [input_tensor.half() for input_tensor in self.input_list]

            # for input, label in zip(self.input_list, self.gt_targets):
            #     print(input.dtype, label.dtype)
        # else:
        #     if self.transforms is None:
        #         raise ValueError("First you have to set the set of transforms")

        #     if self.output_logger:
        #         self.output_logger.debug("Loading Imagenet dataset, it can take some time!")

        #     # Set a sampler on the CPU
        #     sampler_generator = torch.Generator(device=configs.CPU)
        #     sampler_generator.manual_seed(configs.TORCH_SEED)

        #     test_set = tv_datasets.imagenet.ImageNet(root=configs.IMAGENET_DATASET_DIR, transform=self.transforms,
        #                                             split='val')
        #     subset = torch.utils.data.RandomSampler(data_source=test_set, replacement=False, num_samples=self.test_sample,
        #                                             generator=sampler_generator)
        #     test_loader = torch.utils.data.DataLoader(dataset=test_set, sampler=subset, batch_size=self.batch_size,
        #                                             shuffle=False, pin_memory=True)

        #     # TODO: it is necessary to save which images are being loaded
        #     self.selected_samples = list()
        #     for i, (inputs, labels) in enumerate(test_loader):
        #         # Only the inputs must be in the device
        #         self.input_list.append(inputs.to("cuda:0"))
        #         self.gt_targets.append(labels)

    def clear_gpu_memory_and_reload(self):
        if self.output_logger:
            self.output_logger.info("RELOADING THE MODEL AND THE INPUTS AFTER ERROR")
        del self.input_list
        del self.model
        # Free cuda memory
        torch.cuda.empty_cache()
        self.load_data_at_test()

    def compare_inference(self, output, batch_id) -> int:
        # uncomment to test the error detection
        # if self.current_iteration == 4:
        #     output[0] *= 0
        #     output[3] *= 0

        golden = self.golden[batch_id]
        gt_targets = self.gt_targets[batch_id]
        # Make sure that they are on CPU
        out_is_cuda, golden_is_cuda = output.is_cuda, golden.is_cuda
        if out_is_cuda or golden_is_cuda:
            dnn_log_helper.log_and_crash(
                fatal_string=f"Tensors are not on CPU. OUT IS CUDA:{out_is_cuda} GOLDEN IS CUDA:{golden_is_cuda}")

        # First check if the tensors are equal or not
        if common.equal(lhs=golden, rhs=output, threshold=self.float_threshold) is True:
            return 0

        # ------------ Check the size of the tensors
        if output.shape != golden.shape:
            info_detail = f"shape-diff g:{golden.shape} o:{output.shape}"
            if self.output_logger:
                self.output_logger.error(info_detail)
            dnn_log_helper.log_info_detail(info_detail)

        # FP16 to FP32
        golden, output = golden.float(), output.float()

        output_errors = 0
        # Iterate over the batches
        golden_top_k_labels = common.get_top_k_labels_classification(input_tensor=golden,
                                                                     top_k=configs.CLASSIFICATION_CRITICAL_TOP_K, dim=1)
        output_top_k_labels = common.get_top_k_labels_classification(input_tensor=output,
                                                                     top_k=configs.CLASSIFICATION_CRITICAL_TOP_K, dim=1)
        for img_id, (output_batch, golden_batch, output_top_k, golden_top_k, gt_label) in enumerate(
                zip(output, golden, output_top_k_labels, golden_top_k_labels, gt_targets)):
            # using the same approach as the detection, compare only the positions that differ
            if common.equal(lhs=output_batch, rhs=golden_batch, threshold=self.float_threshold) is False:
                # ------------ Check if there is a Critical error ------------------------------------------------------
                err_string = f"batch:{batch_id} imgid:{img_id}"
                for i, (tpk_found, tpk_gold) in enumerate(zip(output_top_k, golden_top_k)):
                    if tpk_found != tpk_gold:
                        output_errors += 1
                        error_detail_ctr = (f"critical {err_string} "
                                            f"i:{i} "
                                            f"g:{tpk_gold} "
                                            f"o:{tpk_found} "
                                            f"gt:{gt_label}")
                        if self.output_logger:
                            self.output_logger.error(error_detail_ctr)
                        dnn_log_helper.log_error_detail(error_detail_ctr)
                # ------------ Check error on the whole output ---------------------------------------------------------
                # Not necessary to save everything, only the good info
                # Data on output tensor
                has_nan, has_inf, min_val, max_val = common.describe_error(input_tensor=output_batch)
                error_detail_out = f"{err_string} output_t nan:{has_nan} inf:{has_inf} min:{min_val} max:{max_val} "
                # Data on abs differences
                abs_diff = torch.abs(torch.subtract(golden_batch, output_batch))
                has_nan_diff, has_inf_diff, min_val_diff, max_val_diff = common.describe_error(input_tensor=abs_diff)
                error_detail_out += f"diff_t nan:{has_nan_diff} inf:{has_inf_diff} min:{min_val_diff} max:{max_val_diff}"
                output_errors += 1
                if self.output_logger:
                    self.output_logger.error(error_detail_out)
                dnn_log_helper.log_error_detail(error_detail_out)

        return output_errors

    def check_dnn_accuracy(self) -> None:
        if self.output_logger:
            self.output_logger.debug(f"Checking DNN accuracy")

        if not self.golden or not self.gt_targets:
            raise ValueError("Golden and Ground truth lists should be populated")

        # Default is classification task
        golden_top_k_labels = common.get_top_k_labels_classification(input_tensor=torch.stack(self.golden).float(),
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
