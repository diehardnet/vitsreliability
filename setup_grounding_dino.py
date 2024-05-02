import argparse
import logging
from typing import Union

import torch

from GroundingDINO.groundingdino.util.slconfig import SLConfig as gdino_SLConfig
from GroundingDINO.groundingdino.models import build_model as gdino_build_model
from GroundingDINO.groundingdino.util.utils import clean_state_dict as gdino_clean_state_dict
import GroundingDINO.groundingdino.datasets.transforms as gdino_transforms
from GroundingDINO.groundingdino.util.misc import collate_fn as gdino_collate_fn
from GroundingDINO.groundingdino.util import get_tokenlizer as gdino_get_tokenlizer

from GroundingDINO.demo.test_ap_on_coco import CocoDetection as GDINOCocoDetection
from GroundingDINO.demo.test_ap_on_coco import PostProcessCocoGrounding as GDINOPostProcessCocoGrounding
from GroundingDINO.groundingdino.datasets.cocogrounding_eval import (
    CocoGroundingEvaluator as GDINOCocoGroundingEvaluator
)

import configs
import dnn_log_helper
import common
import hardened_identity
from setup_base import SetupBase


class SetupGroundingDINO(SetupBase):
    def __init__(self, args: argparse.Namespace, output_logger: logging.Logger):
        super().__init__(args=args, output_logger=output_logger)
        self.text_encoder_type = ""
        self.model = None
        self.golden = list()
        self.input_list = list()
        self.gt_targets = list()
        self.coco_api = None
        self.correctness_threshold = 0.6  # Based on the whole dataset accuracy. Used only for golden generate part

        if args.batchsize != 1:
            raise NotImplementedError("For now, the only batch size allowed is 1")

        if self.precision != configs.FP32:
            raise NotImplementedError("Precisions different than FP32 are not available for now")

    @property
    def num_batches(self):
        return len(self.input_list)

    def __call__(self, batch_id, **kwargs):
        return self.model(self.input_list[batch_id], captions=self.input_captions)

    def load_model(self) -> None:
        # The First option is the baseline option
        cfg_args = gdino_SLConfig.fromfile(self.model_config_path)
        cfg_args.device = configs.GPU_DEVICE
        self.model = gdino_build_model(cfg_args)
        checkpoint = torch.load(self.model_checkpoint_path, map_location=configs.GPU_DEVICE)
        self.model.load_state_dict(gdino_clean_state_dict(checkpoint["model"]), strict=False)
        self.model.eval()
        # Disable also parameter grads
        self.model.zero_grad(set_to_none=True)
        self.model = self.model.to(configs.GPU_DEVICE)

        if self.hardened_model:
            hardened_identity.replace_identity(module=self.model, profile_or_inference="inference",
                                               model_name=self.model_name)
        # TODO: Implement when the serialization is possible
        if self.torch_compile is True:
            # model = torch.compile(model=model, mode="max-autotune")
            dnn_log_helper.log_and_crash(
                fatal_string="Up to now it's not possible to serialize compiled models "
                             "(github.com/pytorch/pytorch/issues/101107#issuecomment-1542688089)"
            )
        self.text_encoder_type = cfg_args.text_encoder_type

    def load_dataset(self) -> None:
        # build dataloader
        transform = gdino_transforms.Compose(
            [
                gdino_transforms.RandomResize([800], max_size=1333),
                gdino_transforms.ToTensor(),
                gdino_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        dataset = GDINOCocoDetection(configs.COCO_DATASET_VAL, configs.COCO_DATASET_ANNOTATIONS, transforms=transform)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=1,
                                                  collate_fn=gdino_collate_fn)
        self.coco_api = dataset.coco

        # build captions
        caption = self.input_captions
        if self.input_captions == '':
            category_dict = dataset.coco.dataset['categories']
            cat_list = [item['name'] for item in category_dict]
            caption = " . ".join(cat_list) + ' .'

        # For each image in the batch I'm searching for the COCO classes
        self.input_captions = [caption] * self.batch_size
        if self.output_logger:
            self.output_logger.debug(f"Input text prompt:{caption}")

        self.input_list, self.gt_targets = list(), list()
        for i, (inputs, targets) in enumerate(test_loader):
            # Only the inputs must be in the device
            self.input_list.append(inputs.to(configs.GPU_DEVICE))
            self.gt_targets.append(targets)
            if i == self.test_sample:
                break

    def get_coco_evaluator(self, predicted: list, targets: list) -> GDINOCocoGroundingEvaluator:
        # build post processor
        tokenlizer = gdino_get_tokenlizer.get_tokenlizer(self.text_encoder_type)
        postprocessor = GDINOPostProcessCocoGrounding(coco_api=self.coco_api, tokenlizer=tokenlizer)
        # build evaluator
        evaluator = GDINOCocoGroundingEvaluator(self.coco_api, iou_types=("bbox",), useCats=True)
        for outputs, targets in zip(predicted, targets):
            original_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessor(outputs, original_target_sizes)
            coco_grounding_result = {target["image_id"]: output for target, output in zip(targets, results)}
            evaluator.update(coco_grounding_result)
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()
        return evaluator

    def get_iou_for_single_image_at_test(self, output: dict, batch_id) -> [list, list]:
        evaluator = self.get_coco_evaluator(predicted=[output], targets=[self.gt_targets[batch_id]])
        bbox_stats = evaluator.coco_eval["bbox"].stats
        if bbox_stats.size != 12:
            raise ValueError(f"Incorrect size of stats array: {bbox_stats.size()}")
        precision, recall = bbox_stats[:6].tolist(), bbox_stats[6:].tolist()
        return precision, recall

    def check_dnn_accuracy(self) -> None:
        evaluator = self.get_coco_evaluator(predicted=self.golden, targets=self.gt_targets)

        if self.output_logger:
            stats_list = evaluator.coco_eval["bbox"].stats.tolist()
            correctness = sum(stats_list) / len(stats_list)
            self.output_logger.debug(f'Final results:{correctness * 100.0:.2f}%')
            if correctness < self.correctness_threshold:
                dnn_log_helper.log_and_crash(fatal_string=f"ACCURACY: {correctness * 100.0}%")

    def compare_inference(self, output: dict, batch_id) -> int:
        # global TEST
        # TEST += 1
        # if TEST == 3:
        #     output["pred_logits"][0, 3] = 39304
        golden_dict = self.golden[batch_id]
        # Make sure that they are on CPU
        for out_or_gold, dict_data in [("out", output), ("gold", golden_dict)]:
            for tensor_type, tensor in dict_data.items():
                if tensor.is_cuda:
                    dnn_log_helper.log_and_crash(
                        fatal_string=f"Tensor {out_or_gold}-{tensor_type} not on CPU:{tensor.is_cuda}")

        # First check if the tensors are equal or not
        for output_tensor, golden_tensor in zip(output.values(), golden_dict.values()):
            out_tensor_filtered = torch.nan_to_num(output_tensor, nan=0.0, posinf=0, neginf=0)
            gld_tensor_filtered = torch.nan_to_num(golden_tensor, nan=0.0, posinf=0, neginf=0)
            if common.equal(lhs=out_tensor_filtered, rhs=gld_tensor_filtered, threshold=self.float_threshold) is False:
                # no need to continue, we save time
                break
        else:
            return 0

        output_errors = 1
        # ------------ Check if there is a Critical error ----------------------------------------------------------
        precision, recall = self.get_iou_for_single_image_at_test(output=output, batch_id=batch_id)
        precision = ";".join(f"{v:.6e}" for v in precision)
        recall = ";".join(f"{v:.6e}" for v in recall)
        err_string = f"batch:{batch_id} gd_pre:{precision} gd_rec:{recall}"

        if self.output_logger:
            self.output_logger.error(err_string)
        dnn_log_helper.log_error_detail(err_string)
        # # ------------ Check error on the whole output -------------------------------------------------------------
        # Not necessary to save everything, only the good info
        for (out_tensor_type, output_tensor), (gld_tensor_type, golden_tensor) in zip(output.items(),
                                                                                      golden_dict.items()):
            # Data on output tensor
            has_nan, has_inf, min_val, max_val = common.describe_error(input_tensor=output_tensor)
            error_detail_out = f"{out_tensor_type} output_t nan:{has_nan} inf:{has_inf} min:{min_val} max:{max_val} "
            # Data on abs differences
            abs_diff = torch.abs(torch.subtract(output_tensor, golden_tensor))
            has_nan_diff, has_inf_diff, min_val_diff, max_val_diff = common.describe_error(input_tensor=abs_diff)
            error_detail_out += f"diff_t nan:{has_nan_diff} inf:{has_inf_diff} min:{min_val_diff} max:{max_val_diff}"

            if self.output_logger:
                self.output_logger.error(error_detail_out)
            dnn_log_helper.log_error_detail(error_detail_out)

        # ------------ log and return
        if output_errors != 0:
            dnn_log_helper.log_error_count(error_count=output_errors)
        return output_errors

    def load_data_at_test(self):
        # This will save time
        if self.generate is False:
            # Save everything in the same list
            [self.golden, self.input_list, self.gt_targets,
             self.model, self.input_captions, self.text_encoder_type, self.coco_api] = torch.load(self.gold_path)
        else:
            # The First step is to load the inputs in the memory
            # Load the model
            self.load_model()
            self.load_dataset()
            self.golden = list()

    def save_setup_data_to_gold_file(self):
        torch.save(
            obj=[self.golden, self.input_list, self.gt_targets,
                 self.model, self.input_captions, self.text_encoder_type, self.coco_api],
            f=self.gold_path
        )

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
        return {k: v.to(configs.CPU) for k, v in dnn_output.items()}

    def post_inference_process(self, dnn_output_cpu, batch_id) -> int:
        errors = 0
        if self.generate is False:
            errors = self.compare_inference(output=dnn_output_cpu, batch_id=batch_id)
        else:
            self.golden.append(dnn_output_cpu)
        return errors
