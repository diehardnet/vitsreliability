import argparse
import logging
import os
import pathlib
import re
import shutil

import numpy
import torch
from PIL import Image

import GroundingDINO.groundingdino.datasets.transforms as gdino_transforms
import common
import configs
import dnn_log_helper
import hardened_identity
from GroundingDINO.demo.test_ap_on_coco import CocoDetection as GDINOCocoDetection
from GroundingDINO.demo.test_ap_on_coco import PostProcessCocoGrounding as GDINOPostProcessCocoGrounding
from GroundingDINO.groundingdino.datasets.cocogrounding_eval import (
    CocoGroundingEvaluator as GDINOCocoGroundingEvaluator
)
from GroundingDINO.groundingdino.models import build_model as gdino_build_model
from GroundingDINO.groundingdino.util import get_tokenlizer as gdino_get_tokenlizer
from GroundingDINO.groundingdino.util.misc import collate_fn as gdino_collate_fn
from GroundingDINO.groundingdino.util.slconfig import SLConfig as gdino_SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict as gdino_clean_state_dict
from setup_base import SetupBase


class SetupGroundingDINO(SetupBase):

    def __init__(self, args: argparse.Namespace, output_logger: logging.Logger):
        super().__init__(args=args, output_logger=output_logger)
        # threshold (10%) to be considered critical, mAP * self.coco_metrics_threshold <= original inference
        self.coco_metrics_threshold = 0.9
        self.text_encoder_type = ""
        self.coco_api = None
        self.correctness_threshold = 0.6  # Based on the whole dataset accuracy. Used only for golden generate part
        self.map_list = list()
        self.nan_inf_list = list()
        if configs.IMAGENET == args.dataset:  # default one
            self.dataset = configs.COCO

        if args.batchsize != 1:
            raise NotImplementedError("For now, the only batch size allowed is 1")

        if self.precision != configs.FP32:
            raise NotImplementedError("Precisions different than FP32 are not available for now")

        self.logits_path = os.path.join(pathlib.Path.home(), "grounding_dino_logits")
        self.save_logits_max_disk_usage = 0.9
        if self.save_logits:
            if self.output_logger:
                self.output_logger.debug(f"Save logits enabled, creating {self.logits_path}")
            pathlib.Path(self.logits_path).mkdir(parents=True, exist_ok=True)

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
        self.text_encoder_type = cfg_args.text_encoder_type

    def load_dataset(self) -> None:
        # COCO default transformations
        transform = gdino_transforms.Compose(
            [
                gdino_transforms.RandomResize([800], max_size=1333),
                gdino_transforms.ToTensor(),
                gdino_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        # build dataloader
        if self.dataset == configs.COCO:
            dataset = GDINOCocoDetection(configs.COCO_DATASET_VAL, configs.COCO_DATASET_ANNOTATIONS,
                                         transforms=transform)
            self.selected_samples = list(range(self.test_sample))
            subset = torch.utils.data.SequentialSampler(self.selected_samples)
            test_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=1,
                                                      collate_fn=gdino_collate_fn, sampler=subset)
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
        elif self.dataset == configs.CUSTOM_DATASET:
            with open(self.imgs_file_path, 'r') as f:
                input_paths = f.readlines()
            for i, input_image_path in enumerate(input_paths):
                image_pil = Image.open(input_image_path).convert("RGB")  # load image
                image, _ = transform(image_pil, None)  # 3, h, w
                self.input_list.append(image.to(configs.GPU_DEVICE))
        else:
            raise NotImplementedError("Dataset not implemented yet")

    def get_coco_evaluator(self, predicted: list, targets: list) -> GDINOCocoGroundingEvaluator:
        # build post-processor
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

    def get_iou_for_single_image_at_test(self, output: dict, batch_id) -> numpy.ndarray:
        evaluator = self.get_coco_evaluator(predicted=[output], targets=[self.gt_targets[batch_id]])
        bbox_stats = evaluator.coco_eval["bbox"].stats
        if bbox_stats.size != 12:
            raise ValueError(f"Incorrect size of stats array: {bbox_stats.size()}")
        return bbox_stats

    def check_dnn_accuracy(self) -> None:
        evaluator = self.get_coco_evaluator(predicted=self.golden, targets=self.gt_targets)

        if self.output_logger:
            stats_list = evaluator.coco_eval["bbox"].stats.tolist()
            correctness = sum(stats_list) / len(stats_list)
            self.output_logger.debug(f'Final results:{correctness * 100.0:.2f}%')
            if correctness < self.correctness_threshold:
                dnn_log_helper.log_and_crash(fatal_string=f"ACCURACY: {correctness * 100.0}%")

    def _save_logits(self, output, batch_id):
        total, used, free = shutil.disk_usage("/")
        used_percent = used / total
        if used_percent > self.save_logits_max_disk_usage:
            dnn_log_helper.log_info_detail(info_detail=f"disk usage: {used_percent:.3f}, not saving the logits")
        elif self.save_logits:
            log_helper_file = re.match(r".*LOCAL:(\S+).log.*", dnn_log_helper.log_file_name).group(1)
            save_file = f"{os.path.basename(log_helper_file)}_btid_{batch_id}_it_{self.current_iteration}.pt"
            save_file = os.path.join(self.logits_path, save_file)
            if self.output_logger:
                self.output_logger.debug(f"Saving logits at:{save_file}")
            dnn_log_helper.log_info_detail(info_detail=f"LOGITS_AT:{save_file}")
            torch.save(output, save_file)

    @staticmethod
    def __return_tensor_nans_and_infs(output_tensor) -> tuple:
        return (
            torch.sum(torch.isnan(output_tensor)),
            torch.sum(torch.isinf(output_tensor))
        )

    def compare_inference(self, output: dict, batch_id) -> int:
        # if self.current_iteration == 2:
        #     output["pred_logits"][0, 4] *= 0
        # self.nan_inf_list[batch_id][1] = (self.nan_inf_list[batch_id][1][0] + 34, self.nan_inf_list[batch_id][1][1])

        golden_dict = self.golden[batch_id]
        golden_metrics = self.map_list[batch_id]
        golden_nan_inf_metrics = self.nan_inf_list[batch_id]

        # Make sure that they are on CPU
        for out_or_gold, dict_data in [("out", output), ("gold", golden_dict)]:
            for tensor_type, tensor in dict_data.items():
                if tensor.is_cuda:
                    dnn_log_helper.log_and_crash(
                        fatal_string=f"Tensor {out_or_gold}-{tensor_type} not on CPU:{tensor.is_cuda}")

        # First check if the tensors are equal or not
        for i, (output_tensor, golden_tensor) in enumerate(zip(output.values(), golden_dict.values())):
            # print("\n\nTORCHIS INF:", torch.sum(torch.isinf(output_tensor)), torch.sum(torch.isinf(golden_tensor)))
            output_nan_inf_metrics_i = self.__return_tensor_nans_and_infs(output_tensor=output_tensor)
            golden_nan_inf_metrics_i = golden_nan_inf_metrics[i]
            if any([out_naninf != golden_naninf for out_naninf, golden_naninf in
                    zip(output_nan_inf_metrics_i, golden_nan_inf_metrics_i)]):
                err_message = f"NaNs/Infs differ:{output_nan_inf_metrics_i} x {golden_nan_inf_metrics_i}"
                dnn_log_helper.log_error_detail(err_message)
                if self.output_logger:
                    self.output_logger.error(err_message)
                break
            # Gambiarra to avoid comparison with Infs
            largest_value = 1e35
            out_tensor_filtered = torch.nan_to_num(output_tensor, nan=0.0, posinf=largest_value, neginf=-largest_value)
            gld_tensor_filtered = torch.nan_to_num(golden_tensor, nan=0.0, posinf=largest_value, neginf=-largest_value)
            if common.equal(lhs=out_tensor_filtered, rhs=gld_tensor_filtered, threshold=self.float_threshold) is False:
                # no need to continue, we save time
                break
        else:
            return 0

        # ------------ Check if there is a Critical error ----------------------------------------------------------
        critical_or_not = "tolerable"
        precision_str = recall_str = "1"
        if self.dataset == configs.COCO:
            current_map = self.get_iou_for_single_image_at_test(output=output, batch_id=batch_id)
            # The original map/mar coco, 0.5:0.95
            map_curr, mar_curr = current_map[0], current_map[6]
            map_gold = golden_metrics[0] * self.coco_metrics_threshold
            mar_gold = golden_metrics[6] * self.coco_metrics_threshold
            if map_curr <= map_gold or mar_curr <= mar_gold:
                critical_or_not = "critical"

            precision_str = ";".join(f"{v:.4e}" for v in current_map[:6])
            recall_str = ";".join(f"{v:.4e}" for v in current_map[6:])

        # if critical_or_not in ["undefined", "critical"]:
        self._save_logits(output=output, batch_id=batch_id)

        err_string = f"{critical_or_not} batch:{batch_id} gd_pre:{precision_str} gd_rec:{recall_str}"
        if self.output_logger:
            self.output_logger.error(err_string)
        dnn_log_helper.log_error_detail(err_string)
        # # ------------ Check error on the whole output -------------------------------------------------------------
        # Not necessary to save everything, only the good info
        output_errors = 1
        for (out_tensor_type, output_tensor), (gld_tensor_type, golden_tensor) in zip(output.items(),
                                                                                      golden_dict.items()):
            output_errors += 1  # put only 1 per item, if you increment too much, it crashes because of too many errors

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
            [self.golden, self.input_list, self.gt_targets, self.model, self.input_captions, self.text_encoder_type,
             self.coco_api, self.map_list, self.nan_inf_list] = torch.load(self.gold_path)
        else:
            # The First step is to load the inputs in the memory
            # Load the model
            self.load_model()
            self.load_dataset()
            self.golden = list()

    def save_setup_data_to_gold_file(self):
        torch.save(
            obj=[self.golden, self.input_list, self.gt_targets,
                 self.model, self.input_captions, self.text_encoder_type, self.coco_api, self.map_list,
                 self.nan_inf_list],
            f=self.gold_path
        )

    @staticmethod
    def copy_to_cpu(dnn_output):
        return {k: v.to(configs.CPU) for k, v in dnn_output.items()}

    def post_inference_process(self, dnn_output_cpu, batch_id) -> int:
        errors = 0
        if self.generate is False:
            errors = self.compare_inference(output=dnn_output_cpu, batch_id=batch_id)
        else:
            self.golden.append(dnn_output_cpu)
            self.map_list.append(self.get_iou_for_single_image_at_test(output=dnn_output_cpu, batch_id=batch_id))
            new_bathes_nan_inf_counters = list()
            for tensor_i in dnn_output_cpu.values():
                new_bathes_nan_inf_counters.append(self.__return_tensor_nans_and_infs(output_tensor=tensor_i))
                # print(new_bathes_nan_inf_counters[-1])
            self.nan_inf_list.append(new_bathes_nan_inf_counters)
        return errors
