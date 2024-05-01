import argparse
import logging
import os
from typing import Tuple, List, Union

import torch
import torchvision.datasets.coco

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
import console_logger
import dnn_log_helper
import common
import hardened_identity


def load_model(model_checkpoint_path: str, model_config_path: str, hardened_model: bool,
               torch_compile: bool, precision: str, model_name: str) -> [str, torch.nn.Module]:
    # The First option is the baseline option
    cfg_args = gdino_SLConfig.fromfile(model_config_path)
    cfg_args.device = configs.GPU_DEVICE
    model = gdino_build_model(cfg_args)
    checkpoint = torch.load(model_checkpoint_path, map_location=configs.GPU_DEVICE)
    model.load_state_dict(gdino_clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    # Disable also parameter grads
    model.zero_grad(set_to_none=True)
    model = model.to(configs.GPU_DEVICE)

    if hardened_model:
        hardened_identity.replace_identity(module=model, profile_or_inference="inference", model_name=model_name)
    if precision == configs.FP16:
        model = model.half()

    elif precision != configs.FP32:
        raise NotImplementedError("Only supports FP32 and FP16")

    # TODO: Implement when the serialization is possible
    if torch_compile is True:
        # model = torch.compile(model=model, mode="max-autotune")
        dnn_log_helper.log_and_crash(fatal_string="Up to now it's not possible to serialize compiled models "
                                                  "(github.com/pytorch/pytorch/issues/101107#issuecomment-1542688089)")
    return cfg_args.text_encoder_type, model


def load_dataset(
        batch_size: int, test_sample: int, precision: str, output_logger: logging.Logger
) -> Tuple[List, List, List, torchvision.datasets.coco.CocoDetection]:
    # build dataloader
    class CustomToFP16(object):
        def __call__(self, tensor_in, target):
            return tensor_in.type(torch.float16), target

    mixed_precision_transforms = list()
    if precision == configs.FP16:
        mixed_precision_transforms = [CustomToFP16()]

    transform = gdino_transforms.Compose(
        [
            gdino_transforms.RandomResize([800], max_size=1333),
            gdino_transforms.ToTensor(),
            gdino_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ] + mixed_precision_transforms
    )
    dataset = GDINOCocoDetection(configs.COCO_DATASET_VAL, configs.COCO_DATASET_ANNOTATIONS, transforms=transform)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1,
                                              collate_fn=gdino_collate_fn)
    coco_api = dataset.coco

    # build captions
    category_dict = dataset.coco.dataset['categories']
    cat_list = [item['name'] for item in category_dict]
    caption = " . ".join(cat_list) + ' .'
    # TODO: try a different prompt
    if output_logger:
        output_logger.debug(f"Input text prompt:{caption}")
    # For each image in the batch I'm searching for the COCO classes
    input_captions = [caption] * batch_size

    input_dataset, gt_targets = list(), list()
    for i, (inputs, targets) in enumerate(test_loader):
        # Only the inputs must be in the device
        input_dataset.append(inputs.to(configs.GPU_DEVICE))
        gt_targets.append(targets)
        if i == test_sample:
            break

    return input_dataset, gt_targets, input_captions, coco_api


def get_iou_for_single_image_at_test(
        corrupted_output: dict, original_targets: list, text_encoder_type: str,
        coco_api: torchvision.datasets.coco.CocoDetection,
        output_logger: logging.Logger
) -> [List, List]:
    # build post processor
    tokenlizer = gdino_get_tokenlizer.get_tokenlizer(text_encoder_type)
    postprocessor = GDINOPostProcessCocoGrounding(coco_api=coco_api, tokenlizer=tokenlizer)
    # build evaluator
    evaluator = GDINOCocoGroundingEvaluator(coco_api, iou_types=("bbox",), useCats=True)
    original_target_sizes = torch.stack([t["orig_size"] for t in original_targets], dim=0)

    # We have to compute both corrupted and golden
    corrupted_results = postprocessor(corrupted_output, original_target_sizes)
    golden_vs_out_result = {target["image_id"]: output for target, output in zip(original_targets, corrupted_results)}

    evaluator.update(golden_vs_out_result)
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    bbox_stats = evaluator.coco_eval["bbox"].stats
    if bbox_stats.size != 12:
        raise ValueError(f"Incorrect size of stats array: {bbox_stats.size()}")
    precision, recall = bbox_stats[:6].tolist(), bbox_stats[6:].tolist()

    if output_logger:
        output_logger.info(f"Precision:{precision} Recall:{recall}")

    return precision, recall


def compare_detection(
        output_dict: dict,
        golden_dict: dict,
        gt_targets: List,
        batch_id: int,
        output_logger: logging.Logger, float_threshold: float,
        text_encoder_type: str,
        coco_api: torchvision.datasets.coco.CocoDetection
) -> int:
    # global TEST
    # TEST += 1
    # if TEST == 3:
    #     output_dict["pred_logits"][0, 3] = 39304

    # Make sure that they are on CPU
    for out_or_gold, dict_data in [("out", output_dict), ("gold", golden_dict)]:
        for tensor_type, tensor in dict_data.items():
            if tensor.is_cuda:
                dnn_log_helper.log_and_crash(
                    fatal_string=f"Tensor {out_or_gold}-{tensor_type} not on CPU:{tensor.is_cuda}")

    # First check if the tensors are equal or not
    for output_tensor, golden_tensor in zip(output_dict.values(), golden_dict.values()):
        out_tensor_filtered = torch.nan_to_num(output_tensor, nan=0.0, posinf=0, neginf=0)
        gld_tensor_filtered = torch.nan_to_num(golden_tensor, nan=0.0, posinf=0, neginf=0)
        if common.equal(lhs=out_tensor_filtered, rhs=gld_tensor_filtered, threshold=float_threshold) is False:
            # no need to continue, we save time
            break
    else:
        return 0

    output_errors = 1
    # ------------ Check if there is a Critical error ----------------------------------------------------------
    # FIXME: for multiple batch sizes
    # gld_precision, gld_recall = get_iou_for_single_image(predicted=output_dict, targets=[golden_dict],
    #                                                      text_encoder_type=text_encoder_type,
    #                                                      coco_api=coco_api, output_logger=output_logger)
    precision, recall = get_iou_for_single_image_at_test(corrupted_output=output_dict,
                                                         original_targets=gt_targets,
                                                         text_encoder_type=text_encoder_type,
                                                         coco_api=coco_api, output_logger=output_logger)
    err_string = f"batch:{batch_id} gd_pre:{precision} gd_rec:{recall}"

    if output_logger:
        output_logger.error(err_string)
    dnn_log_helper.log_error_detail(err_string)
    # # ------------ Check error on the whole output -------------------------------------------------------------
    # Not necessary to save everything, only the good info
    for (out_tensor_type, output_tensor), (gld_tensor_type, golden_tensor) in zip(output_dict.items(),
                                                                                  golden_dict.items()):
        # Data on output tensor
        has_nan, has_inf, min_val, max_val = common.describe_error(input_tensor=output_tensor)
        error_detail_out = f"{out_tensor_type} output_t nan:{has_nan} inf:{has_inf} min:{min_val} max:{max_val} "
        # Data on abs differences
        abs_diff = torch.abs(torch.subtract(output_tensor, golden_tensor))
        has_nan_diff, has_inf_diff, min_val_diff, max_val_diff = common.describe_error(input_tensor=abs_diff)
        error_detail_out += f"diff_t nan:{has_nan_diff} inf:{has_inf_diff} min:{min_val_diff} max:{max_val_diff}"

        if output_logger:
            output_logger.error(error_detail_out)
        dnn_log_helper.log_error_detail(error_detail_out)

    # ------------ log and return
    if output_errors != 0:
        dnn_log_helper.log_error_count(error_count=output_errors)
    return output_errors


def print_setup_iteration(batch_id: Union[int, None], comparison_time: float, copy_to_cpu_time: float, errors: int,
                          kernel_time: float, setup_iteration: int, terminal_logger: logging.Logger) -> None:
    if terminal_logger:
        wasted_time = comparison_time + copy_to_cpu_time
        time_pct = (wasted_time / (wasted_time + kernel_time)) * 100.0
        iteration_out = f"It:{setup_iteration:<3} batch_id:{batch_id:<3} inference time:{kernel_time:.5f}, "
        iteration_out += f"compare time:{comparison_time:.5f} copy time:{copy_to_cpu_time:.5f} "
        iteration_out += f"(wasted:{time_pct:.1f}%) errors:{errors}"
        terminal_logger.debug(iteration_out)


def check_dnn_accuracy(predicted: List, ground_truth: List, output_logger: logging.Logger,
                       coco_api: torchvision.datasets.coco.CocoDetection, text_encoder_type: str) -> None:
    # build post processor
    tokenlizer = gdino_get_tokenlizer.get_tokenlizer(text_encoder_type)
    postprocessor = GDINOPostProcessCocoGrounding(coco_api=coco_api, tokenlizer=tokenlizer)
    # build evaluator
    evaluator = GDINOCocoGroundingEvaluator(coco_api, iou_types=("bbox",), useCats=True)
    for outputs, targets in zip(predicted, ground_truth):
        original_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessor(outputs, original_target_sizes)
        coco_grounding_result = {target["image_id"]: output for target, output in zip(targets, results)}
        evaluator.update(coco_grounding_result)
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    if output_logger:
        stats_list = evaluator.coco_eval["bbox"].stats.tolist()
        correctness = sum(stats_list) / len(stats_list)
        output_logger.debug(f'Final results:{correctness * 100.0:.2f}%')
        correctness_threshold = 0.6  # Base on the whole dataset accuracy
        if correctness < correctness_threshold:
            dnn_log_helper.log_and_crash(fatal_string=f"ACCURACY: {correctness * 100.0}%")


# Force no grad
@torch.no_grad()
def run_setup_grounding_dino(args: argparse.Namespace):
    # Define DNN goal
    dnn_goal = configs.DNN_GOAL[args.model]
    dataset = configs.DATASETS[dnn_goal]
    float_threshold = configs.DNN_THRESHOLD[dnn_goal]
    args_dict = vars(args)
    log_args = dict(
        framework_name="PyTorch", torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name(), activate_logging=not args.generate, dnn_goal=dnn_goal, dataset=dataset,
        float_threshold=float_threshold, **args_dict
    )
    dnn_log_helper.start_setup_log_file(**log_args)
    if args.batchsize != 1:
        raise NotImplementedError("For now, the only batch size allowed is 1")

    # Check if a device is ok and disable grad
    common.check_and_setup_gpu()

    # Defining a timer
    timer = common.Timer()
    # Terminal console
    main_logger_name = str(os.path.basename(__file__)).replace(".py", "")
    terminal_logger = console_logger.ColoredLogger(main_logger_name) if args.disableconsolelog is False else None

    # Load if it is not a gold generating op
    timer.tic()
    # This will save time
    if args.generate is False:
        # Save everything in the same list
        [golden, input_list, gt_targets, model, input_captions, text_encoder_type, coco_api] = torch.load(args.goldpath)
    else:
        # The First step is to load the inputs in the memory
        # Load the model
        text_encoder_type, model = load_model(
            model_checkpoint_path=args.checkpointpath, model_config_path=args.configpath,
            hardened_model=args.hardenedid, torch_compile=args.usetorchcompile, precision=args.precision,
            model_name=args.model
        )
        input_list, gt_targets, input_captions, coco_api = load_dataset(batch_size=args.batchsize,
                                                                        test_sample=args.testsamples,
                                                                        precision=args.precision,
                                                                        output_logger=terminal_logger)
        golden = list()

    timer.toc()
    golden_load_diff_time = timer.diff_time_str

    if terminal_logger:
        terminal_logger.debug("\n".join(f"{k}:{v}" for k, v in args_dict.items()))
        terminal_logger.debug(f"Time necessary to load the golden outputs, model, and inputs: {golden_load_diff_time}")

    # Main setup loop
    setup_iteration = 0
    while setup_iteration < args.iterations:
        # Loop over the input list
        batch_id = 0  # It must be like this, because I may reload the list in the middle of the process
        while batch_id < len(input_list):
            timer.tic()
            dnn_log_helper.start_iteration()
            dnn_output = model(input_list[batch_id], captions=input_captions)
            torch.cuda.synchronize(device=configs.GPU_DEVICE)
            dnn_log_helper.end_iteration()
            timer.toc()
            kernel_time = timer.diff_time
            # Always copy to CPU
            timer.tic()
            dnn_output_cpu = {k: v.to(configs.CPU) for k, v in dnn_output.items()}
            timer.toc()
            copy_to_cpu_time = timer.diff_time
            # Then compare the golden with the output
            timer.tic()
            errors = 0
            if args.generate is False:
                errors = compare_detection(output_dict=dnn_output_cpu,
                                           golden_dict=golden[batch_id],
                                           gt_targets=gt_targets[batch_id],
                                           batch_id=batch_id,
                                           output_logger=terminal_logger,
                                           float_threshold=float_threshold,
                                           text_encoder_type=text_encoder_type,
                                           coco_api=coco_api)
            else:
                golden.append(dnn_output_cpu)

            timer.toc()
            comparison_time = timer.diff_time

            # Reload all the memories after error
            if errors != 0:
                if terminal_logger:
                    terminal_logger.info("RELOADING THE MODEL AND THE INPUTS AFTER ERROR")
                del input_list
                del model
                # Free cuda memory
                torch.cuda.empty_cache()
                # Everything in the same list
                [golden, input_list, gt_targets, model, input_captions, text_encoder_type, coco_api] = torch.load(
                    args.goldpath)

            # Printing timing information
            print_setup_iteration(batch_id=batch_id, comparison_time=comparison_time, copy_to_cpu_time=copy_to_cpu_time,
                                  errors=errors, kernel_time=kernel_time, setup_iteration=setup_iteration,
                                  terminal_logger=terminal_logger)
            batch_id += 1
        setup_iteration += 1

    if args.generate is True:
        torch.save(
            obj=[golden, input_list, gt_targets, model, input_captions, text_encoder_type, coco_api],
            f=args.goldpath
        )
        check_dnn_accuracy(predicted=golden, ground_truth=gt_targets, output_logger=terminal_logger, coco_api=coco_api,
                           text_encoder_type=text_encoder_type)

    if terminal_logger:
        terminal_logger.debug("Finish computation.")

    dnn_log_helper.end_log_file()
