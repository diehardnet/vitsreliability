import argparse
import logging
import os
import random
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


def load_model(model_checkpoint_path: str, model_config_path: str, hardened_model: bool,
               torch_compile: bool) -> [str, torch.nn.Module]:
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
        raise NotImplementedError("Hardened Id not ready")

    # TODO: Implement when the serialization is possible
    if torch_compile is True:
        # model = torch.compile(model=model, mode="max-autotune")
        dnn_log_helper.log_and_crash(fatal_string="Up to now it's not possible to serialize compiled models "
                                                  "(github.com/pytorch/pytorch/issues/101107#issuecomment-1542688089)")
    return cfg_args.text_encoder_type, model


def load_dataset(batch_size: int, test_sample: int) -> Tuple[List, List, List, torchvision.datasets.coco.CocoDetection]:
    # build dataloader
    transform = gdino_transforms.Compose(
        [
            gdino_transforms.RandomResize([800], max_size=1333),
            gdino_transforms.ToTensor(),
            gdino_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = GDINOCocoDetection(configs.COCO_DATASET_VAL, configs.COCO_DATASET_ANNOTATIONS, transforms=transform)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1,
                                              collate_fn=gdino_collate_fn)

    # # build post processor
    # tokenlizer = gdino_get_tokenlizer.get_tokenlizer(text_encoder_type)
    # postprocessor = GDINOPostProcessCocoGrounding(coco_api=dataset.coco, tokenlizer=tokenlizer)
    #
    # # build evaluator
    # evaluator = GDINOCocoGroundingEvaluator(dataset.coco, iou_types=("bbox",), useCats=True)
    coco_api = dataset.coco

    # build captions
    category_dict = dataset.coco.dataset['categories']
    cat_list = [item['name'] for item in category_dict]
    caption = " . ".join(cat_list) + ' .'
    # TODO: try a different prompt
    print("Input text prompt:", caption)
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


def get_iou_for_single_image(outputs, targets, text_encoder_type: str,
                             coco_api: torchvision.datasets.coco.CocoDetection) -> [float, float]:
    # build post processor
    tokenlizer = gdino_get_tokenlizer.get_tokenlizer(text_encoder_type)
    postprocessor = GDINOPostProcessCocoGrounding(coco_api=coco_api, tokenlizer=tokenlizer)
    # build evaluator
    evaluator = GDINOCocoGroundingEvaluator(coco_api, iou_types=("bbox",), useCats=True)
    orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
    results = postprocessor(outputs[1], orig_target_sizes)
    cocogrounding_res = {target["image_id"]: output for target, output in zip(targets, results)}
    evaluator.update(cocogrounding_res)
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    average_precision, average_recall = 0, 0
    return average_precision, average_recall


def compare_detection(
        output_tensor: torch.tensor,
        golden_tensor: torch.tensor,
        ground_truth_labels: torch.tensor,
        batch_id: int,
        output_logger: logging.Logger, float_threshold: float,
        text_encoder_type: str,
        coco_api: torchvision.datasets.coco.CocoDetection
) -> int:
    # global TEST
    # TEST += 1
    # if TEST == 3:
    #     # # Simulate a non-critical error
    #     # output_tensor[3, 0] *= 0.9
    #     # Simulate a critical error
    #     output_tensor[3, 6] = 39304
    #     # Shape SDC
    #     # output_tensor = torch.reshape(output_tensor, (4, 3200))

    # Make sure that they are on CPU
    out_is_cuda, golden_is_cuda = output_tensor.is_cuda, golden_tensor.is_cuda
    if out_is_cuda or golden_is_cuda:
        dnn_log_helper.log_and_crash(
            fatal_string=f"Tensors are not on CPU. OUT IS CUDA:{out_is_cuda} GOLDEN IS CUDA:{golden_is_cuda}")

    # First check if the tensors are equal or not
    if common.equal(rhs=output_tensor, lhs=golden_tensor, threshold=float_threshold) is True:
        return 0

    # ------------ Check the size of the tensors
    if output_tensor.shape != golden_tensor.shape:
        info_detail = f"shape-diff g:{golden_tensor.shape} o:{output_tensor.shape}"
        if output_logger:
            output_logger.error(info_detail)
        dnn_log_helper.log_info_detail(info_detail)

    # TODO: Prepare a way for only single images for the critical failure detection

    output_errors = 0
    # Iterate over the batches
    for img_id, (output_batch, golden_batch, ground_truth_label) in enumerate(
            zip(output_tensor, golden_tensor, ground_truth_labels)):
        # using the same approach as the detection, compare only the positions that differ
        if common.equal(rhs=golden_batch, lhs=output_batch, threshold=float_threshold) is False:
            # ------------ Check if there is a Critical error ----------------------------------------------------------
            avg_precision, avg_recall = get_iou_for_single_image(outputs=output_batch, targets=golden_batch,
                                                                 text_encoder_type=text_encoder_type,
                                                                 coco_api=coco_api)
            err_string = f"batch:{batch_id} imgid:{img_id} gd_pre:{avg_precision} gd_rec:{avg_recall} "
            avg_precision, avg_recall = get_iou_for_single_image(outputs=output_batch, targets=ground_truth_label,
                                                                 text_encoder_type=text_encoder_type,
                                                                 coco_api=coco_api)
            err_string += f" gt_pre:{avg_precision} gt_rec:{avg_recall} "
            # ------------ Check error on the whole output -------------------------------------------------------------
            # Not necessary to save everything, only the good info
            # Data on output tensor
            has_nan, has_inf, min_val, max_val = common.describe_error(input_tensor=output_batch)
            error_detail_out = f"{err_string} output_t nan:{has_nan} inf:{has_inf} min:{min_val} max:{max_val} "
            # Data on abs differences
            abs_diff = torch.abs(torch.subtract(golden_batch, output_batch))
            has_nan_diff, has_inf_diff, min_val_diff, max_val_diff = common.describe_error(input_tensor=abs_diff)
            error_detail_out += f"diff_t nan:{has_nan_diff} inf:{has_inf_diff} min:{min_val_diff} max:{max_val_diff}"
            output_errors += 1
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
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessor(outputs[1], orig_target_sizes)
        cocogrounding_res = {target["image_id"]: output for target, output in zip(targets, results)}

        evaluator.update(cocogrounding_res)

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    if output_logger:
        stats_list = evaluator.coco_eval["bbox"].stats.tolist()
        correctness = sum(stats_list) / len(stats_list)
        output_logger.debug(f'Final results:{correctness * 100.0:.2f}%')
        correctness_threshold = 0.6  # Base on the whole dataset accuracy
        if correctness < correctness_threshold:
            dnn_log_helper.log_and_crash(fatal_string=f"ACCURACY LOWER THAN {correctness_threshold * 100.0}%")


# Force no grad
@torch.no_grad()
def run_setup_grounding_dino(args: argparse.Namespace, args_text_list: List[str]):
    # Define DNN goal
    dnn_goal = configs.DNN_GOAL[args.model]
    dataset = configs.DATASETS[dnn_goal]
    float_threshold = configs.DNN_THRESHOLD[dnn_goal]
    log_args = dict(
        framework_name="PyTorch", torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name(), args_conf=args_text_list, dnn_name=args.model,
        activate_logging=not args.generate, dnn_goal=dnn_goal, dataset=dataset,
        float_threshold=float_threshold
    )
    dnn_log_helper.start_setup_log_file(**log_args)

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
            hardened_model=args.hardenedid, torch_compile=args.usetorchcompile
        )
        input_list, gt_targets, input_captions, coco_api = load_dataset(batch_size=args.batchsize,
                                                                        test_sample=args.testsamples)
        golden = list()

    timer.toc()
    golden_load_diff_time = timer.diff_time_str

    if terminal_logger:
        terminal_logger.debug("\n".join(args_text_list))
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
                errors = compare_detection(
                    output_tensor=dnn_output_cpu,
                    golden_tensor=golden[batch_id],
                    ground_truth_labels=gt_targets[batch_id],
                    batch_id=batch_id,
                    output_logger=terminal_logger, float_threshold=float_threshold,
                    text_encoder_type=text_encoder_type,
                    coco_api=coco_api)
            else:
                golden.append((batch_id, dnn_output_cpu))

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
                [golden, input_list, gt_targets, model,
                 input_captions, text_encoder_type, coco_api] = torch.load(args.goldpath)

            # Printing timing information
            print_setup_iteration(batch_id=batch_id, comparison_time=comparison_time, copy_to_cpu_time=copy_to_cpu_time,
                                  errors=errors, kernel_time=kernel_time, setup_iteration=setup_iteration,
                                  terminal_logger=terminal_logger)
            batch_id += 1
        setup_iteration += 1

    if args.generate is True:
        torch.save(
            obj=[golden, input_list, gt_targets, model, input_captions],
            f=args.goldpath
        )
        check_dnn_accuracy(predicted=golden, ground_truth=gt_targets, output_logger=terminal_logger, coco_api=coco_api,
                           text_encoder_type=text_encoder_type)

    if terminal_logger:
        terminal_logger.debug("Finish computation.")

    dnn_log_helper.end_log_file()
