#!/usr/bin/python3
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
import hardened_identity
from common import Timer, equal, describe_error


def replace_identity(module: torch.nn.Module, model_name: str):
    """Recursively put desired module in nn.module module.
    """
    pass
    # # go through all attributes of module nn.module (e.g., network or layer) and put batch norms if present
    # for attr_str in dir(module):
    #     target_attr = getattr(module, attr_str)
    #     if type(target_attr) == torch.nn.Identity:
    #         new_identity = hardened_identity.HardenedIdentity(model_name=model_name)
    #         setattr(module, attr_str, new_identity)
    #
    # # Iterate through immediate child modules. Note, our code does the recursion no need to use named_modules()
    # for _, immediate_child_module in module.named_children():
    #     replace_identity(module=immediate_child_module, model_name=model_name)


def load_model(model_name: str, hardened_model: bool, torch_compile: bool) -> [torch.nn.Module, tv_transforms.Compose]:
    # The First option is the baseline option
    # model = timm.create_model(model_name, pretrained=True)
    # model.eval()
    # if hardened_model:
    #     replace_identity(module=model, model_name=model_name)
    # # Disable also parameter grads
    # model.zero_grad(set_to_none=True)
    # model = model.to(configs.GPU_DEVICE)
    # config = timm.data.resolve_data_config({}, model=model)
    # transform = timm.data.transforms_factory.create_transform(**config)
    # # TODO: Implement when the serialization is possible
    # if torch_compile is True:
    #     # model = torch.compile(model=model, mode="max-autotune")
    #     dnn_log_helper.log_and_crash(fatal_string="Up to now it's not possible to serialize compiled models "
    #                                               "(github.com/pytorch/pytorch/issues/101107#issuecomment-1542688089)")
    # return model, transform
    pass


def load_data_at_test(gold_path: str) -> Tuple:
    # gold_model_path = gold_path.replace(".pt", "_traced_model.pt")
    # The order -> golden, input_list, input_labels, model, original_dataset_order
    # [golden, input_list, input_labels, model, original_dataset_order] = torch.load(gold_path)
    # # model = torch.jit.load(gold_model_path)
    # model.zero_grad(set_to_none=True)
    # return golden, input_labels, input_list, model, original_dataset_order
    pass


def save_data_at_test(model: torch.nn.Module,
                      golden: torch.tensor,
                      input_list: List[torch.tensor],
                      input_labels: List,
                      original_dataset_order: List,
                      gold_path: str) -> None:
    # gold_model_path = gold_path.replace(".pt", "_traced_model.pt")
    # output_file = [golden, input_list, input_labels, model, original_dataset_order]
    # torch.save(output_file, gold_path)
    # torch.jit.save(model, gold_model_path)
    pass


def load_dataset(batch_size: int, dataset: str, test_sample: int,
                 transform: tv_transforms.Compose) -> Tuple[List, List, List]:
    # Using sequential sampler is the same as passing the shuffle=False
    # Using the RandomSampler with a fixed seed is better
    # input_dataset, input_labels, original_order = list(), list(), list()
    # sampler_generator = torch.Generator(device="cpu")
    # sampler_generator.manual_seed(configs.SAMPLER_SEED)
    # test_set = None
    # if dataset == configs.IMAGENET:
    #     test_set = tv_datasets.imagenet.ImageNet(root=configs.IMAGENET_DATASET_DIR, transform=transform,
    #                                              split='val')
    # elif dataset == configs.COCO:
    #     # This is only used when performing det/seg and these models already perform transforms
    #     test_set = tv_datasets.coco.CocoDetection(root=configs.COCO_DATASET_VAL,
    #                                               annFile=configs.COCO_DATASET_ANNOTATIONS,
    #                                               transform=transform)
    #     # test_loader = torch.utils.data.DataLoader(dataset=test_set, sampler=subset, batch_size=batch_size,
    #     #                                           shuffle=False, pin_memory=True, collate_fn=lambda x: x)
    #     # for iterator in test_loader:
    #     #     inputs, labels = list(), list()
    #     #     for input_i, label_i in iterator:
    #     #         inputs.append(input_i)
    #     #         labels.append(label_i)
    #     #     # Only the inputs must be in the device
    #     #     input_dataset.append(torch.stack(inputs).to(configs.DEVICE))
    #     #     # Labels keys dict_keys(['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id'])
    #     #     input_labels.append(labels)
    #
    # subset = torch.utils.data.RandomSampler(data_source=test_set, replacement=False, num_samples=test_sample,
    #                                         generator=sampler_generator)
    # test_loader = torch.utils.data.DataLoader(dataset=test_set, sampler=subset, batch_size=batch_size,
    #                                           shuffle=False, pin_memory=True)
    # for inputs, labels in test_loader:
    #     # Only the inputs must be in the device
    #     input_dataset.append(inputs.to(configs.GPU_DEVICE))
    #     input_labels.append(labels)
    #
    # return input_dataset, input_labels, original_order
    pass


def compare_classification(output_tensor: torch.tensor,
                           golden_tensor: torch.tensor,
                           golden_top_k_labels: torch.tensor,
                           ground_truth_labels: torch.tensor,
                           batch_id: int,
                           top_k: int,
                           output_logger: logging.Logger, float_threshold: float, original_dataset_order: range) -> int:
    # output_errors = 0
    # # Iterate over the batches
    # for img_id, (output_batch, golden_batch, golden_batch_label, ground_truth_label) in enumerate(
    #         zip(output_tensor, golden_tensor, golden_top_k_labels, ground_truth_labels)):
    #     # using the same approach as the detection, compare only the positions that differ
    #     if equal(rhs=golden_batch, lhs=output_batch, threshold=float_threshold) is False:
    #         # ------------ Check if there is a Critical error ----------------------------------------------------------
    #         # top_k_batch_label_flatten = torch.topk(output_batch, k=top_k).indices.squeeze(0).flatten()
    #         top_k_batch_label_flatten = get_top_k_labels(input_tensor=output_batch, top_k=top_k).flatten()
    #         golden_batch_label_flatten = golden_batch_label.flatten()
    #         err_string = f"batch:{batch_id} imgid:{img_id} "
    #         for i, (tpk_gold, tpk_found) in enumerate(zip(golden_batch_label_flatten, top_k_batch_label_flatten)):
    #             # Both are integers, and log only if it is feasible
    #             if tpk_found != tpk_gold:
    #                 output_errors += 1
    #                 error_detail_ctr = f"critical {err_string} i:{i} g:{tpk_gold} o:{tpk_found} gt:{ground_truth_label}"
    #                 if output_logger:
    #                     output_logger.error(error_detail_ctr)
    #                 dnn_log_helper.log_error_detail(error_detail_ctr)
    #
    #         # ------------ Check error on the whole output -------------------------------------------------------------
    #         # Not necessary to save everything, only the good info
    #         # Data on output tensor
    #         has_nan, has_inf, min_val, max_val = describe_error(input_tensor=output_batch)
    #         error_detail_out = f"{err_string} output_t nan:{has_nan} inf:{has_inf} min:{min_val} max:{max_val} "
    #         # Data on abs differences
    #         abs_diff = torch.abs(torch.subtract(golden_batch, output_batch))
    #         has_nan_diff, has_inf_diff, min_val_diff, max_val_diff = describe_error(input_tensor=abs_diff)
    #         error_detail_out += f"diff_t nan:{has_nan_diff} inf:{has_inf_diff} min:{min_val_diff} max:{max_val_diff}"
    #         output_errors += 1
    #         if output_logger:
    #             output_logger.error(error_detail_out)
    #         dnn_log_helper.log_error_detail(error_detail_out)
    #
    # return output_errors
    pass


def compare(output_tensor: torch.tensor,
            golden: Dict[str, List[torch.tensor]],
            ground_truth_labels: Union[List[torch.tensor], List[dict]],
            batch_id: int,
            output_logger: logging.Logger, dnn_goal: str, setup_iteration: int, float_threshold: float,
            original_dataset_order: range) -> int:
    pass
    # golden_tensor = golden["output_list"][batch_id]
    # # global TEST
    # # TEST += 1
    # # if TEST == 3:
    # #     # # Simulate a non-critical error
    # #     # output_tensor[3, 0] *= 0.9
    # #     # Simulate a critical error
    # #     output_tensor[3, 6] = 39304
    # #     # Shape SDC
    # #     # output_tensor = torch.reshape(output_tensor, (4, 3200))
    #
    # # Make sure that they are on CPU
    # out_is_cuda, golden_is_cuda = output_tensor.is_cuda, golden_tensor.is_cuda
    # if out_is_cuda or golden_is_cuda:
    #     dnn_log_helper.log_and_crash(
    #         fatal_string=f"Tensors are not on CPU. OUT IS CUDA:{out_is_cuda} GOLDEN IS CUDA:{golden_is_cuda}")
    #
    # # First check if the tensors are equal or not
    # if equal(rhs=output_tensor, lhs=golden_tensor, threshold=float_threshold) is True:
    #     return 0
    #
    # # ------------ Check the size of the tensors
    # if output_tensor.shape != golden_tensor.shape:
    #     info_detail = f"shape-diff g:{golden_tensor.shape} o:{output_tensor.shape}"
    #     if output_logger:
    #         output_logger.error(info_detail)
    #     dnn_log_helper.log_info_detail(info_detail)
    #
    # # ------------ Main check
    # output_errors = 0
    # if dnn_goal == configs.CLASSIFICATION:
    #     golden_top_k_labels = golden["top_k_labels"][batch_id]
    #     output_errors = compare_classification(output_tensor=output_tensor,
    #                                            golden_tensor=golden_tensor,
    #                                            golden_top_k_labels=golden_top_k_labels,
    #                                            ground_truth_labels=ground_truth_labels[batch_id],
    #                                            batch_id=batch_id,
    #                                            top_k=configs.CLASSIFICATION_CRITICAL_TOP_K,
    #                                            output_logger=output_logger, float_threshold=float_threshold,
    #                                            original_dataset_order=original_dataset_order)
    #
    # # ------------ log and return
    # if output_errors != 0:
    #     dnn_log_helper.log_error_count(error_count=output_errors)
    # return output_errors


def check_dnn_accuracy(predicted: Union[Dict[str, List[torch.tensor]], torch.tensor], ground_truth: List[torch.tensor],
                       output_logger: logging.Logger, dnn_goal: str) -> None:
    pass
    # correct, gt_count = 0, 0
    # if dnn_goal == configs.CLASSIFICATION:
    #     predicted = predicted["top_k_labels"]
    #     for pred, gt in zip(predicted, ground_truth):
    #         gt_count += gt.shape[0]
    #         correct += torch.sum(torch.eq(pred, gt))
    # elif dnn_goal == configs.SEGMENTATION:
    #     dnn_log_helper.log_and_crash(fatal_string="Checking segmentation is not implemented")
    #
    # if output_logger:
    #     correctness = correct / gt_count
    #     output_logger.debug(f"Correct predicted samples:{correct} - ({correctness * 100:.2f}%)")
    #     correctness_threshold = 0.7
    #     if correctness < correctness_threshold:
    #         dnn_log_helper.log_and_crash(fatal_string=f"ACCURACY LOWER THAN {correctness_threshold * 100.0}%")
    #


# def update_golden(golden: Dict[str, list], output: torch.tensor, dnn_goal: str) -> Dict[str, list]:
#     if dnn_goal == configs.CLASSIFICATION:
#         golden["output_list"].append(output)
#         golden["top_k_labels"].append(torch.tensor(
#             [get_top_k_labels(input_tensor=output_batch, top_k=configs.CLASSIFICATION_CRITICAL_TOP_K) for output_batch
#              in output]
#         ))
#     elif dnn_goal == configs.SEGMENTATION:
#         golden["output_list"].append(output)
#
#     return golden


# def copy_output_to_cpu(dnn_output: Union[torch.tensor, collections.OrderedDict],
#                        dnn_goal: str) -> torch.tensor:
#     if dnn_goal == configs.CLASSIFICATION:
#         return dnn_output.to("cpu")
#     elif dnn_goal == configs.SEGMENTATION:
#         return dnn_output["out"].to('cpu')

def print_setup_iteration(batch_id: Union[int, None], comparison_time: float, copy_to_cpu_time: float, errors: int,
                          kernel_time: float, setup_iteration: int, terminal_logger: logging.Logger) -> None:
    # if terminal_logger:
    #     wasted_time = comparison_time + copy_to_cpu_time
    #     time_pct = (wasted_time / (wasted_time + kernel_time)) * 100.0
    #     iteration_out = f"It:{setup_iteration:<3} batch_id:{batch_id:<3} inference time:{kernel_time:.5f}, "
    #     iteration_out += f"compare time:{comparison_time:.5f} copy time:{copy_to_cpu_time:.5f} "
    #     iteration_out += f"(wasted:{time_pct:.1f}%) errors:{errors}"
    #     terminal_logger.debug(iteration_out)
    pass


# Force no grad
@torch.no_grad()
def run_setup_selective_ecc(args: argparse.Namespace, args_text_list: List[str]):
    pass
    # args, args_text_list = parse_args()
    # # Define DNN goal
    # dnn_goal = configs.DNN_GOAL[args.model]
    # dataset = configs.DATASETS[dnn_goal]
    # float_threshold = configs.DNN_THRESHOLD[dnn_goal]
    # dnn_log_helper.start_setup_log_file(framework_name="PyTorch", torch_version=torch.__version__,
    #                                     gpu=torch.cuda.get_device_name(), timm_version=timm.__version__,
    #                                     args_conf=args_text_list, dnn_name=args.model,
    #                                     activate_logging=not args.generate, dnn_goal=dnn_goal, dataset=dataset,
    #                                     float_threshold=float_threshold)
    #
    # # Check if a device is ok and disable grad
    # check_and_setup_gpu()
    #
    # # Defining a timer
    # timer = Timer()
    # # Terminal console
    # main_logger_name = str(os.path.basename(__file__)).replace(".py", "")
    # terminal_logger = console_logger.ColoredLogger(main_logger_name) if args.disableconsolelog is False else None
    #
    # # Load if it is not a gold generating op
    # timer.tic()
    # # This will save time
    # if args.generate is False:
    #     # Save everything in the same list
    #     golden, input_labels, input_list, model, original_dataset_order = load_data_at_test(gold_path=args.goldpath)
    # else:
    #     # The First step is to load the inputs in the memory
    #     # Load the model
    #     model, transform = load_model(model_name=args.model, hardened_model=args.hardenedid,
    #                                   torch_compile=args.usetorchcompile)
    #     input_list, input_labels, original_dataset_order = load_dataset(batch_size=args.batchsize, dataset=dataset,
    #                                                                     test_sample=args.testsamples,
    #                                                                     transform=transform)
    #     golden: Dict[str, List[torch.tensor]] = dict(output_list=list(), top_k_labels=list())
    #     # # Tracing the model with example input
    #     # model = torch.jit.trace(model, input_list[0])
    #     # # Invoking torch.jit.freeze
    #     # model = torch.jit.freeze(model)
    #     # model = torch.jit.script(model)
    #
    # timer.toc()
    # golden_load_diff_time = timer.diff_time_str
    #
    # if terminal_logger:
    #     terminal_logger.debug("\n".join(args_text_list))
    #     terminal_logger.debug(f"Time necessary to load the golden outputs, model, and inputs: {golden_load_diff_time}")
    #
    # # Main setup loop
    # setup_iteration = 0
    # while setup_iteration < args.iterations:
    #     # Loop over the input list
    #     batch_id = 0  # It must be like this, because I may reload the list in the middle of the process
    #     while batch_id < len(input_list):
    #         timer.tic()
    #         dnn_log_helper.start_iteration()
    #         dnn_output = model(input_list[batch_id])
    #         torch.cuda.synchronize(device=configs.GPU_DEVICE)
    #         dnn_log_helper.end_iteration()
    #         timer.toc()
    #         kernel_time = timer.diff_time
    #         # Always copy to CPU
    #         timer.tic()
    #         dnn_output_cpu = copy_output_to_cpu(dnn_output=dnn_output, dnn_goal=dnn_goal)
    #         timer.toc()
    #         copy_to_cpu_time = timer.diff_time
    #         # Then compare the golden with the output
    #         timer.tic()
    #         errors = 0
    #         if args.generate is False:
    #             errors = compare(output_tensor=dnn_output_cpu,
    #                              golden=golden,
    #                              ground_truth_labels=input_labels,
    #                              batch_id=batch_id,
    #                              output_logger=terminal_logger, dnn_goal=dnn_goal, setup_iteration=setup_iteration,
    #                              float_threshold=float_threshold, original_dataset_order=original_dataset_order)
    #         else:
    #             golden = update_golden(golden=golden, output=dnn_output_cpu, dnn_goal=dnn_goal)
    #
    #         timer.toc()
    #         comparison_time = timer.diff_time
    #
    #         # Reload all the memories after error
    #         if errors != 0:
    #             if terminal_logger:
    #                 terminal_logger.info("RELOADING THE MODEL AND THE INPUTS AFTER ERROR")
    #             del input_list
    #             del model
    #             # Free cuda memory
    #             torch.cuda.empty_cache()
    #             # Everything in the same list
    #             golden, input_labels, input_list, model, original_dataset_order = load_data_at_test(
    #                 gold_path=args.goldpath)
    #
    #         # Printing timing information
    #         print_setup_iteration(batch_id=batch_id, comparison_time=comparison_time, copy_to_cpu_time=copy_to_cpu_time,
    #                               errors=errors, kernel_time=kernel_time, setup_iteration=setup_iteration,
    #                               terminal_logger=terminal_logger)
    #         batch_id += 1
    #     setup_iteration += 1
    #
    # if args.generate is True:
    #     save_data_at_test(model=model, golden=golden, input_list=input_list, input_labels=input_labels,
    #                       original_dataset_order=original_dataset_order, gold_path=args.goldpath)
    #     check_dnn_accuracy(predicted=golden, ground_truth=input_labels, output_logger=terminal_logger,
    #                        dnn_goal=dnn_goal)
    #
    # if terminal_logger:
    #     terminal_logger.debug("Finish computation.")
    #
    # dnn_log_helper.end_log_file()
