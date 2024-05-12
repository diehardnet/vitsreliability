#!/usr/bin/python3
import argparse
import logging

import torch
import os
import console_logger
import configs

from typing import Union

import dnn_log_helper

from setup_grounding_dino import SetupGroundingDINO
from setup_selective_ecc import SetupSelectiveECC
from setup_vits import SetupVits

import common


def parse_args() -> argparse.Namespace:
    """ Parse the args and return an args namespace and the tostring from the args    """
    parser = argparse.ArgumentParser(description='PyTorch ViTs radiation setup', add_help=True)
    parser.add_argument('--iterations', default=1879048191, help="Iterations to run forever", type=int)
    parser.add_argument('--testsamples', default=128, help="Test samples to be used in the test.", type=int)
    parser.add_argument('--generate', default=False, action="store_true", help="Set this flag to generate the gold")
    parser.add_argument('--disableconsolelog', default=False, action="store_true",
                        help="Set this flag disable console logging")
    parser.add_argument('--goldpath', help="Path to the gold file")
    parser.add_argument('--checkpointpath', help="Path to checkpoint")
    parser.add_argument('--configpath', help="Path to configuration file")
    parser.add_argument('--batchsize', type=int, help="Batch size to be used.", default=1)
    # for ViTs
    parser.add_argument('--timm', default=False, action="store_true", help='Use a timm model instead of model file')
    # Only for pytorch 2.0
    parser.add_argument('--usetorchcompile', default=False, action="store_true",
                        help="Disable or enable torch compile (GPU Arch >= 700)")
    parser.add_argument('--hardenedid', default=False, action="store_true",
                        help="Disable or enable HardenedIdentity. Work only for the profiled models.")

    parser.add_argument('--setup_type', help="Setup type", choices=configs.ALL_SETUP_TYPES, type=str, required=True)

    parser.add_argument('--precision', help="Float precision", choices=configs.ALLOWED_MODEL_PRECISIONS, type=str,
                        required=True, default=configs.FP32)

    parser.add_argument('--textprompt', help="For the multimodal models define the text prompt",
                        type=str, required=False, default='')

    parser.add_argument('--dataset', help="For some models it's necessary to specify the dataset",
                        type=str, required=False, default=configs.IMAGENET, choices=configs.DATASETS)

    parser.add_argument('--floatthreshold', help="Float value threshold to consider a failure",
                        type=float, required=True, default=1e-3)

    parser.add_argument('--loghelperinterval', help="LogHelper interval of iteration logging",
                        type=int, required=True, default=1)

    parser.add_argument('--model', help="Model name", choices=configs.ALL_POSSIBLE_MODELS, type=str, required=True)

    args = parser.parse_args()

    if args.testsamples % args.batchsize != 0:
        dnn_log_helper.log_and_crash(fatal_string="Test samples should be multiple of batch size")

    # Check if it is only to generate the gold values
    if args.generate is True:
        args.iterations = 1

    if args.usetorchcompile is True:
        dnn_log_helper.log_and_crash(fatal_string="Torch compile is not savable yet.")

    return args


@torch.no_grad()
def run_setup(
        args: argparse.Namespace,
        setup_object: Union[SetupGroundingDINO, SetupSelectiveECC, SetupVits],
        terminal_logger: logging.Logger
):
    args_dict = vars(args)
    log_args = dict(framework_name="PyTorch", torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
                    activate_logging=not args.generate, **args_dict)
    dnn_log_helper.start_setup_log_file(**log_args)

    # Check if a device is ok and disable grad
    common.check_and_setup_gpu()

    # Defining a timer
    timer = common.Timer()
    # Load if it is not a gold generating op
    timer.tic()
    setup_object.load_data_at_test()
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
        while batch_id < setup_object.num_batches:
            timer.tic()
            dnn_log_helper.start_iteration()
            dnn_output = setup_object(batch_id=batch_id)
            torch.cuda.synchronize(device=configs.GPU_DEVICE)
            dnn_log_helper.end_iteration()
            timer.toc()
            kernel_time = timer.diff_time
            # Always copy to CPU
            timer.tic()
            dnn_output_cpu = setup_object.copy_to_cpu(dnn_output=dnn_output)
            timer.toc()
            copy_to_cpu_time = timer.diff_time
            # Then compare the golden with the output
            timer.tic()
            # If generate errors==0, otherwise it will compare the output
            errors = setup_object.post_inference_process(dnn_output_cpu=dnn_output_cpu, batch_id=batch_id)
            timer.toc()
            comparison_time = timer.diff_time

            # Reload all the memories after error
            if errors != 0:
                setup_object.clear_gpu_memory_and_reload()

            # Printing timing information
            setup_object.print_setup_iteration(batch_id=batch_id, comparison_time=comparison_time,
                                               copy_to_cpu_time=copy_to_cpu_time,
                                               errors=errors, kernel_time=kernel_time,
                                               setup_iteration=setup_iteration)
            batch_id += 1
        setup_iteration += 1

    if args.generate is True:
        setup_object.save_setup_data_to_gold_file()
        setup_object.check_dnn_accuracy()

    if terminal_logger:
        terminal_logger.debug("Finish computation.")

    dnn_log_helper.end_log_file()


def main():
    args = parse_args()
    # Terminal console
    main_logger_name = str(os.path.basename(__file__)).replace(".py", "")
    terminal_logger = console_logger.ColoredLogger(main_logger_name) if args.disableconsolelog is False else None

    setup_object = None
    if args.setup_type == configs.GROUNDING_DINO:
        setup_object = SetupGroundingDINO(args=args, output_logger=terminal_logger)
    elif args.setup_type == configs.MAXIMALS:
        pass
    elif args.setup_type == configs.SELECTIVE_ECC:
        pass
    elif args.setup_type == configs.VITS:
        setup_object = SetupVits(args=args, output_logger=terminal_logger)
    else:
        dnn_log_helper.log_and_crash(fatal_string=f"Code type {args.code_type} not implemented")

    run_setup(args=args, setup_object=setup_object, terminal_logger=terminal_logger)


if __name__ == '__main__':
    try:
        main()
    except Exception as main_function_exception:
        dnn_log_helper.log_and_crash(fatal_string=f"EXCEPTION:{main_function_exception}")
