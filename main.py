#!/usr/bin/python3
import argparse
from typing import Tuple, List

import configs
import dnn_log_helper
from setup_grounding_dino import run_setup_grounding_dino
from setup_selective_ecc import run_setup_selective_ecc


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
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
    # Only for pytorch 2.0
    parser.add_argument('--usetorchcompile', default=False, action="store_true",
                        help="Disable or enable torch compile (GPU Arch >= 700)")
    parser.add_argument('--hardenedid', default=False, action="store_true",
                        help="Disable or enable HardenedIdentity. Work only for the profiled models.")

    parser.add_argument('--setup_type', help="Setup type", choices=configs.ALL_SETUP_TYPES, type=str, required=True)

    parser.add_argument('--model', help="Model name", choices=configs.ALL_POSSIBLE_MODELS, type=str, required=True)

    args = parser.parse_args()

    if args.testsamples % args.batchsize != 0:
        dnn_log_helper.log_and_crash(fatal_string="Test samples should be multiple of batch size")

    # Check if it is only to generate the gold values
    if args.generate is True:
        args.iterations = 1

    if args.usetorchcompile is True:
        dnn_log_helper.log_and_crash(fatal_string="Torch compile is not savable yet.")

    args_text_list = [f"{k}={v}" for k, v in vars(args).items()]
    return args, args_text_list


def main():
    args, args_text_list = parse_args()

    if args.setup_type == configs.GROUNDING_DINO:
        run_setup_grounding_dino(args=args, args_text_list=args_text_list)
    elif args.setup_type == configs.MAXIMALS:
        pass
    elif args.setup_type == configs.SELECTIVE_ECC:
        run_setup_selective_ecc(args=args, args_text_list=args_text_list)
    else:
        dnn_log_helper.log_and_crash(fatal_string=f"Code type {args.code_type} not implemented")


if __name__ == '__main__':
    try:
        main()
    except Exception as main_function_exception:
        dnn_log_helper.log_and_crash(fatal_string=f"EXCEPTION:{main_function_exception}")
