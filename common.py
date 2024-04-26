import argparse
import time
from typing import Union, Tuple, List

import torch
import dnn_log_helper
import configs


class Timer:
    time_measure = 0

    def tic(self): self.time_measure = time.time()

    def toc(self): self.time_measure = time.time() - self.time_measure

    @property
    def diff_time(self): return self.time_measure

    @property
    def diff_time_str(self): return str(self)

    def __str__(self): return f"{self.time_measure:.4f}s"

    def __repr__(self): return str(self)


def equal(rhs: torch.Tensor, lhs: torch.Tensor, threshold: Union[None, float]) -> bool:
    """ Compare based or not in a threshold, if the threshold is none then it is equal comparison    """
    if threshold is not None:
        return bool(
            torch.all(
                torch.le(
                    torch.abs(
                        torch.subtract(rhs, lhs)
                    ), threshold
                )
            )
        )
    else:
        return bool(torch.equal(rhs, lhs))


def describe_error(input_tensor: torch.tensor) -> Tuple[int, int, float, float]:
    flattened_tensor = input_tensor.flatten()
    is_nan_tensor, is_inf_tensor = torch.isnan(flattened_tensor), torch.isinf(flattened_tensor)
    has_nan, has_inf = int(torch.any(is_nan_tensor)), int(torch.any(is_inf_tensor))
    filtered_tensor = flattened_tensor[~is_nan_tensor & ~is_inf_tensor]
    min_val = float(torch.min(filtered_tensor)) if filtered_tensor.numel() > 0 else 0
    max_val = float(torch.max(filtered_tensor)) if filtered_tensor.numel() > 0 else 0
    return has_nan, has_inf, min_val, max_val


def check_and_setup_gpu() -> None:
    # Disable all torch grads
    torch.set_grad_enabled(mode=False)
    if torch.cuda.is_available() is False:
        dnn_log_helper.log_and_crash(fatal_string=f"Device {configs.GPU_DEVICE} not available.")
    dev_capability = torch.cuda.get_device_capability()
    if dev_capability[0] < configs.MINIMUM_DEVICE_CAPABILITY:
        dnn_log_helper.log_and_crash(fatal_string=f"Device cap:{dev_capability} is too old.")


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    """ Parse the args and return an args namespace and the tostring from the args    """
    parser = argparse.ArgumentParser(description='PyTorch Maximals radiation setup', add_help=True)
    # parser = argparse.ArgumentParser(description='PyTorch DNN radiation setup')
    parser.add_argument('--iterations', default=int(1e12), help="Iterations to run forever", type=int)
    parser.add_argument('--testsamples', default=128, help="Test samples to be used in the test.", type=int)
    parser.add_argument('--generate', default=False, action="store_true", help="Set this flag to generate the gold")
    parser.add_argument('--disableconsolelog', default=False, action="store_true",
                        help="Set this flag disable console logging")
    parser.add_argument('--goldpath', help="Path to the gold file")
    parser.add_argument('--checkpointdir', help="Path to checkpoint dir")
    parser.add_argument('--model', help="Model name: " + ", ".join(configs.ALL_POSSIBLE_MODELS),
                        type=str, default=configs.RESNET50D_IMAGENET_TIMM)
    parser.add_argument('--batchsize', type=int, help="Batch size to be used.", default=1)
    # Only for pytorch 2.0
    parser.add_argument('--usetorchcompile', default=False, action="store_true",
                        help="Disable or enable torch compile (GPU Arch >= 700)")
    parser.add_argument('--hardenedid', default=False, action="store_true",
                        help="Disable or enable HardenedIdentity. Work only for the profiled models.")
    args = parser.parse_args()

    if args.testsamples % args.batchsize != 0:
        dnn_log_helper.log_and_crash(fatal_string="Test samples should be multiple of batch size")

    # Check if it is only to generate the gold values
    if args.generate is True:
        args.iterations = 1

    if args.usetorchcompile is True:
        dnn_log_helper.log_and_crash(fatal_string="Torch compile is not savable yet.")

    # Only valid models
    if args.model not in configs.ALL_POSSIBLE_MODELS:
        dnn_log_helper.log_and_crash(fatal_string=f"model == {args.model} is invalid")

    args_text_list = [f"{k}={v}" for k, v in vars(args).items()]
    return args, args_text_list
