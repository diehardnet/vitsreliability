import time
from typing import Union, Tuple

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

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(configs.TORCH_SEED)
    torch.use_deterministic_algorithms(mode=True, warn_only=True)
