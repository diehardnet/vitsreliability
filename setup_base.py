import logging
from typing import Union
import argparse


class SetupBase:
    def __init__(self, args: argparse.Namespace, output_logger: logging.Logger):
        self.output_logger = output_logger
        self.model_checkpoint_path = args.checkpointpath
        self.model_config_path = args.configpath
        self.hardened_model = args.hardenedid
        self.torch_compile = args.usetorchcompile
        self.precision = args.precision
        self.model_name = args.model
        self.batch_size = args.batchsize
        self.test_sample = args.testsamples
        self.gold_path = args.goldpath
        self.generate = args.generate
        self.input_captions = args.textprompt
        self.float_threshold = args.floatthreshold

    def print_setup_iteration(self,
                              batch_id: Union[int, None], comparison_time: float, copy_to_cpu_time: float,
                              errors: int, kernel_time: float, setup_iteration: int) -> None:
        if self.output_logger:
            wasted_time = comparison_time + copy_to_cpu_time
            time_pct = (wasted_time / (wasted_time + kernel_time)) * 100.0
            iteration_out = f"It:{setup_iteration:<3} batch_id:{batch_id:<3} inference time:{kernel_time:.5f}, "
            iteration_out += f"compare time:{comparison_time:.5f} copy time:{copy_to_cpu_time:.5f} "
            iteration_out += f"(wasted:{time_pct:.1f}%) errors:{errors}"
            self.output_logger.debug(iteration_out)
