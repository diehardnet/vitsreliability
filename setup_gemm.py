import torch
import configs
import dnn_log_helper
import setup_base
import common
import logging
import argparse

class SetupGEMM(setup_base.SetupBase):
    def __init__(self, args: argparse.Namespace, output_logger: logging.Logger):
        super().__init__(args=args, output_logger=output_logger)
        self.size = args.matrix_size


    def __call__(self, batch_id, **kwargs):
        input_a, input_b = self.input_list[batch_id]
        return torch.matmul(input_a, input_b)

    def load_dataset(self) -> None:
        torch.manual_seed(configs.TORCH_SEED)
        r1, r2 = configs.GENERATOR_MIN_ABS_VALUE_GEMM, configs.GENERATOR_MAX_ABS_VALUE_GEMM
        for _ in range(self.batch_size):
            input_a = torch.FloatTensor(self.size, self.size).uniform_(r1, r2).to(configs.GPU_DEVICE)
            input_b = torch.FloatTensor(self.size, self.size).uniform_(r1, r2).to(configs.GPU_DEVICE)
            self.input_list.append((input_a, input_b))


    def check_dnn_accuracy(self) -> None:
        # No sense here as its not a model
        pass

    def __compare_output(self, output_tensor: torch.tensor, golden_tensor: torch.tensor, output_logger: logging.Logger) -> int:
        output_errors = 0

        # Get non-equal elements' indices
        # Identify non-equal elements
        diff_mask = torch.ne(output_tensor, golden_tensor)
        # Get indices where elements differ
        diff_indices = torch.nonzero(diff_mask)
        for index in diff_indices:
            i, j = index
            gold_value = golden_tensor[i, j]
            read_value = output_tensor[i, j]

            if gold_value != read_value:
                error_detail = f"p:[{i}, {j}] r:{read_value}, e:{gold_value}"
                if output_logger and output_errors < 10:
                    output_logger.debug(error_detail)

                dnn_log_helper.log_error_detail(error_detail)
                output_errors += 1

        # ------------ Check error on the whole output -------------------------------------------------------------
        # Not necessary to save everything, only the good info
        # Data on output tensor
        has_nan, has_inf, min_val, max_val = common.describe_error(input_tensor=output_tensor)
        error_detail_out = f"output_t nan:{has_nan} inf:{has_inf} min:{min_val} max:{max_val} "
        # Data on abs differences
        abs_diff = torch.abs(torch.subtract(output_tensor, golden_tensor))
        has_nan_diff, has_inf_diff, min_val_diff, max_val_diff = common.describe_error(input_tensor=abs_diff)
        error_detail_out += f"diff_t nan:{has_nan_diff} inf:{has_inf_diff} min:{min_val_diff} max:{max_val_diff}"
        output_errors += 1
        if output_logger:
            output_logger.error(error_detail_out)
        dnn_log_helper.log_error_detail(error_detail_out)

        return output_errors

    def compare_inference(self, output, batch_id) -> int:
        # if self.current_iteration % 8 == 0:
        #     output[3, 6] = 39304
        #     output[0, 456] = 245
        #     output[800, 666] = 9876

        # Make sure that they are on CPU
        golden_tensor = self.golden[batch_id]
        out_is_cuda, golden_is_cuda = output.is_cuda, golden_tensor.is_cuda
        if out_is_cuda or golden_is_cuda:
            dnn_log_helper.log_and_crash(
                fatal_string=f"Tensors are not on CPU. OUT IS CUDA:{out_is_cuda} GOLDEN IS CUDA:{golden_is_cuda}")

        # First check if the tensors are equal or not
        if common.equal(lhs=output, rhs=golden_tensor, threshold=self.float_threshold) is True:
            return 0

        # ------------ Check the size of the tensors
        if output.shape != golden_tensor.shape:
            info_detail = f"shape-diff g:{golden_tensor.shape} o:{output.shape}"
            if self.output_logger:
                self.output_logger.error(info_detail)
            dnn_log_helper.log_info_detail(info_detail)

        # ------------ Main check
        output_errors = self.__compare_output(output, golden_tensor, self.output_logger)

        # ------------ log and return
        if output_errors != 0:
            dnn_log_helper.log_error_count(error_count=output_errors)
        return output_errors

    def load_data_at_test(self) -> None:
        # This will save time
        if self.generate is False:
            # Save everything in the same list
            [self.golden, self.input_list] = torch.load(self.gold_path)
        else:
            # The First step is to load the inputs in the memory
            self.load_dataset()
            self.golden = list()

    def save_setup_data_to_gold_file(self) -> None:
        torch.save(
            obj=[self.golden, self.input_list,],
            f=self.gold_path
        )

    def clear_gpu_memory_and_reload(self) -> None:
        if self.output_logger:
            self.output_logger.info("RELOADING THE MODEL AND THE INPUTS AFTER ERROR")
        del self.input_list
        # Free cuda memory
        torch.cuda.empty_cache()
        self.load_data_at_test()

    @staticmethod
    def copy_to_cpu(dnn_output):
        return dnn_output.to(configs.CPU)