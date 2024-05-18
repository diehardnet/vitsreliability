import argparse
import copy
import logging
import enum
import re

import timm
import torch
import configs
import dnn_log_helper
import common
from setup_base import SetupBaseImageNet


class ErrorGeometry(enum.Enum):
    SINGLE, RANDOM, LINE, BLOCK, CUBIC = [enum.auto()] * 5

    def __str__(self): return self.name

    def __repr__(self): return self


_MICRO_BENCHMARKS_DATA = dict()


def create_hook(layer_id, name):
    def hook_function_to_extract_layers(module, module_input, kwargs, module_output) -> None:
        global _MICRO_BENCHMARKS_DATA
        layer_class = module.__class__.__name__.strip()
        layer_num_parameters = sum(p.numel() for p in module.parameters())
        save_path = f"id_{1}_name_{0}_class_{layer_class}"
        save_path += f"_params_{layer_num_parameters}_output_size_{module_output.numel()}.pt"
        assert len(module_input) == 1, "Problem on module input, >1"
        input_by_output = module_output.numel()
        _MICRO_BENCHMARKS_DATA[save_path] = [
            module_input[0].clone().detach(),
            module_output.clone().detach(),
            copy.deepcopy(module),
            layer_id,
            name,
            layer_num_parameters,
            layer_class,
            kwargs
        ]

        # Get only the largest layers
        _MAX_PARAMETERS_VALUE = input_by_output

    return hook_function_to_extract_layers


def _set_hooks_in_the_layers(model: torch.nn.Module):
    handlers = list()
    for layer_id, (name, layer) in enumerate(model.named_modules()):
        if sum(p.numel() for p in layer.parameters()) != 0:
            # handler = layer.register_forward_hook(
            #     functools.partial(_hook_function_to_extract_layers, layer_id, name),
            #     with_kwargs=True
            # )
            handler = layer.register_forward_hook(create_hook(layer_id, name), with_kwargs=True)
            handlers.append(handler)

    return handlers


class SetupViTMicroBenchmarks(SetupBaseImageNet):
    def __init__(self, args: argparse.Namespace, output_logger: logging.Logger):
        super().__init__(args=args, output_logger=output_logger)
        if self.test_sample != self.batch_size:
            raise ValueError("For the micro-benchmarks only 1 batch is allowed")

    def __call__(self, batch_id, **kwargs):
        # in this benchmark each call is an iteration
        return self.model(self.input_list)

    def load_model(self) -> None:
        # only for the generation part
        if self.generate is False:
            raise ValueError("You should not call this method during the experiment")
        model = timm.create_model(self.model_name, pretrained=True)
        # Only case tested will be the FP16
        model.eval()
        model.to(configs.GPU_DEVICE)
        model.zero_grad(set_to_none=True)
        self.model = model
        self.transforms = timm.data.transforms_factory.create_transform(
            **timm.data.resolve_data_config({}, model=self.model))
        if self.precision == configs.FP16:
            class CustomToFP16:
                def __call__(self, tensor_in):
                    return tensor_in.type(torch.float16)

            self.model = self.model.half()
            self.transforms.transforms.insert(-1, CustomToFP16())

    def check_dnn_accuracy(self) -> None:
        pass  # This does nothing as it's only micro benchmarks

    def __get_error_geometry(self, output_tensor: torch.tensor) -> ErrorGeometry:
        s = output_tensor == self.golden
        return ErrorGeometry.SINGLE

    def compare_inference(self, output, batch_id) -> int:
        # Make sure that they are on CPU
        for out_or_gold, dict_data in [("out", output), ("gold", self.golden)]:
            for tensor_type, tensor in dict_data.items():
                if tensor.is_cuda:
                    dnn_log_helper.log_and_crash(
                        fatal_string=f"Tensor {out_or_gold}-{tensor_type} not on CPU:{tensor.is_cuda}")

        # First check if the tensors are equal or not
        output_errors = 0
        if common.equal(lhs=output, rhs=self.golden, threshold=self.float_threshold) is False:
            # no need to continue, we save time
            output_errors = 1
            # ----------------------------------------------------------------------------------------------------------
            # Error geometry
            error_geometry = self.__get_error_geometry(output_tensor=output, )
            error_detail_out = f"geometry: {error_geometry} "
            # ----------------------------------------------------------------------------------------------------------
            # Data on output tensor
            has_nan, has_inf, min_val, max_val = common.describe_error(input_tensor=output)
            error_detail_out += f"output_t nan:{has_nan} inf:{has_inf} min:{min_val} max:{max_val} "
            # Data on abs differences
            abs_diff = torch.abs(torch.subtract(output, self.golden))
            has_nan_diff, has_inf_diff, min_val_diff, max_val_diff = common.describe_error(input_tensor=abs_diff)
            error_detail_out += f"diff_t nan:{has_nan_diff} inf:{has_inf_diff} min:{min_val_diff} max:{max_val_diff}"
            dnn_log_helper.log_error_detail(error_detail=error_detail_out)

            # Dump the file
            log_helper_file = re.match(r".*LOCAL:(\S+).log.*", dnn_log_helper.log_file_name).group(1)
            save_file = f"{log_helper_file}_sdcit_{self.current_iteration}.pt"
            torch.save(output, save_file)
            dnn_log_helper.log_error_count(output_errors)

        return output_errors

    def __select_the_best_microbenchmark(self):
        del self.model, self.input_list, self.golden
        self.model, self.input_list, self.golden = None, None, None

        # Double check if the re-execution works
        for path_tensors in _MICRO_BENCHMARKS_DATA.keys():
            micro_data = _MICRO_BENCHMARKS_DATA[path_tensors]

            module_input, module_output, module, layer_id, name, layer_num_parameters, layer_class, kwargs = micro_data
            print(layer_id, name, layer_num_parameters, layer_class)
            output_test = module(module_input, **kwargs)
            assert module_input.is_cuda and next(module.parameters()).is_cuda
            if common.equal(lhs=output_test, rhs=module_output, threshold=0.0) is False:
                print(output_test.shape, module_output.shape)
                print("Diff", torch.sum(torch.abs(torch.subtract(output_test, module_output))))
                raise ValueError("Output tensors not equal")

        # base_key = f"id_{}_name_{}_class_{}_params_{}"
        # op_base_path = f"{micro_benchmarks_dir}/{base_key}"

    def load_data_at_test(self):
        # This will save time
        if self.generate is False:
            # Save everything in the same list
            [self.golden, self.input_list, self.model] = torch.load(self.gold_path)
        else:
            # The First step is to load the inputs in the memory
            # Load the model
            self.load_model()
            self.load_dataset()
            model_copy = copy.deepcopy(self.model)
            handlers = _set_hooks_in_the_layers(model=model_copy)
            _ = model_copy(self.input_list[0])
            torch.cuda.synchronize()

            # Release the handlers
            for handler in handlers:
                handler.remove()

            # Based on te parameters we select the best candidate
            self.__select_the_best_microbenchmark()

    def save_setup_data_to_gold_file(self):
        torch.save(
            obj=[self.golden, self.input_list, self.model],
            f=self.gold_path
        )

    @staticmethod
    def copy_to_cpu(dnn_output):
        return dnn_output.to(configs.CPU)
