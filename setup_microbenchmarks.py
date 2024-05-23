import argparse
import copy
import logging
import os
import pathlib
import re
import shutil

import timm
import torch

import common
import configs
import dnn_log_helper
from setup_base import SetupBaseImageNet

_MICRO_BENCHMARKS_DATA = [1e-10]


class LayerSaveHook:
    def __init__(self, layer_id, name, micro_op):
        self.layer_id = layer_id
        self.name = name
        self.micro_op = micro_op

    # def create_hook(layer_id, name):
    def hook_function_to_extract_layers(self, module, module_input, kwargs, module_output) -> None:
        global _MICRO_BENCHMARKS_DATA
        layer_class = module.__class__.__name__.strip()
        layer_num_parameters = sum(p.numel() for p in module.parameters())
        input_size = sum(p.numel() for p in module_input)
        micro_op = layer_num_parameters * input_size
        save_path = f"id_{1}_name_{0}_class_{layer_class}"
        save_path += f"_params_{layer_num_parameters}_output_size_{module_output.numel()}.pt"
        assert len(module_input) == 1, "Problem on module input, >1"

        # IT will select only the largest one
        if layer_class == self.micro_op and _MICRO_BENCHMARKS_DATA[-1] < micro_op:
            _MICRO_BENCHMARKS_DATA = [
                module_input[0].clone().detach(),
                module_output.clone().detach(),
                copy.deepcopy(module),
                self.layer_id,
                self.name,
                layer_num_parameters,
                layer_class,
                kwargs,
                save_path,
                micro_op,
                input_size
            ]

    def clear(self):
        del self.layer_id, self.name


class SetupViTMicroBenchmarks(SetupBaseImageNet):
    def __init__(self, args: argparse.Namespace, output_logger: logging.Logger):
        super().__init__(args=args, output_logger=output_logger)
        if self.test_sample != self.batch_size:
            raise ValueError("For the micro-benchmarks only 1 batch is allowed")

        self.micro_op = args.microop
        if self.micro_op is None:
            raise ValueError("You should pass microop argument for the micro-benchmarks")
        self.best_fit = dict()
        self.kwargs = dict()

        self.logits_path = os.path.join(pathlib.Path.home(), "micro_logits")
        self.save_logits_max_disk_usage = 0.9
        if self.save_logits:
            if self.output_logger:
                self.output_logger.debug(f"Save logits enabled, creating {self.logits_path}")
            pathlib.Path(self.logits_path).mkdir(parents=True, exist_ok=True)

    def __call__(self, batch_id, **kwargs):
        # in this benchmark each call is an iteration
        return self.model(self.input_list)

    @property
    def num_batches(self):
        return 1

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

    # def __get_error_geometry(self, output_tensor: torch.tensor) -> geometry_parser.ErrorGeometry:
    #     # FIXME: The geometry is not always correct
    #     geometry_type = geometry_parser.ErrorGeometry.MASKED
    #     golden_cpu = self.golden.to(configs.CPU)
    #     output_cpu = output_tensor.to(configs.CPU)
    #     for batch_out, batch_golden in zip(golden_cpu, output_cpu):
    #         diff_matrix = torch.not_equal(torch.abs(torch.sub(batch_out, batch_golden)), 0.0).type(torch.int)
    #         geo_batch = geometry_parser.geometry_comparison(diff=diff_matrix.numpy())
    #         # Get the largest error type
    #         if geo_batch > geometry_type:
    #             geometry_type = geo_batch
    #     return geometry_type

    def __save_logits(self, output):
        total, used, free = shutil.disk_usage("/")
        used_percent = used / total
        if used_percent > self.save_logits_max_disk_usage:
            dnn_log_helper.log_info_detail(info_detail=f"disk usage: {used_percent:.3f}, not saving the logits")
        elif self.save_logits:
            log_helper_file = re.match(r".*LOCAL:(\S+).log.*", dnn_log_helper.log_file_name).group(1)
            save_file = f"{os.path.basename(log_helper_file)}_it_{self.current_iteration}.pt"
            save_file = os.path.join(self.logits_path, save_file)
            if self.output_logger:
                self.output_logger.debug(f"Saving logits at:{save_file}")
            dnn_log_helper.log_info_detail(info_detail=f"LOGITS_AT:{save_file}")
            torch.save(output, save_file)

    def compare_inference(self, output, batch_id) -> int:
        # if self.current_iteration % 7 == 0:
        #     output[3, 23, 2] *= 34
        # Make sure that they are on CPU
        for out_or_gold, tensor in [("out", output), ("gold", self.golden)]:
            if tensor.is_cuda:
                dnn_log_helper.log_and_crash(fatal_string=f"Tensor {out_or_gold} not on CPU")

        # First check if the tensors are equal or not
        output_errors = 0
        if common.equal(lhs=output, rhs=self.golden, threshold=self.float_threshold) is False:
            # no need to continue, we save time
            output_errors = common.count_errors(lhs=output, rhs=self.golden)
            # ----------------------------------------------------------------------------------------------------------
            # Error geometry
            # error_geometry = self.__get_error_geometry(output_tensor=output)
            # error_detail_out = f"geometry:{error_geometry} "
            self.__save_logits(output=output)
            # ----------------------------------------------------------------------------------------------------------
            # Data on output tensor
            has_nan, has_inf, min_val, max_val = common.describe_error(input_tensor=output)
            error_detail_out = f"output_t nan:{has_nan} inf:{has_inf} min:{min_val} max:{max_val} "
            # Data on abs differences
            abs_diff = torch.abs(torch.subtract(output, self.golden))
            has_nan_diff, has_inf_diff, min_val_diff, max_val_diff = common.describe_error(input_tensor=abs_diff)
            error_detail_out += f"diff_t nan:{has_nan_diff} inf:{has_inf_diff} min:{min_val_diff} max:{max_val_diff}"
            dnn_log_helper.log_error_detail(error_detail=error_detail_out)

            # Dump the file
            # log_helper_file = re.match(r".*LOCAL:(\S+).log.*", dnn_log_helper.log_file_name).group(1)
            # save_file = f"{log_helper_file}_sdcit_{self.current_iteration}.pt"
            # torch.save(output, save_file)
            dnn_log_helper.log_error_count(output_errors)

        return output_errors

    def __select_the_best_microbenchmark(self):
        del self.model, self.input_list, self.golden
        self.model, self.input_list, self.golden = None, None, None

        (module_input, module_output, module, layer_id, name,
         layer_num_parameters, layer_class, kwargs, save_path, micro_op, internal_input_size) = _MICRO_BENCHMARKS_DATA
        assert internal_input_size == module_input.numel(), (f"Inputs size differs:"
                                                             f"{internal_input_size} != {module_input.numel()}")
        # max_num_mac_ops = -1e10
        # for path_tensors in _MICRO_BENCHMARKS_DATA.keys():
        #     micro_data = _MICRO_BENCHMARKS_DATA[path_tensors]
        #     module_input, module_output, module, layer_id, name, layer_num_parameters, layer_class
        #     , kwargs = micro_data
        # ------------------------------------------------------------
        # Double check if the re-execution works
        output_test = module(module_input, **kwargs)

        assert module_input.is_cuda and next(module.parameters()).is_cuda
        if common.equal(lhs=output_test, rhs=module_output, threshold=0.0) is False:
            if self.output_logger:
                self.output_logger.error(f"Diff:{torch.sum(torch.abs(torch.subtract(output_test, module_output)))}")
            raise ValueError("Output tensors not equal")
            # ------------------------------------------------------------
            # Save the best fit (largest layer)
            # mac_op = layer_num_parameters * module_input.numel()
            # if layer_class == self.micro_op and max_num_mac_ops < mac_op:
            #     max_num_mac_ops = mac_op
        self.model, self.input_list, self.golden, self.kwargs = module, module_input, module_output, kwargs
        self.best_fit = {"id": layer_id, "name": name, "num_param": layer_num_parameters, "class": layer_class}

    def _extract_internal_layers_data(self):
        model_copy = copy.deepcopy(self.model)

        handlers = list()
        for layer_id, (name, layer) in enumerate(model_copy.named_modules()):
            if sum(p.numel() for p in layer.parameters()) != 0:
                hook = LayerSaveHook(layer_id=layer_id, name=name, micro_op=self.micro_op)
                handler = layer.register_forward_hook(hook.hook_function_to_extract_layers, with_kwargs=True)
                handlers.append((handler, hook))
        # Run the actual model for the first batch
        _ = model_copy(self.input_list[0])
        torch.cuda.synchronize()
        # Release the handlers
        for handler, hook in handlers:
            hook.clear()
            handler.remove()

    def load_data_at_test(self):
        # This will save time
        if self.generate is False:
            # Save everything in the same list
            [self.golden, self.input_list, self.model, self.best_fit] = torch.load(self.gold_path)
        else:
            # The First step is to load the inputs in the memory
            # Load the model
            self.load_model()
            self.load_dataset()
            self._extract_internal_layers_data()
            # Based on te parameters we select the best candidate
            self.__select_the_best_microbenchmark()

    def save_setup_data_to_gold_file(self):
        torch.save(
            obj=[self.golden.to(configs.CPU), self.input_list, self.model, self.best_fit],
            f=self.gold_path
        )

    @staticmethod
    def copy_to_cpu(dnn_output):
        return dnn_output.to(configs.CPU)

    def post_inference_process(self, dnn_output_cpu, batch_id) -> int:
        errors = 0
        if self.generate is False:
            errors = self.compare_inference(output=dnn_output_cpu, batch_id=batch_id)

        return errors
