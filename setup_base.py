import logging
import random
from typing import Union
import argparse
import torch
import torchvision
import torchvision.datasets as tv_datasets
import torchvision.transforms as tv_transforms

import dnn_log_helper

import configs
import common


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
        self.dataset = args.dataset

        # default attributes
        self.correctness_threshold = 0.7  # Based on the whole dataset accuracy. Used only for golden generate part
        self.model = None
        self.golden = list()
        self.input_list = list()
        self.gt_targets = list()
        self.selected_samples = None

    @property
    def num_batches(self):
        return len(self.input_list)

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

    def __call__(self, batch_id, **kwargs):
        raise NotImplementedError("This base method should not be called!")

    def load_model(self) -> None:
        raise NotImplementedError("This base method should not be called!")

    def load_dataset(self) -> None:
        raise NotImplementedError("This base method should not be called!")

    def check_dnn_accuracy(self) -> None:
        raise NotImplementedError("This base method should not be called!")

    def compare_inference(self, output, batch_id) -> int:
        raise NotImplementedError("This base method should not be called!")

    def load_data_at_test(self) -> None:
        raise NotImplementedError("This base method should not be called!")

    def save_setup_data_to_gold_file(self) -> None:
        raise NotImplementedError("This base method should not be called!")

    def clear_gpu_memory_and_reload(self) -> None:
        if self.output_logger:
            self.output_logger.info("RELOADING THE MODEL AND THE INPUTS AFTER ERROR")
        del self.input_list
        del self.model
        # Free cuda memory
        torch.cuda.empty_cache()
        self.load_data_at_test()

    @staticmethod
    def copy_to_cpu(dnn_output) -> torch.tensor:
        raise NotImplementedError("This base method should not be called!")

    def post_inference_process(self, dnn_output_cpu, batch_id) -> int:
        errors = 0
        if self.generate is False:
            errors = self.compare_inference(output=dnn_output_cpu, batch_id=batch_id)
        else:
            self.golden.append(dnn_output_cpu)
        return errors


class SetupBaseClassification(SetupBase):
    def __call__(self, batch_id, **kwargs):
        return self.model(self.input_list[batch_id])

    def compare_inference(self, output, batch_id) -> int:
        golden = self.golden[batch_id]
        gt_targets = self.gt_targets[batch_id]
        # Make sure that they are on CPU
        out_is_cuda, golden_is_cuda = output.is_cuda, golden.is_cuda
        if out_is_cuda or golden_is_cuda:
            dnn_log_helper.log_and_crash(
                fatal_string=f"Tensors are not on CPU. OUT IS CUDA:{out_is_cuda} GOLDEN IS CUDA:{golden_is_cuda}")

        # First check if the tensors are equal or not
        if common.equal(lhs=golden, rhs=output, threshold=self.float_threshold) is True:
            return 0

        # ------------ Check the size of the tensors
        if output.shape != golden.shape:
            info_detail = f"shape-diff g:{golden.shape} o:{output.shape}"
            if self.output_logger:
                self.output_logger.error(info_detail)
            dnn_log_helper.log_info_detail(info_detail)

        output_errors = 0
        # Iterate over the batches
        golden_top_k_labels = common.get_top_k_labels_classification(input_tensor=golden,
                                                                     top_k=configs.CLASSIFICATION_CRITICAL_TOP_K, dim=1)
        output_top_k_labels = common.get_top_k_labels_classification(input_tensor=output,
                                                                     top_k=configs.CLASSIFICATION_CRITICAL_TOP_K, dim=1)
        for img_id, (output_batch, golden_batch, output_top_k, golden_top_k, gt_label, real_img_id) in enumerate(
                zip(output, golden, output_top_k_labels, golden_top_k_labels, gt_targets, self.selected_samples)):
            # using the same approach as the detection, compare only the positions that differ
            if common.equal(lhs=output_batch, rhs=golden_batch, threshold=self.float_threshold) is False:
                # ------------ Check if there is a Critical error ------------------------------------------------------
                err_string = f"batch:{batch_id} imgid:{img_id} rimgid:{real_img_id} "
                for i, (tpk_found, tpk_gold) in enumerate(zip(output_top_k, golden_top_k)):
                    if tpk_found != tpk_gold:
                        output_errors += 1
                        error_detail_ctr = (f"critical {err_string} "
                                            f"i:{i} "
                                            f"g:{tpk_gold} "
                                            f"o:{tpk_found} "
                                            f"gt:{gt_label}")
                        if self.output_logger:
                            self.output_logger.error(error_detail_ctr)
                        dnn_log_helper.log_error_detail(error_detail_ctr)
                # ------------ Check error on the whole output ---------------------------------------------------------
                # Not necessary to save everything, only the good info
                # Data on output tensor
                has_nan, has_inf, min_val, max_val = common.describe_error(input_tensor=output_batch)
                error_detail_out = f"{err_string} output_t nan:{has_nan} inf:{has_inf} min:{min_val} max:{max_val} "
                # Data on abs differences
                abs_diff = torch.abs(torch.subtract(golden_batch, output_batch))
                has_nan_diff, has_inf_diff, min_val_diff, max_val_diff = common.describe_error(input_tensor=abs_diff)
                error_detail_out += f"diff_t nan:{has_nan_diff} inf:{has_inf_diff} min:{min_val_diff} max:{max_val_diff}"
                output_errors += 1
                if self.output_logger:
                    self.output_logger.error(error_detail_out)
                dnn_log_helper.log_error_detail(error_detail_out)

        return output_errors

    def check_dnn_accuracy(self) -> None:
        if self.output_logger:
            self.output_logger.debug(f"Checking DNN accuracy")

        if not self.golden or not self.gt_targets:
            raise ValueError("Golden and Ground truth lists should be populated")

        # Default is classification task
        golden_top_k_labels = common.get_top_k_labels_classification(input_tensor=torch.stack(self.golden),
                                                                     top_k=configs.CLASSIFICATION_CRITICAL_TOP_K, dim=2)
        gt_top_k_labels = common.get_top_k_labels_classification(input_tensor=torch.stack(self.gt_targets),
                                                                 top_k=configs.CLASSIFICATION_CRITICAL_TOP_K, dim=2)

        gt_count = gt_top_k_labels.numel()
        correct = torch.sum(torch.eq(golden_top_k_labels, gt_top_k_labels))

        if self.output_logger:
            correctness = correct / gt_count
            self.output_logger.debug(f"Correct predicted samples:{correct} - ({correctness * 100:.2f}%)")
            if correctness < self.correctness_threshold:
                raise ValueError(f"Low accuracy {correctness * 100.0}%")

    def load_data_at_test(self):
        # This will save time
        if self.generate is False:
            # Save everything in the same list
            [self.golden, self.input_list, self.gt_targets,
             self.model, self.selected_samples] = torch.load(self.gold_path)
        else:
            # The First step is to load the inputs in the memory
            # Load the model
            self.load_model()
            self.load_dataset()
            self.golden = list()

    def save_setup_data_to_gold_file(self):
        torch.save(
            obj=[self.golden, self.input_list, self.gt_targets,
                 self.model, self.selected_samples],
            f=self.gold_path
        )

    @staticmethod
    def copy_to_cpu(dnn_output):
        return dnn_output.to(configs.CPU)


class SetupBaseImageNet(SetupBaseClassification):
    transforms: tv_transforms.Compose = None

    def load_dataset(self) -> None:
        if self.transforms is None:
            raise ValueError("First you have to set the set of transforms")

        if self.output_logger:
            self.output_logger.debug("Loading Imagenet dataset, it can take some time!")

        # Set a sampler on the CPU
        sampler_generator = torch.Generator(device=configs.CPU)
        sampler_generator.manual_seed(configs.TORCH_SEED)

        test_set = tv_datasets.imagenet.ImageNet(root=configs.IMAGENET_DATASET_DIR, transform=self.transforms,
                                                 split='val')
        subset = torch.utils.data.RandomSampler(data_source=test_set, replacement=False, num_samples=self.test_sample,
                                                generator=sampler_generator)
        test_loader = torch.utils.data.DataLoader(dataset=test_set, sampler=subset, batch_size=self.batch_size,
                                                  shuffle=False, pin_memory=True)

        # TODO: it is necessary to save which images are being loaded
        self.selected_samples = list()
        for i, (inputs, labels) in enumerate(test_loader):
            # Only the inputs must be in the device
            self.input_list.append(inputs.to("cuda:0"))
            self.gt_targets.append(labels)


class SetupBaseCIFAR(SetupBaseClassification):
    def load_dataset(self) -> None:
        transforms = tv_transforms.Compose(
            [
                tv_transforms.ToTensor(),
                tv_transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

        if self.output_logger:
            self.output_logger.debug(f"Loading {self.dataset} dataset")

        if self.dataset == configs.CIFAR10:
            test_set = torchvision.datasets.cifar.CIFAR10(root=configs.CIFAR_DATASET_DIR, download=False, train=False,
                                                          transform=transforms)
        elif self.dataset == configs.CIFAR100:
            test_set = torchvision.datasets.cifar.CIFAR100(root=configs.CIFAR_DATASET_DIR, download=False, train=False,
                                                           transform=transforms)
        else:
            raise NotImplementedError("Only CIFAR10 and CIFAR100 allowed here")
        self.selected_samples = list(range(self.test_sample))
        subset = torch.utils.data.SequentialSampler(self.selected_samples)
        test_loader = torch.utils.data.DataLoader(dataset=test_set, sampler=subset, batch_size=self.batch_size,
                                                  shuffle=False, pin_memory=True)
        self.input_list, self.gt_targets = list(), list()

        for inputs, targets in test_loader:
            # Only the inputs must be in the device
            self.input_list.append(inputs.to(configs.GPU_DEVICE))
            self.gt_targets.append(targets)
