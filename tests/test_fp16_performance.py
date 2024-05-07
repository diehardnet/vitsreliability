#!/usr/bin/python3

import torch
import timm
import torchvision.transforms as tv_transforms
import torchvision.datasets as tv_datasets

import configs
import common


def load_model(model_name: str, fp16: bool) -> [torch.nn.Module, tv_transforms.Compose]:
    # The First option is the baseline option
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    # Disable also parameter grads
    model.zero_grad(set_to_none=True)
    model = model.to(configs.GPU_DEVICE)

    config = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.transforms_factory.create_transform(**config)

    if fp16:
        model = model.half()

        class CustomToFP16:
            def __call__(self, tensor_in):
                return tensor_in.type(torch.float16)

        transform.transforms.insert(-1, CustomToFP16())
    return model, transform


def load_dataset(batch_size: int, test_sample: int, transform: tv_transforms.Compose):
    # Using sequential sampler is the same as passing the shuffle=False
    # Using the RandomSampler with a fixed seed is better
    input_dataset, input_labels, original_order = list(), list(), list()
    sampler_generator = torch.Generator(device="cpu")
    sampler_generator.manual_seed(configs.TORCH_SEED)
    test_set = None
    test_set = tv_datasets.imagenet.ImageNet(root=configs.IMAGENET_DATASET_DIR, transform=transform,
                                             split='val')

    subset = torch.utils.data.RandomSampler(data_source=test_set, replacement=False, num_samples=test_sample,
                                            generator=sampler_generator)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, sampler=subset, batch_size=batch_size,
                                              shuffle=False, pin_memory=True)
    for inputs, labels in test_loader:
        # Only the inputs must be in the device
        input_dataset.append(inputs.to(configs.GPU_DEVICE))
        input_labels.append(labels)

    return input_dataset, input_labels, original_order


@torch.no_grad()
def test_precision(fp16: bool, model_name: str):
    model, transform = load_model(model_name=model_name, fp16=fp16)
    input_dataset, input_labels, original_order = load_dataset(batch_size=2, test_sample=10, transform=transform)
    timer = common.Timer()
    input_size = len(input_dataset)
    iterations = 10
    full_time = 0
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs  # in bytes
    out = model(input_dataset[0])

    for i in range(iterations):
        total_time = 0
        for batch in input_dataset:
            timer.tic()
            out = model(batch)
            torch.cuda.synchronize(device=configs.GPU_DEVICE)
            timer.toc()
            kernel_time = timer.diff_time
            total_time += kernel_time
        # print("FP16", fp16, i, f"time {total_time / input_size:.3f}")
        full_time += total_time

    del out, model, input_dataset, input_labels, original_order
    return mem / (1024 ** 2), full_time / (input_size * iterations)


def main():
    for model_name in configs.VIT_CLASSIFICATION_CONFIGS:
        mem_usage_fp16, time_fp16 = test_precision(fp16=True, model_name=model_name)
        mem_usage_fp32, time_fp32 = test_precision(fp16=False, model_name=model_name)
        print(
            f"Speedup: {time_fp32 / time_fp16:.2f},",
            f"Mem usage: fp16 {mem_usage_fp16:.3f}MB - fp32 {mem_usage_fp32:.3f}MB",
            "Model:", model_name,
        )


main()
