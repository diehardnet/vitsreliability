#!/usr/bin/python3

import timm
import torch
import torchvision.datasets as tv_datasets
import torchvision.transforms as tv_transforms

IMAGENET = "imagenet"
IMAGENET_DATASET_DIR = "/home/carol/ILSVRC2012"
DEVICE = "cuda:0"

CURRENT_GOLDEN_LAYERS_TENSORS = list()


def golden_layers_hook(module: torch.nn.Module, module_input: torch.tensor, module_output: torch.tensor):
    global CURRENT_GOLDEN_LAYERS_TENSORS
    # module_name = module.__class__.__name__
    CURRENT_GOLDEN_LAYERS_TENSORS.append((module_input, module_output))


def hook_fn(module: torch.nn.Module, module_input: torch.tensor, module_output: torch.tensor):
    pass


def load_dataset(batch_size: int, dataset: str, test_sample: int, transform: tv_transforms.Compose):
    # Using sequential sampler is the same as passing the shuffle=False
    # Using the RandomSampler with a fixed seed is better
    input_dataset, input_labels = list(), list()
    sampler_generator = torch.Generator(device="cpu")
    sampler_generator.manual_seed(0)
    test_set = None
    if dataset == IMAGENET:
        test_set = tv_datasets.imagenet.ImageNet(root=IMAGENET_DATASET_DIR, transform=transform,
                                                 split='val')

    subset = torch.utils.data.RandomSampler(data_source=test_set, replacement=False, num_samples=test_sample,
                                            generator=sampler_generator)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, sampler=subset, batch_size=batch_size,
                                              shuffle=False, pin_memory=True)
    for inputs, labels in test_loader:
        # Only the inputs must be in the device
        input_dataset.append(inputs.to(DEVICE))
        input_labels.append(labels)

    return input_dataset, input_labels


def main():
    model_name = "vit_base_patch16_224"
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    # Disable also parameter grads
    model.zero_grad(set_to_none=True)
    model = model.to("cuda:0")
    config = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.transforms_factory.create_transform(**config)
    input_list, input_labels = load_dataset(batch_size=16, dataset=IMAGENET, test_sample=16, transform=transform)

    # ------------------------------------------------------------------------------------------------------------------
    # First save the golden layers
    for layer_id, name in enumerate(model.modules()):
        #
        # if re.match(r'.*\.attn(?:_block|_grid)?$', name):
        #     # layer.register_forward_hook(attention_module_hook_fn)
        #     attentions += 1 __class__.__name__
        # class_name = name.__class__.__name__.strip()
        name.register_forward_hook(golden_layers_hook)
    # ------------------------------------------------------------------------------------------------------------------
    # process
    single_batch = input_list[0]
    golden_out = model(single_batch)
    torch.cuda.synchronize()
    print(golden_out.shape)
    torch.save(CURRENT_GOLDEN_LAYERS_TENSORS, f"data/test_{model_name}_layer_saving.pt")


if __name__ == '__main__':
    main()
