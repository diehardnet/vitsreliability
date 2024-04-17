#!/usr/bin/python3

import timm
import torch
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision.datasets as tv_datasets
import torchvision.transforms as tv_transforms

IMAGENET = "imagenet"
IMAGENET_DATASET_DIR = "/home/carol/ILSVRC2012"
DEVICE = "cuda:0"

#
CURRENT_GOLDEN_LAYERS_TENSORS = list()


# TODO:
#       XXFor the experiments try to compare layers in parallel with the inference
#       (not possible, check the implement sha512)
#       when an error is identified save that given layer and let it propagate
#       MAYBE save also the activations
#       implement sha512 per layer to check if we have an error in the weights or not
#       (https://github.com/akneni/hashwise)
#       https://huggingface.co/timm/vit_tiny_r_s16_p8_384.augreg_in21k_ft_in1k we may use this with Alessio work
#       Save the corrupted layer
# https://github.com/akneni/hashwise/blob/main/hashwise/cuda-libraries/sha256.cu

def golden_layers_hook(module: torch.nn.Module, module_input: torch.tensor, module_output: torch.tensor):
    global CURRENT_GOLDEN_LAYERS_TENSORS
    module_name = module.__class__.__name__
    print(module_name)
    # module_input = (mi.to_sparse() for mi in module_input)

    CURRENT_GOLDEN_LAYERS_TENSORS.append(module_input)


# def hook_fn(module: torch.nn.Module, module_input: torch.tensor, module_output: torch.tensor):
#     pass

def _is_leaf_layer(layer):
    return not "Sequential".lower() in str(layer)[:15].lower()


def is_leaf(name, layer):
    print(name)
    if name == "":
        return False
    try:
        parts = name.split('.')
        int(parts[-1])
        return False
    except:
        # layers named without any dots are avgpool and fc (hopefully)
        # so they will throw an exception on split
        # but I will add an additional test to make sure they are not numbers (should never happen)
        try:
            int(name)
            return False
        except:
            pass
        # additionally,
        # layers that end in a number (e.g., layers.0) are not leaves, but complex/sequential layers
        # so only layers that do not end in a number will throw an exception on type cast to int
        return _is_leaf_layer(layer)


# def reload_nn_final_layer(model_url, model_name, device):
#     nn = torch.hub.load(model_url, model_name, pretrained=True)
#     return nn.eval().to(device)


# def reload_nn_layers(nn, device):
#     # nn = reload_nn_final_layer(model_url, model_name, device)
#
#     layers = {}
#     excluded_layers = {}
#     for name, layer in nn.named_modules():
#         print(str(layer))
#         if name != 'ResNet' and layer.is_leaf:
#             layers[name] = name
#         else:
#             excluded_layers[name] = layer
#
#     return create_feature_extractor(nn, return_nodes=layers).to(device)


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
    model_name = "resnet18"
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    # Disable also parameter grads
    model.zero_grad(set_to_none=True)
    model = model.to("cuda:0")
    config = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.transforms_factory.create_transform(**config)
    input_list, input_labels = load_dataset(batch_size=16, dataset=IMAGENET, test_sample=16, transform=transform)

    # ------------------------------------------------------------------------------------------------------------------
    # # First save the golden layers
    for layer_id, name in enumerate(model.modules()):
        #     #
        #     # if re.match(r'.*\.attn(?:_block|_grid)?$', name):
        #     #     # layer.register_forward_hook(attention_module_hook_fn)
        #     #     attentions += 1 __class__.__name__
        #     # class_name = name.__class__.__name__.strip()
        name.register_forward_hook(golden_layers_hook)
    # ------------------------------------------------------------------------------------------------------------------
    # process
    single_batch = input_list[0]
    golden_out = model(single_batch)
    torch.cuda.synchronize()
    # golden_layers = reload_nn_layers(nn=model, device=DEVICE)
    print(golden_out.shape)
    torch.save(CURRENT_GOLDEN_LAYERS_TENSORS, f"../data/test_{model_name}_layer_saving.pt")


if __name__ == '__main__':
    main()
