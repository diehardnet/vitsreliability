#!/usr/bin/python3

import timm
import torch
from torchvision import datasets as tv_datasets

MAXVIT_LARGE_TF_384 = 'maxvit_large_tf_384.in21k_ft_in1k'
MAXVIT_LARGE_TF_512 = 'maxvit_large_tf_512.in21k_ft_in1k'
VIT_LARGE_PATCH14_CLIP_224 = "vit_large_patch14_clip_224.laion2b_ft_in12k_in1k"
SELECTED_MODEL = MAXVIT_LARGE_TF_512
# Values for VIT_LARGE_PATCH14_CLIP_224
# MAX Value found:124.64655303955078 - MIN Value found:-231.63162231445312 - BATCH SIZE 250
# MAX Value found:124.64654541015625 - MIN Value found:-231.63150024414062 - BATCH SIZE 1

# MAXVIT_LARGE_TF_512
# MAX Value found:40804.88671875
# MIN Value found:-83223.84375

# BATCH_SIZE = 250
BATCH_SIZE = 1
DEVICE = "cuda:0"
MIN_IDENTITY_VALUES = list()
MAX_IDENTITY_VALUES = list()
IMAGENET_DATASET_DIR = "/srv/tempdd/ffernand/ILSVRC2012"
TOTAL = 50000


class ProfileIdentity(torch.nn.Identity):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Keep the layer original behavior and implement smart hardening
        global MIN_IDENTITY_VALUES, MAX_IDENTITY_VALUES
        MIN_IDENTITY_VALUES.append(float(torch.min(input)))
        MAX_IDENTITY_VALUES.append(float(torch.max(input)))
        return input


def replace_identity(module, name):
    """Recursively put desired module in nn.module module."""
    # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.Identity:
            # print("replaced: ", name, attr_str)
            new_identity = ProfileIdentity()
            setattr(module, attr_str, new_identity)

    # Iterate through immediate child modules. Note, our code does the recursion no need to use named_modules()
    for name, immediate_child_module in module.named_children():
        replace_identity(immediate_child_module, name)

def get_top_k_labels(tensor: torch.tensor, top_k: int) -> torch.tensor:
    proba = torch.nn.functional.softmax(tensor, dim=1)
    return torch.topk(proba, k=top_k).indices.squeeze(0).flatten()


@torch.no_grad()
def main():
    torch.set_grad_enabled(mode=False)
    if torch.cuda.is_available() is False:
        raise ValueError(f"Device {DEVICE} not available.")

    print("Creating model")
    model = timm.create_model(SELECTED_MODEL, pretrained=True)
    replace_identity(module=model, name=SELECTED_MODEL)
    model.eval()
    # replace_identity(model, "model")
    # Disable also parameter grads
    model.zero_grad(set_to_none=True)
    model = model.to(DEVICE)
    config = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.transforms_factory.create_transform(**config)
    print("Model created, creating the test loader")

    test_set = tv_datasets.imagenet.ImageNet(root=IMAGENET_DATASET_DIR, transform=transform,
                                             split='val')

    subset = torch.utils.data.SequentialSampler(data_source=test_set)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, sampler=subset, batch_size=BATCH_SIZE, shuffle=False,
                                              pin_memory=True)
    gt = 0
    print("Test loader created, iterating over the dataset")

    for it, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE).flatten()
        output = model(inputs)
        torch.cuda.synchronize(device=DEVICE)
        top_k = get_top_k_labels(tensor=output, top_k=1)
        sum_equals =  torch.sum((top_k == labels).int())
        gt += sum_equals
        print(f"Classifying iteration:{it} of {TOTAL / BATCH_SIZE} correct ones:{sum_equals}")

    print(f"MAX Value found:{max(MAX_IDENTITY_VALUES)}")
    print(f"MIN Value found:{min(MIN_IDENTITY_VALUES)}")
    print(f"Correct ones {gt} total {TOTAL}")


if __name__ == '__main__':
    main()
