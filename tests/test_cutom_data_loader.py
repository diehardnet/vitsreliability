import torch
import torchvision.datasets as tv_datasets
from torch.utils.data import DataLoader
from torchvision import transforms as tv_transforms


def get_dataset(transforms: tv_transforms.Compose, batch_size: int):
    dataset_path = "/home/ILSVRC2012"
    test_set = tv_datasets.imagenet.ImageNet(root=dataset_path, transform=transforms,
                                             split='val')

    indices = [345, 214, 35333, 53, 2]
    indices.sort()
    print(indices)
    subset = torch.utils.data.Subset(test_set, indices)
    test_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)
    return test_loader


def main():
    batch_size = 1
    transforms = tv_transforms.Compose([
        tv_transforms.Resize(256),
        tv_transforms.CenterCrop(224),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_loader = get_dataset(transforms, batch_size)
    for inputs, labels in test_loader:
        print(labels)


if __name__ == "__main__":
    main()
