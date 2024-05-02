#!/usr/bin/python3
import numpy
import torch
import torchvision.transforms
import torchvision
import pandas as pd

import configs
import torchinfo
import timm
import gc

OUTPUT_DATABASE = "../data/profile_layers.csv"


def main():
    # Disable all torch grad
    torch.set_grad_enabled(mode=False)
    # Terminal console
    dev_capability = torch.cuda.get_device_capability()
    if dev_capability[0] < configs.MINIMUM_DEVICE_CAPABILITY:
        raise ValueError(f"Device cap:{dev_capability} is too old.")

    input_sample = torch.rand(3, 600, 600)
    to_pil_transform = torchvision.transforms.ToPILImage()
    input_sample_pil = to_pil_transform(input_sample)
    data_list = list()
    for model_name in configs.VIT_CLASSIFICATION_CONFIGS:
        print(f"Profiling {model_name}")
        model = timm.create_model(model_name, pretrained=True).eval().to("cuda:0")
        config = timm.data.resolve_data_config({}, model=model)
        transform = timm.data.transforms_factory.create_transform(**config)
        out_sample = torch.stack([transform(input_sample_pil)], dim=0).to("cuda:0")
        info = torchinfo.summary(model=model, input_size=list(out_sample.shape), verbose=torchinfo.Verbosity.QUIET)
        # Freeing must be in this order
        # print(model)
        # out_sample.cpu()
        del out_sample, transform, config
        gc.collect()
        torch.cuda.empty_cache()
        for layer_info in info.summary_list:
            if layer_info.executed:
                data_list.append({
                    "net": model_name, "layer": layer_info.class_name, "var_name": layer_info.var_name,
                    "layer_params": layer_info.num_params, "depth": layer_info.depth,
                    "is_leaf": layer_info.is_leaf_layer, "mac_ops": layer_info.macs,
                    "output_size": numpy.prod(layer_info.output_size)
                })

        model.cpu()
        del model
    df = pd.DataFrame(data_list)
    df.to_csv(OUTPUT_DATABASE, index=False)


if __name__ == '__main__':
    main()
