#!/usr/bin/python3
import functools
import os
import re
import typing

import pandas as pd
import torch

import configs
import configure
from setuppuretorch import load_data_at_test

_MICRO_BENCHMARKS_DATA = dict()
_MICRO_MODULES = dict()


def hook_fn(base_path: str, module: torch.nn.Module, module_input: torch.tensor, module_output: torch.tensor) -> None:
    global _MICRO_BENCHMARKS_DATA
    save_path = f"{base_path}_output_size_{module_output.numel()}.pt"
    assert len(module_input) == 1, "Problem on module input, >1"
    _MICRO_BENCHMARKS_DATA[save_path] = [
        module_input[0].clone().detach(),
        module_output.clone().detach(),
        module
    ]


def set_hooks_in_the_layers(model: torch.nn.Module, layers_to_extract_from_model: pd.DataFrame,
                            micro_benchmarks_dir: str) -> typing.List:
    layer_types = layers_to_extract_from_model['layer'].to_list()
    parameter_size = layers_to_extract_from_model['layer_params'].to_list()
    handlers = list()
    for layer_id, (name, layer) in enumerate(model.named_modules()):
        class_name = layer.__class__.__name__.strip()
        pytorch_total_params = sum(p.numel() for p in layer.parameters())
        base_key = f"id_{layer_id}_name_{name}_class_{class_name}_params_{pytorch_total_params}"
        op_base_path = f"{micro_benchmarks_dir}/{base_key}"
        if class_name in layer_types and pytorch_total_params in parameter_size:
            _MICRO_MODULES[base_key] = layer
            handler = layer.register_forward_hook(
                functools.partial(hook_fn, op_base_path)
            )
            handlers.append(handler)

    return handlers


def select_layers_to_extract(csv_path: str, models_to_evaluate: list[str]) -> pd.DataFrame:
    print("Pre-selecting the layers that will be extracted")
    df = pd.read_csv(csv_path)
    df["var_name_layer"] = df["var_name"]
    df.loc[df["layer"].str.lower().str.contains("block"), "var_name_layer"] = "Block"
    var_names = ['norm', 'attn', 'block', 'mlp', 'stage', "swiglu", "gelu", "act"]
    # filter the dataframe by var_name substrings
    df_filtered = df[df['var_name_layer'].str.lower().str.contains('|'.join(var_names))]

    # group by 'net' and 'var_name', and get the index of the row with the highest 'layer_params'
    idx = df_filtered.groupby(['net', 'var_name_layer'])['output_size'].idxmax()

    # select the rows with the highest 'layer_params' using the index
    result = df_filtered.loc[idx]
    result = result[(result["layer"] != "Identity") &
                    (result["layer"] != "Dropout") &
                    (result["layer"] != "Sequential") &
                    (result["net"].isin(models_to_evaluate))
                    ]
    result = result[~result["var_name"].isin(["norm1", "norm2", "norm_pre"])]
    return result


@torch.no_grad()
def generate_micro_operations_files(layers_to_extract: pd.DataFrame, models_to_evaluate: list[str],
                                    micro_data_dir: str, output_txt_file: str) -> None:
    print("Extracting the layers")

    device = "cuda:0"
    torch.set_grad_enabled(mode=False)
    # save the selected layers
    current_directory = os.getcwd()
    for torch_compile in configure.TORCH_COMPILE_CONFIGS:
        for dnn_model in models_to_evaluate:
            configuration_name = f"{dnn_model}_torch_compile_{torch_compile}"
            data_dir = f"{current_directory}/data"
            gold_path = f"{data_dir}/{configuration_name}.pt"
            print(f"Extracting layers for {configuration_name}")
            _, _, input_list, model, _ = load_data_at_test(gold_path=gold_path)
            model.zero_grad(set_to_none=True)
            model = model.to(device)
            model.eval()
            # Select data to extract from specific model
            layers_to_extract_from_model = layers_to_extract[layers_to_extract["net"] == dnn_model]
            # Set the path to save
            micro_benchmarks_dir = f"{micro_data_dir}/{dnn_model}"
            os.makedirs(micro_benchmarks_dir, exist_ok=True)

            handlers = set_hooks_in_the_layers(model=model, layers_to_extract_from_model=layers_to_extract_from_model,
                                               micro_benchmarks_dir=micro_benchmarks_dir)
            input_cuda = input_list[0].to(device)
            _ = model(input_cuda)
            torch.cuda.synchronize()

            # Release the handlers
            for handler in handlers:
                handler.remove()
    # Save also a list of microbenchmarks
    with open(output_txt_file, "w") as fp:
        # Saving step
        for path_tensors, key_modules in zip(_MICRO_BENCHMARKS_DATA.keys(), _MICRO_MODULES.keys()):
            print("Saving", path_tensors)
            data = _MICRO_BENCHMARKS_DATA[path_tensors]  # + [_MICRO_MODULES[key_modules]]
            torch.save(data, path_tensors)
            fp.write(path_tensors + "\n")


def generate_micro_setup_csv(output_txt_file):
    data = list()
    with open(output_txt_file) as fp:
        for line in fp:
            basename = line.strip().replace("data/microbenchmarks/", "")
            pattern = r"(\S+)/id_(\d+)_name_(\S+)_class_(\S+)_params_(\d+)_output_size_(\d+).pt"
            match = re.match(pattern, basename)
            data.append(dict(
                full_path=line.strip(),
                net=match.group(1),
                id=match.group(2),
                name=match.group(3),  # 'blocks'
                class_name=match.group(4),  # 'LayerNorm'
                params_num=match.group(5),  # '2048'
                output_size=match.group(6),  # '1052672'
            ))
    df = pd.DataFrame(data)
    df["params_num"] = df["params_num"].astype(int)
    df["output_size"] = df["output_size"].astype(int)

    # group the dataframe by 'class_name' and get the index of the row with the highest 'output_size'
    max_rows = df.groupby(['net', 'class_name'])['output_size'].idxmax()

    # use the index to retrieve the rows with the highest 'output_size' for each group
    result = df.loc[max_rows]
    result.to_csv(configure.MICROBENCHMARKS_CSV, index=False)


# Force no grad
@torch.no_grad()
def main():
    micro_data_dir = "data/microbenchmarks"
    output_txt_file = f"{micro_data_dir}_micro_list.txt"

    models_to_evaluate = [
        configs.EVA_BASE_PATCH14_448_MIM,
        configs.VIT_LARGE_PATCH14_CLIP_224
    ]

    # Select specific layers that are most resource demanding
    layers_to_extract = select_layers_to_extract(csv_path="data/profile_layers.csv",
                                                 models_to_evaluate=models_to_evaluate)

    # Generate the layers based on the data
    generate_micro_operations_files(layers_to_extract=layers_to_extract, models_to_evaluate=models_to_evaluate,
                                    micro_data_dir=micro_data_dir, output_txt_file=output_txt_file)

    # Final step, put everything on csv
    generate_micro_setup_csv(output_txt_file=output_txt_file)


if __name__ == '__main__':
    main()
