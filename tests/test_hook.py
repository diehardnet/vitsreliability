#!/usr/bin/python3
import functools

import torch
import timm
import torchvision

_micro_data = dict()


def hook_fn(base_path: str, module: torch.nn.Module, module_input: torch.tensor, module_output: torch.tensor) -> None:
    global _micro_data
    # print(module_output.shape)
    # if module.__class__.__name__ == op_name and module.layer_num == layer_num:
    save_path = f"{base_path}_output_size_{module_output.numel()}.pt"
    module_input_cpu = (md_input_i.detach() for md_input_i in module_input)
    # module_output_cpu = (md_output_i.detach().cpu() for md_output_i in module_output)
    _micro_data[save_path] = [
        module_input_cpu,
        module,
        # module_output_cpu
    ]


model_name = "vit_base_patch16_224"
model = timm.create_model(model_name, pretrained=True)
model.eval()
# Disable also parameter grads
model.zero_grad(set_to_none=True)
model = model.to("cuda:0")
config = timm.data.resolve_data_config({}, model=model)
transform = timm.data.transforms_factory.create_transform(**config)

input_sample = torch.rand(3, 600, 600)
to_pil_transform = torchvision.transforms.ToPILImage()
input_sample_pil = to_pil_transform(input_sample)

input_transformed = torch.stack([transform(input_sample_pil)], dim=0).to("cuda:0")

print(model)

for layer_id, name in enumerate(model.modules()):
    #
    # if re.match(r'.*\.attn(?:_block|_grid)?$', name):
    #     # layer.register_forward_hook(attention_module_hook_fn)
    #     attentions += 1 __class__.__name__
    class_name = name.__class__.__name__.strip()
    if class_name == "LayerNorm":
        # print(class_name)
        name.register_forward_hook(
            functools.partial(hook_fn, f"/tmp/test_layer_{layer_id}")
            )
out = model(input_transformed)
torch.cuda.synchronize()

print(out.shape)
for path, data in _micro_data.items():
    print("Saving", path)
    module_input, module = data
    for d in module_input:
        print(d.shape)

    # torch.save(data, path)
