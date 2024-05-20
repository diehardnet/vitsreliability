import copy
import time

import timm
import torch
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

model_name = "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k"
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
input_sample_pil = transform(input_sample_pil)

input_transformed = torch.stack([input_sample_pil] * 4, dim=0).to("cuda:0")

# tic = time.time()
# for _ in range(100):
#     _ = model(input_transformed)
# time_normal = time.time() - tic

layers = {}
train_nodes, eval_nodes = get_graph_node_names(model)
print(eval_nodes)
for name, layer in model.named_modules():
    if name in eval_nodes and len(list(layer.children())) == 0:
        layers[name] = name

model_extract = create_feature_extractor(copy.deepcopy(model), return_nodes=layers).to("cuda:0")
tic = time.time()
for _ in range(100):
    _ = model_extract(input_transformed)
time_hooks = time.time() - tic

#
# print("Time hooks:", time_hooks, "Time normal:", time_normal)
#
# test = model_extract(input_transformed)
# print(test)