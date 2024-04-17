#!/usr/bin/python3

import torch

module_input, module_output, module = torch.load(
    "../data/microbenchmarks/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k/"
    "id_272_name_fc_norm_class_LayerNorm_params_1536_output_size_3072.pt")


outp = module(module_input).clone().detach()


print((torch.eq(module_output.cpu(), outp.cpu())))
