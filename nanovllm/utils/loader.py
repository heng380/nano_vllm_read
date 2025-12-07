import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():   # 遍历 safetensor 的权重名
                # print (f"{weight_name}: {f.get_tensor(weight_name).shape}")
                for k in packed_modules_mapping:   # 如果在给定的 mapping 中
                    if k in weight_name:   # 如果权重名字有 q_proj, k_proj, v_proj, gate_proj, up_proj, 需要多传一个 shard 参数把矩阵分片加载进来
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:     # 不在需要特殊处理的 mapping 中
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader) # 如果parameter 有自定义映射 weight_loader, 就用 weight_loader 函数进行加载
                    weight_loader(param, f.get_tensor(weight_name))
