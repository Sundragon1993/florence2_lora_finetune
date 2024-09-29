import numpy as np
import torch
from tqdm import tqdm
import os
import torch.nn.functional as F

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std = torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) / std)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / self.rank

    def forward(self, x):
        x = self.scaling * (x @ self.A @ self.B)
        return x


class QkvWithLoRA(torch.nn.Module):
    def __init__(self, qkv, rank, alpha, do_k_lora):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.lora_q = LoRALayer(self.dim, self.dim, rank, alpha)
        self.lora_v = LoRALayer(self.dim, self.dim, rank, alpha)
        self.do_k_lora = do_k_lora
        if do_k_lora:
            self.lora_k = LoRALayer(self.dim, self.dim, rank, alpha)

    def forward(self, x):
        qkv = self.qkv(x)
        qkv[:, :, :self.dim] += self.lora_q(x)
        qkv[:, :, -self.dim:] += self.lora_v(x)
        if self.do_k_lora:
            qkv[:, :, self.dim:2*self.dim] += self.lora_k(x)
        return qkv

class KvWithLoRA(torch.nn.Module):
    def __init__(self, kv, rank, alpha):
        super().__init__()
        self.kv = kv
        self.dim = kv.in_features
        self.lora_k = LoRALayer(self.dim, self.dim, rank, alpha)
        self.lora_v = LoRALayer(self.dim, self.dim, rank, alpha)

    def forward(self, x):
        kv = self.kv(x)
        kv[:, :, :self.dim] += self.lora_k(x)
        kv[:, :, -self.dim:] += self.lora_v(x)
        return kv

class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)