"""Worked example of a viscous HJB equation with a cosine terminal value."""

import torch

from .benchmarks import Benchmark


class CosineHJBBenchmark(Benchmark):
    pde = "cosine_hjb"
    T = 0.5
    sigma = 1.0
    actor_kind = "gradient"

    def __init__(self, d: int):
        self.d = d
        self.name = f"cosine_hjb_d{d}"

    def terminal(self, x):
        return x.cos().mean(dim=-1)

    def terminal_gradient(self, x):
        return -x.sin() / self.d

    def driver(self, r, q):
        return (1.0 + q.square().sum(dim=-1)).sqrt() - 1.0

    def tangent_coefficients(self, r, q):
        scale = (1.0 + q.square().sum(dim=-1)).sqrt()
        c = torch.zeros_like(scale)
        b = q / scale.unsqueeze(-1)
        h = scale.reciprocal() - 1.0
        return c, b, h
