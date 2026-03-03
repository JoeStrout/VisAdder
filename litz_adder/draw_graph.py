#!/usr/bin/env python3
"""Generate a torchview graph diagram of the TinyAdder model."""

import torch
from torchview import draw_graph
from tinyadder_module import TinyAdderModule

model = TinyAdderModule()
x = torch.tensor([[11, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 13,
                    0, 0, 0, 0, 0, 0, 0, 4, 5, 6, 10]])

graph = draw_graph(model, input_data=x, depth=2, save_graph=True,
                   filename="tinyadder_graph", directory="./")
print("Saved to: tinyadder_graph.png")
