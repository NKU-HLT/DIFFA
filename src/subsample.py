# Modified from https://github.com/mubingshen/MLC-SLM-Baseline

# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Subsampling layer definition."""

from typing import Tuple, Union

import torch

class WhisperProjector(torch.nn.Module):
    def __init__(self, downsample_rate: int, idim: int, odim: int):
        super().__init__()
        self.ds_rate = downsample_rate
        self.idim = idim
        self.odim = odim
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(idim, idim, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv1d(idim, idim*self.ds_rate, kernel_size=3, stride=self.ds_rate, padding=1)
        )
        self.linear_connector = torch.nn.Sequential(
            torch.nn.Linear(self.idim*self.ds_rate, self.idim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.idim, self.odim),
            torch.nn.ReLU(),
        )
        self.layernorm = torch.nn.LayerNorm(self.odim)
    
    def forward(
        self,
        x: torch.Tensor
    ):
        num_frames_to_discard = x.size(1) % self.ds_rate
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        x = x.transpose(1, 2) # B, D, T
        x = self.conv(x)
        x = x.transpose(1, 2) # B, T, D
        x = self.linear_connector(x)
        x = self.layernorm(x)
        return x
