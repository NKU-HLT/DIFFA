import logging

import torch

from src.models.src_step_audio.funasr_detach.models.transformer.model import Transformer
from src.models.src_step_audio.funasr_detach.register import tables


@tables.register("model_classes", "Conformer")
class Conformer(Transformer):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
