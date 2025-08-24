from .diva import DiVAAssistant
from .qwen2 import Qwen2Assistant
from .naive import NaiveAssistant
from .mini_omni import MiniOmniAssistant
from .mini_omni2 import MiniOmni2Assistant
from .gpt4o import GPT4oAssistant, GPT4oMiniAssistant
from .naive2 import Naive2Assistant
from .naive3 import Naive3Assistant
from .naive4 import Naive4Assistant
from .moshi import MoshiAssistant
from .glm import GLMAssistant
from .ultravox import UltravoxAssistant, Ultravox0d5Assistant
from .ichigo import IchigoeAssistant
from .megrez import MegrezAssistant
from .meralion import MERaLiONAssistant
from .lyra import LyraMiniAssistant, LyraBaseAssistant
from .freeze_omni import FreezeOmniAssistant
from .minicpm import MiniCPMAssistant
from .baichuan import BaichuanOmniAssistant, BaichuanAudioAssistant
from .step_audio import StepAssistant
from .phi import PhiAssistant

model_cls_mapping = {
    'qwen2': Qwen2Assistant,
    'diva': DiVAAssistant,
    'naive': NaiveAssistant,
    'naive2': Naive2Assistant,
    'naive3': Naive3Assistant,
    'naive4': Naive4Assistant,
    'mini_omni': MiniOmniAssistant,
    'mini_omni2': MiniOmni2Assistant,
    'gpt4o': GPT4oAssistant,
    'gpt4o_mini': GPT4oMiniAssistant,
    'moshi': MoshiAssistant,
    'glm': GLMAssistant,
    'ultravox': UltravoxAssistant,
    'ultravox0_5': Ultravox0d5Assistant,
    'ichigo': IchigoeAssistant,
    'megrez': MegrezAssistant,
    'meralion': MERaLiONAssistant,
    'lyra_mini': LyraMiniAssistant,
    'lyra_base': LyraBaseAssistant,
    'freeze_omni': FreezeOmniAssistant,
    'minicpm': MiniCPMAssistant,
    'baichuan_omni': BaichuanOmniAssistant,
    'baichuan_audio': BaichuanAudioAssistant,
    'step': StepAssistant,
    'phi': PhiAssistant,
}
