# # Modified from blsp

"""SLM config"""

from transformers import PretrainedConfig, AutoConfig, WhisperConfig
from transformers import logging

logger = logging.get_logger(__name__)

class DIFFAConfig(PretrainedConfig):
    def __init__(
        self, 
        whisper_config=None, 
        llm_config=None,
        conv_kernel_sizes="5,5,5",
        adapter_inner_dim=512,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.prompt_size=64

        # wait to fix:
        self.whisper_model_id ="openai/whisper-small"
        if whisper_config is None:
            whisper_config = {}
            logger.info("whisper config is None. Initializing the WhisperConfig with default values")
        
        if llm_config is None:
            llm_config = {}
            logger.info("llama config is None. Initializing the LlamaConfig with default values")
        
        self.whisper_config = whisper_config
        self.llm_config = llm_config
        self.conv_kernel_sizes = conv_kernel_sizes
        self.adapter_inner_dim = adapter_inner_dim

        self.acoustic_connector_config={}



