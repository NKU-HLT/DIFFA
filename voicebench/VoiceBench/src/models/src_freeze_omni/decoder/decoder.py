import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Tuple, Optional, Union
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding
from transformers.cache_utils import DynamicCache

from src.models.src_freeze_omni.encoder.encoder import add_encoder_args
from src.models.src_freeze_omni.masks import *

IGNORE_ID = -1

class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, ignore_index=-1):
        super(CrossEntropyLoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=ignore_index)
        
    def forward(self, logits, target, target_subsampling_factor=1):
        """
        logits: B*T1*D
        target: B*T2
        """
        logits = logits[:, :target.shape[1], :]
        logits = logits.transpose(1, 2)
        target = target.to(torch.long)
        loss = self.criterion(logits, target)
        return loss

class LLM2TTSCodecAR(torch.nn.Module):
    """E2E module.

    Args:
        idim (int): dimension of inputs
        odim (int): dimension of outputs
        args (namespace): argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Extend arguments for transducer."""
        group = parser.add_argument_group("TDNN model setting")

        group.add_argument('--encoder-pre-norm-type',
                           default='ln', type=str, help="Type of input norm.")
        group.add_argument('--encoder-drop-rate', default=0.0,
                           type=float, help="Dropout rate for output.")
        group.add_argument('--encoder-criterion', default='cross-entropy',
                           type=str, help="Criterion for output")
        group.add_argument('--encoder-upsample-rate', default=1, type=int)
        group.add_argument('--kv-cache-prefix-finetune', default=0, type=int)

        group = add_encoder_args(group)

        return parser

    def __init__(self, idim, odim, args):
        """Initialize transducer modules.

        Args:
            idim (int): dimension of inputs
            odim (int): dimension of outputs
            args (Namespace): argument Namespace containing options

        """
        super(LLM2TTSCodecAR, self).__init__()
        self.idim = args.idim
        self.odim = args.odim
        self.encoder_pre_norm_type = args.encoder_pre_norm_type
        self.encoder_drop_rate = args.encoder_drop_rate
        self.encoder_criterion = args.encoder_criterion
        self.encoder_upsample_rate = args.encoder_upsample_rate
        self.reporter = None

        self.vocab_size = self.odim
        config = LlamaConfig(vocab_size=self.vocab_size + 4, hidden_size=args.transformer_attention_dim, 
                            intermediate_size=args.transformer_linear_units, 
                            num_hidden_layers=args.transformer_num_blocks, 
                            num_attention_heads=args.transformer_attention_heads, max_position_embeddings=2048, 
                            bos_token_id=self.vocab_size + 1, 
                            eos_token_id=self.vocab_size + 2, pad_token_id=self.vocab_size + 3,
                            attention_dropout=args.transformer_dropout_rate)

        self.embedding = nn.Embedding(self.vocab_size + 4, self.idim, padding_idx=self.vocab_size + 3)
        self.init_pre_nn(config)

        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        self.dropout = nn.Dropout(p=self.encoder_drop_rate)
        self.out_fnn = nn.Linear(args.encoder_output_dim, self.vocab_size + 4)

        self.kv_cache_prefix_finetune = args.kv_cache_prefix_finetune
        if self.kv_cache_prefix_finetune:
            self.init_kv_cache_prefix(config)
            self.embedding.eval()
            self.layers.eval()
            self.norm.eval()
            self.rotary_emb.eval()
            self.out_fnn.eval()
            for (name, param) in self.embedding.named_parameters():
                param.requires_grad = False
            for (name, param) in self.layers.named_parameters():
                param.requires_grad = False
            for (name, param) in self.norm.named_parameters():
                param.requires_grad = False
            for (name, param) in self.rotary_emb.named_parameters():
                param.requires_grad = False
            for (name, param) in self.out_fnn.named_parameters():
                param.requires_grad = False

        if self.encoder_criterion == 'ce':
            self.criterion = CrossEntropyLoss(ignore_index=self.vocab_size + 3)
    
    def init_kv_cache_prefix(self, config):
        self.layers_prefix = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rotary_emb_prefix = LlamaRotaryEmbedding(config=config)
    
    def kv_cache_prefix_forward(self, prefix, prefix_lens, past_key_values):
        inputs_embeds = prefix
        past_seen_tokens = 0
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + \
                                      inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb_prefix(hidden_states, position_ids)
        next_decoder_cache = None
        batch_size, max_len, _ = prefix.size()
        input_mask = torch.zeros(batch_size, max_len, max_len, dtype=torch.bool, device=prefix.device)
        for i in range(batch_size):
            input_mask[i, :prefix_lens[i], :prefix_lens[i]] = True
        attention_mask = ~(input_mask.unsqueeze(1)) * torch.finfo(inputs_embeds.dtype).min
        for decoder_layer in self.layers_prefix:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=True,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]
            next_decoder_cache = layer_outputs[1]
        past_key_values = next_decoder_cache
    
    def init_pre_nn(self, config):
        self.layers_pre_nn = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers // 2)]
        )
        self.rotary_emb_pre_nn = LlamaRotaryEmbedding(config=config)
    
    def pre_nn_forward(self, hidden, hidden_lens):
        inputs_embeds = hidden
        past_seen_tokens = 0
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + \
                                      inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb_pre_nn(hidden_states, position_ids)
        next_decoder_cache = None
        batch_size, max_len, _ = hidden.size()
        input_mask = torch.zeros(batch_size, max_len, max_len, dtype=torch.bool, device=hidden.device)
        for i in range(batch_size):
            input_mask[i, :hidden_lens[i], :hidden_lens[i]] = True
        attention_mask = ~(input_mask.unsqueeze(1)) * torch.finfo(inputs_embeds.dtype).min
        for decoder_layer in self.layers_pre_nn:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]
        return hidden_states

    def forward(self, batch):
        llm_hidden = batch['x']
        llm_hidden_lens = batch['x_lens']
        y = batch['y']
        y[y == IGNORE_ID] = self.vocab_size + 3
        y_lens = batch['y_lens']
        past_key_values = DynamicCache.from_legacy_cache(None)

        if self.kv_cache_prefix_finetune:
            self.kv_cache_prefix_forward(batch['x_prefix'], batch['x_prefix_lens'], past_key_values)

        # text_ids: (batch_size, max_len)
        batch_size, max_len = y.size()

        # Create bos, sos and eos tokens
        bos_token = torch.full((batch_size, 1), self.vocab_size, dtype=torch.long, device=y.device)
        sos_token = torch.full((batch_size, 1), self.vocab_size + 1, dtype=torch.long, device=y.device)
        eos_token = torch.full((batch_size, 1), self.vocab_size + 2, dtype=torch.long, device=y.device)
        padding_token = torch.full((batch_size, 1), self.vocab_size + 3, dtype=torch.long, device=y.device)

        # Pass through pre_nn
        llm_hidden = self.pre_nn_forward(llm_hidden, llm_hidden_lens)

        # Concat bos embedding
        bos_emb = self.embedding(bos_token)
        llm_hidden = torch.cat([bos_emb, llm_hidden], dim=1)
        llm_hidden_lens = llm_hidden_lens + 1

        # Create input x with sos token at the beginning
        x = torch.cat([sos_token, y], dim=1)  # (batch_size, max_len + 1)
        
        # Create output y with eos token at the end
        y = torch.cat([y, padding_token], dim=1)
        eos_positions = torch.arange(max_len + 1, device=y.device).expand(batch_size, max_len + 1) \
                        == y_lens.unsqueeze(1)
        y = y.masked_scatter(eos_positions, eos_token.expand_as(y)[eos_positions])

        # Embed the input sequence
        x_emb = self.embedding(x)  # (batch_size, max_len + 1, d_model)

        # compute masks
        if self.kv_cache_prefix_finetune:
            x_prefix = batch['x_prefix']
            x_prefix_lens = batch['x_prefix_lens']
            input_lens = llm_hidden.size(1) + max_len + 1
            input_mask = torch.zeros(batch_size, input_lens, x_prefix.size(1) + input_lens, \
                                     dtype=torch.bool, device=x_emb.device)
            for i in range(batch_size):
                input_mask[i, :llm_hidden_lens[i], :x_prefix_lens[i]] = True
                input_mask[i, :llm_hidden_lens[i], x_prefix.size(1): x_prefix.size(1) + llm_hidden_lens[i]] = True
                input_mask[i, llm_hidden.size(1): llm_hidden.size(1) + y_lens[i] + 1, :x_prefix_lens[i]] = True
                input_mask[i, llm_hidden.size(1): llm_hidden.size(1) + y_lens[i] + 1, \
                           x_prefix.size(1): x_prefix.size(1) + llm_hidden_lens[i]] = True
                input_mask[i, llm_hidden.size(1): llm_hidden.size(1) + y_lens[i] + 1, \
                           x_prefix.size(1) + llm_hidden.size(1): x_prefix.size(1) + \
                                                                  llm_hidden.size(1) + y_lens[i] + 1] \
                           = subsequent_mask(y_lens[i] + 1, x_emb.device)
        else:
            input_lens = llm_hidden.size(1) + max_len + 1
            input_mask = torch.zeros(batch_size, input_lens, input_lens, dtype=torch.bool, device=x_emb.device)
            for i in range(batch_size):
                input_mask[i, :llm_hidden_lens[i], :llm_hidden_lens[i]] = True
                input_mask[i, llm_hidden.size(1): llm_hidden.size(1) + y_lens[i] + 1, :llm_hidden_lens[i]] = True
                input_mask[i, llm_hidden.size(1): llm_hidden.size(1) + y_lens[i] + 1, \
                           llm_hidden.size(1): llm_hidden.size(1) + y_lens[i] + 1] \
                           = subsequent_mask(y_lens[i] + 1, x_emb.device)

        # Pass through the transformer
        inputs_embeds = torch.cat([llm_hidden, x_emb], 1)
        llm_hidden = self.dropout(llm_hidden)
        past_seen_tokens = 0
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], \
                                      device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        attention_mask = ~(input_mask.unsqueeze(1)) * torch.finfo(inputs_embeds.dtype).min
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=True,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]
        hidden_states = self.norm(hidden_states)

        encoder_out = hidden_states[:, llm_hidden.size(1):]

        # Project to vocabulary size
        logits = self.out_fnn(encoder_out)

        if self.encoder_criterion == 'ce':
            loss = self.criterion(logits, y)

        if self.training:
            self.reporter.log_loss('loss', float(loss))

        return loss
    
    def transformer_infer(self, inputs_embeds, cache_position, past_key_values):
        position_ids = cache_position.unsqueeze(0)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        next_decoder_cache = None
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=True,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]
            next_decoder_cache = layer_outputs[1]
        return hidden_states
            
    def infer(self, hidden, top_k, prefix, penalty_window_size, penalty, max_tokens=1000):
        # Pass through pre_nn
        hidden = self.pre_nn_forward(hidden, [hidden.size(1)])
        # Concat bos embedding
        bos_emb = self.embedding(torch.full((1, 1), self.vocab_size, dtype=torch.long, device=hidden.device))
        hidden = torch.cat([bos_emb, hidden], dim=1)
        # init past key values
        past_key_values = DynamicCache.from_legacy_cache(None)
        # Pass through the prefix nar decoder
        if prefix is not None and self.kv_cache_prefix_finetune:
            self.kv_cache_prefix_forward(prefix, [prefix.size(1)], past_key_values)
        inputs_embeds = hidden
        past_seen_tokens = 0
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], \
                                      device=inputs_embeds.device)
        hidden_states = self.transformer_infer(inputs_embeds, cache_position, past_key_values)

        # init generated tokens
        cur_token = torch.full((1, 1), self.vocab_size + 1, dtype=torch.long, device=hidden.device)
        generated_tokens = torch.full((1, 1), self.vocab_size + 1, dtype=torch.long, device=hidden.device)
        # generate tokens
        for i in range(max_tokens):
            inputs_embeds = self.embedding(cur_token)
            past_seen_tokens = past_key_values.get_seq_length()
            if prefix is not None:
                past_seen_tokens = past_seen_tokens - prefix.size(1)
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], \
                                          device=inputs_embeds.device)
            hidden_states = self.transformer_infer(inputs_embeds, cache_position, past_key_values)
            hidden_states = self.norm(hidden_states)

            # Project to vocabulary size
            logits = self.out_fnn(hidden_states)

            # apply penalty
            if penalty_window_size > 0:
                for token in set(generated_tokens[0][-penalty_window_size:]):
                    logits[:, :, token] /= penalty

            # top k sampling
            output = logits.squeeze(0).squeeze(0)
            probs = torch.nn.functional.softmax(output, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, top_k)
            probs = torch.zeros_like(probs).scatter_(0, top_k_indices, top_k_probs)
            probs = probs / probs.sum()
            next_token_id = torch.multinomial(probs, 1).unsqueeze(0)

            generated_tokens = torch.cat([generated_tokens, next_token_id], dim=-1)
            cur_token = next_token_id

            # eos
            if next_token_id == self.vocab_size + 2:
                break
            yield next_token_id
